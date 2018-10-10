# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cPickle as pickle
import json
import os
import sys
from tqdm import tqdm
from PIL import Image

import torch
import torchvision

import spacy


parser = argparse.ArgumentParser()
parser.add_argument('--flickr30k_root', type=str)
parser.add_argument('--batch_size', type=int, default=16)
cfg = parser.parse_args()

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
preprocess_1c = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])


class MyImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]  # return image path


def normf(t, p=2, d=1):
    return t / t.norm(p, d, keepdim=True).expand_as(t)


def get_pil_img(root, img_name):
    img_pil = Image.open(os.path.join(root, img_name))
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    return img_pil


def main():
    model = torchvision.models.resnet152(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = torch.nn.DataParallel(model).cuda()

    dataset = MyImageFolder(cfg.flickr30k_root, preprocess_1c)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16,
                                         pin_memory=True)
    print('Extracting image features...')
    img_features = {}
    progress = tqdm(loader, mininterval=1, leave=False, file=sys.stdout)
    for (inputs, target), files in progress:
        outputs = normf(model(torch.autograd.Variable(inputs)))
        for ii, img_path in enumerate(files[0]):
            img_features[img_path.split('/')[-1]] = outputs[ii, :].detach().squeeze().cpu().numpy()

    print('Generating dataset pkls...')
    all_splits = ['train', 'val', 'test']
    captions = {s: [] for s in all_splits}
    features = {s: {} for s in all_splits}
    spacy_nlp = spacy.load('en')
    with open(os.path.join(cfg.flickr30k_root, 'dataset_flickr30k.json')) as f:
        raw_data = json.load(f)
        data = {split: [x for x in raw_data['images'] if x['split'] == split] for split in all_splits}

        for split in all_splits:
            for img_id, image_with_caption in enumerate(data[split]):
                img_name = image_with_caption['filename']
                features[split][img_id] = img_features[img_name]

                for cap in image_with_caption['sentences']:
                    d = {'image_id': img_id, 'image_path': img_name, 'caption': [t.text for t in spacy_nlp(cap['raw'])]}
                    captions[split].append(d)

    output_root = os.path.join('data', 'datasets', 'flickr30k')
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    for split in captions:
        print('Saving to disk (%s)...' % split)
        with open(os.path.join(output_root, '%s.pkl' % split), 'wb') as f:
            pickle.dump({'features': features[split], 'captions': captions[split]}, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
