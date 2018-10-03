# Copyright 2017-present Facebook. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import os

from PIL import Image

import torch
from torchtext import data
from torchtext.data import Example
import torchvision.transforms as T

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

preprocess_1c = T.Compose([
    T.Resize(size=256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std)
])

preprocess_rc = T.Compose([
    T.RandomResizedCrop(size=224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std)
])


class ImageCaptionDataset(data.Dataset):
    n_classes = None

    def __init__(self, root_path, img_dir, filename, fields, train, **kwargs):
        with open(os.path.join(root_path, filename), 'rb') as f:
            data = pickle.load(f)

        examples = []
        rand_crop = 'rand_crop' in kwargs and kwargs['rand_crop']
        self.img_transform = preprocess_rc if train and rand_crop else preprocess_1c
        self.train = train
        self.cap_field = fields['caption'][1]
        for cnt, ex in enumerate(data['captions']):
            img_id = ex['image_id']
            img_path = ex['image_path']
            examples.append({
                'image_id': img_id,
                'img_to_load': os.path.join(root_path, img_dir, img_path) if rand_crop else None,
                'img_1c_feat': torch.Tensor(data['features'][img_id]),
                'caption': ex['caption'],
                'caption_id': cnt
            })
        examples = [Example.fromdict(ex, fields) for ex in examples]
        super(ImageCaptionDataset, self).__init__(examples, fields.values())

    def __getitem__(self, i):
        ex = self.examples[i]
        image = None
        if ex.img_to_load is not None:
            image = Image.open(ex.img_to_load).convert('RGB')
        image = self.img_transform(image) if image else None
        return ex.image_id, ex.img_1c_feat, image, ex.caption_id, ex.caption

    @classmethod
    def splits(cls, text_field, root_path, img_dir, **kwargs):
        train, valid, test = 'train.pkl', 'val.pkl', 'test.pkl'
        fields = {
            'image_id': ('image_id', data.RawField()),
            'img_1c_feat': ('img_1c_feat', data.RawField()),
            'img_to_load': ('img_to_load', data.RawField()),
            'caption': ('caption', text_field),
            'caption_id': ('caption_id', data.RawField()),
        }

        train_data = None if train is None else cls(root_path, img_dir, train, fields, True, **kwargs)
        val_data = None if valid is None else cls(root_path, img_dir, valid, fields, False, **kwargs)
        test_data = None if test is None else cls(root_path, img_dir, test, fields, False, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

    def get_dataloader(self, batch_size, num_workers=4):
        return torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=self.train, pin_memory=True,
                                           num_workers=num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x[4]), reverse=True)
        image_ids, img_1c_feats, images, caption_ids, captions = zip(*batch)

        img_1c_feats = torch.stack(img_1c_feats)
        images = torch.stack(images) if images[0] is not None else None

        captions, caption_lens = self.cap_field.pad(captions)
        caption_lens = torch.Tensor(caption_lens).long()
        captions = [[self.cap_field.vocab.stoi[w] for w in caption] for caption in captions]
        captions = torch.Tensor(captions).t().long()

        return image_ids, img_1c_feats, images, caption_ids, captions, caption_lens
