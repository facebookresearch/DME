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

import os
from tqdm import tqdm

import jsonlines

import spacy


def main():
    root_path = os.path.join('data', 'datasets')
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    dir_name = 'multinli_1.0'
    url = 'https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'
    filename = url.split('/')[-1]
    os.chdir(root_path)
    os.system('wget %s' % url)
    os.system('unzip %s' % filename)
    os.system('rm %s' % filename)

    filenames = ['multinli_1.0_dev_mismatched', 'multinli_1.0_dev_matched', 'multinli_1.0_train']

    spacy_nlp = spacy.load('en')

    for filename in filenames:
        p = os.path.join(dir_name, filename)
        print(p)
        with jsonlines.open(p + '.jsonl', mode='r') as f_r, jsonlines.open(p + '.tokenized.jsonl', mode='w') as f_w:
            progress = tqdm(f_r, mininterval=1, leave=False)
            for line in progress:
                sentence1, sentence2 = line['sentence1'], line['sentence2']
                sentence1_spacy = ' '.join([t.text for t in spacy_nlp(sentence1)])
                sentence2_spacy = ' '.join([t.text for t in spacy_nlp(sentence2)])
                f_w.write({'sentence1': sentence1_spacy, 'sentence2': sentence2_spacy, 'label': line['gold_label'],
                           'genre': line['genre']})


if __name__ == '__main__':
    main()
