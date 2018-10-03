# Copyright 2017-present Facebook. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from tqdm import tqdm
from torchtext.datasets import SNLI
import jsonlines
import spacy


def main():
    root_path = os.path.join('data', 'datasets')
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    SNLI.download(root_path)
    dir_name = os.path.join('snli', 'snli_1.0')
    file_names = ['snli_1.0_dev', 'snli_1.0_test', 'snli_1.0_train']

    spacy_nlp = spacy.load('en')

    for file_name in file_names:
        p = os.path.join(root_path, dir_name, file_name)
        print(p)
        with jsonlines.open(p + '.jsonl', mode='r') as f_r, jsonlines.open(p + '.tokenized.jsonl', mode='w') as f_w:
            progress = tqdm(f_r, mininterval=1, leave=False)
            for line in progress:
                sentence1, sentence2 = line['sentence1'], line['sentence2']
                sentence1_spacy = ' '.join([t.text for t in spacy_nlp(sentence1)])
                sentence2_spacy = ' '.join([t.text for t in spacy_nlp(sentence2)])
                f_w.write({'sentence1': sentence1_spacy, 'sentence2': sentence2_spacy, 'label': line['gold_label']})


if __name__ == '__main__':
    main()
