# Copyright 2017-present Facebook. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import codecs

import jsonlines
from tqdm import tqdm

import spacy


def main():
    root_path = os.path.join('data', 'datasets', 'sst')
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    os.chdir(root_path)

    base_url = 'https://raw.githubusercontent.com/PrincetonML/SIF/master/data'
    filenames = ['sentiment-train', 'sentiment-dev', 'sentiment-test']
    for filename in filenames:
        if not os.path.exists(filename):
            os.system('wget %s' % (base_url + '/' + filename))

    spacy_nlp = spacy.load('en')

    for filename in filenames:
        print(filename)
        with codecs.open(filename, 'r', 'utf-8') as f, jsonlines.open(filename + '.jsonl', mode='w') as writer:
            lines = [l for l in f]
            progress = tqdm(lines, mininterval=1, leave=False)
            for line in progress:
                text, label = line.strip().split('\t')
                text_spacy = ' '.join([t.text for t in spacy_nlp(text)])
                writer.write({'text': text_spacy, 'label': label})


if __name__ == '__main__':
    main()
