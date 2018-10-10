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
import codecs

from torchtext import data


class NLIDataset(data.TabularDataset):
    dirname = ''
    name = ''
    n_classes = 3

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, fields, root, train='', validation='', test=''):
        path = os.path.join(root, cls.name, cls.dirname)
        return super(NLIDataset, cls).splits(path, root, train, validation, test, format='json', fields=fields,
                                             filter_pred=lambda ex: ex.label != '-')

    @classmethod
    def get_fields(cls, text_field, label_field, with_genre=False):
        fields = {'label': ('label', label_field), 'sentence1': ('premise', text_field),
                  'sentence2': ('hypothesis', text_field)}
        if with_genre:
            fields['genre'] = ('genre', data.RawField())
        return fields


class SNLIDataset(NLIDataset):
    dirname = 'snli_1.0'
    name = 'snli'

    @classmethod
    def splits(cls, text_field, label_field, root, train='', validation='', test=''):
        fields = NLIDataset.get_fields(text_field, label_field, with_genre=False)
        return super(SNLIDataset, cls).splits(fields, root=root, train='snli_1.0_train.tokenized.jsonl',
                                              validation='snli_1.0_dev.tokenized.jsonl',
                                              test='snli_1.0_test.tokenized.jsonl')


class MultiNLIDataset(NLIDataset):
    dirname = 'multinli_1.0'
    name = ''

    @classmethod
    def splits(cls, text_field, label_field, root, train='', validation='', test=''):
        fields = NLIDataset.get_fields(text_field, label_field, with_genre=True)
        return super(MultiNLIDataset, cls).splits(fields, root=root, train='multinli_1.0_train.tokenized.jsonl',
                                                  validation='multinli_1.0_dev_matched.tokenized.jsonl',
                                                  test='multinli_1.0_dev_mismatched.tokenized.jsonl')


class AllNLIDataset(data.Dataset):
    n_classes = 3

    def __init__(self, paths, fields, **kwargs):
        examples = []
        for path in paths:
            with codecs.open(path, 'r', encoding='utf-8') as f:
                examples += [data.Example.fromJSON(line, fields) for line in f]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(AllNLIDataset, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, root):
        snli_train = os.path.join(root, SNLIDataset.name, SNLIDataset.dirname, 'snli_1.0_train.tokenized.jsonl')
        mnli_train = os.path.join(root, MultiNLIDataset.name, MultiNLIDataset.dirname,
                                  'multinli_1.0_train.tokenized.jsonl')
        snli_dev = os.path.join(root, SNLIDataset.name, SNLIDataset.dirname, 'snli_1.0_dev.tokenized.jsonl')
        mnli_dev = os.path.join(root, MultiNLIDataset.name, MultiNLIDataset.dirname,
                                'multinli_1.0_dev_matched.tokenized.jsonl')
        snli_test = os.path.join(root, SNLIDataset.name, SNLIDataset.dirname, 'snli_1.0_test.tokenized.jsonl')
        mnli_test = os.path.join(root, MultiNLIDataset.name, MultiNLIDataset.dirname,
                                 'multinli_1.0_dev_mismatched.tokenized.jsonl')

        fields = NLIDataset.get_fields(text_field, label_field, with_genre=False)
        train = cls((snli_train, mnli_train), fields, filter_pred=lambda ex: ex.label != '-')
        val = cls((snli_dev, mnli_dev), fields, filter_pred=lambda ex: ex.label != '-')
        test = cls((snli_test, mnli_test), fields, filter_pred=lambda ex: ex.label != '-')

        return train, val, test