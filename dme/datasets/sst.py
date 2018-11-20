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

from torchtext import data


class SST2Dataset(data.TabularDataset):
    name = 'sst'
    dirname = ''
    n_classes = 2

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, text_field, label_field, root, train='sentiment-train.jsonl', validation='sentiment-dev.jsonl',
               test='sentiment-test.jsonl'):
        path = os.path.join(root, cls.name, cls.dirname)
        fields = {'label': ('label', label_field), 'text': ('text', text_field)}
        return super(SST2Dataset, cls).splits(path, root, train, validation, test, format='json', fields=fields)
