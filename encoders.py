# Copyright 2017-present Facebook. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from embedders import CatEmbedder, ProjSumEmbedder
from tasks.BaseTask import nn_init


def get_embedder(args, logger):
    assert args.mixmode in {'cat', 'proj_sum'}
    if args.mixmode == 'cat':
        return CatEmbedder(args, logger)
    elif args.mixmode == 'proj_sum':
        return ProjSumEmbedder(args, logger)


class SentEncoder(nn.Module):
    def __init__(self, args, logger):
        super(SentEncoder, self).__init__()
        self.args = args
        self.embedder = get_embedder(args, logger)
        self.rnn = nn.LSTM(self.embedder.emb_sz, args.rnn_dim, bidirectional=True)
        nn_init(self.rnn, 'orthogonal')

    def forward(self, words, seq_lens):
        emb = self.embedder(words)
        seq_lens, idx_sort = torch.sort(seq_lens, descending=True)
        _, idx_unsort = torch.sort(idx_sort, descending=False)

        sorted_input = emb.index_select(1, torch.autograd.Variable(idx_sort))
        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input, seq_lens.tolist())
        packed_output = self.rnn(packed_input)[0]
        sorted_rnn_output = nn.utils.rnn.pad_packed_sequence(packed_output)[0]
        rnn_output = sorted_rnn_output.index_select(1, torch.autograd.Variable(idx_unsort))

        return rnn_output.max(0)[0]
