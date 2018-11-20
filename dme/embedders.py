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
from six.moves import cPickle as pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dme.tasks.base import nn_init


def load_cached_pretrained_embeddings(logger, emb_path, cache_path, emb_size):
    if os.path.exists(cache_path):
        logger.info('Loading cached embeddings %s' % cache_path)
        with open(cache_path, 'rb') as f:
            pretrained_embeddings = pickle.load(f)
    else:
        logger.info('Loading embeddings %s' % cache_path)
        pretrained_embeddings = {}
        for line in codecs.open(emb_path, 'r', 'utf-8', errors='replace'):
            line = line.strip().split()
            if len(line) != emb_size + 1:
                continue
            pretrained_embeddings[line[0]] = torch.Tensor([float(i) for i in line[1:]])
        with open(cache_path, 'wb') as f:
            pickle.dump(pretrained_embeddings, f, protocol=2)
        logger.info('Cached embeddings %s' % cache_path)
    return pretrained_embeddings


def match_pretrained_embeddings(logger, embeddings, pretrained_embeddings, vocab2idx):
    logger.info('Matching embeddings')
    indices_matched, indices_missed = [], []
    count = {}
    for word, idx in vocab2idx.items():
        emb_key, match_type = get_emb_key(word, pretrained_embeddings)
        if emb_key is not None:
            embeddings[idx].copy_(pretrained_embeddings[emb_key])
            if match_type not in count:
                count[match_type] = 1
            else:
                count[match_type] += 1
            indices_matched.append(idx)
        else:
            indices_missed.append(idx)

    logger.info('Coverage %.4f (%d/%d), %s' % (float(sum(count.values())) / len(vocab2idx), sum(count.values()),
                                               len(vocab2idx), str(count)))
    return indices_matched, indices_missed


def normalize_embeddings(logger, embeddings, indices_to_normalize, indices_to_zero):
    logger.info('Normalizing embeddings')
    if len(indices_to_normalize) > 0:
        embeddings = embeddings - embeddings[indices_to_normalize, :].mean(0)
    if len(indices_to_zero) > 0:
        embeddings[indices_to_zero, :] = 0


def get_emb_key(word, embeds):
    if word in embeds:
        return word, 'exact'
    elif word.lower() in embeds:
        return word.lower(), 'lower'
    else:
        return None, ''


def get_embeds_vocab(args, cache_root, logger):
    embeds_names = '_'.join(sorted([name for name, _, _ in args.embeds]))
    cache_path = os.path.join(cache_root, 'vocab-%s.pkl' % embeds_names)

    if os.path.exists(cache_path):
        logger.info('Loading embedding vocab from cache %s' % cache_path)
        with open(cache_path, 'rb') as f:
            vocab = pickle.load(f)
    else:
        logger.info('Generating embedding vocab')
        vocab = set()
        for _, path, dim in args.embeds:
            for line in codecs.open(path, 'r', 'utf-8', errors='replace'):
                line = line.strip().split()
                if len(line) != dim + 1:
                    continue
                vocab.add(line[0])
        with open(cache_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=2)
    logger.info('Embedding vocab size: %d' % len(vocab))
    return vocab


class SingleEmbedder(nn.Module):
    def __init__(self, args, logger, emb_path, emb_size):
        super(SingleEmbedder, self).__init__()
        self.args = args

        self.embedder = nn.Embedding(len(args.vocab_stoi), emb_size, padding_idx=args.vocab_stoi['<pad>'])

        bound = math.sqrt(3.0 / emb_size)
        self.embedder.weight.data.uniform_(-1.0 * bound, bound)

        cache_path = os.path.join(args.cache_path, '%s.pkl' % emb_path.split('/')[-1])
        pretrained_embeddings = load_cached_pretrained_embeddings(logger, emb_path, cache_path, emb_size)
        indices_to_normalize, indices_to_zero = match_pretrained_embeddings(logger, self.embedder.weight.data,
                                                                            pretrained_embeddings, args.vocab_stoi)
        normalize_embeddings(logger, self.embedder.weight.data, indices_to_normalize, indices_to_zero)

        self.embedder.cuda()
        self.embedder.weight.requires_grad = False

    def forward(self, x):
        return self.embedder(x)


class CatEmbedder(nn.Module):
    def __init__(self, args, logger):
        super(CatEmbedder, self).__init__()
        assert args.attnnet == 'none'
        self.args = args
        self.n_emb = len(args.embeds)
        self.emb_sz = sum([x[2] for x in args.embeds])
        self.emb_names = sorted([name for name, _, _ in args.embeds])
        self.embedders = nn.ModuleDict({
            name: SingleEmbedder(args, logger, path, dim) for name, path, dim in args.embeds
        })
        self.dropout = nn.Dropout(p=args.emb_dropout)

    def forward(self, words):
        out = torch.cat([self.embedders[name](words) for name in self.emb_names], -1)
        if self.args.nonlin == 'relu':
            out = F.relu(out)
        if self.args.emb_dropout > 0.0:
            out = self.dropout(out)
        return out


class ProjSumEmbedder(nn.Module):
    def __init__(self, args, logger):
        super(ProjSumEmbedder, self).__init__()
        assert args.attnnet in {'none', 'no_dep_softmax', 'dep_softmax', 'no_dep_gating', 'dep_gating'}
        self.args = args
        self.n_emb = len(args.embeds)
        assert not (self.n_emb == 1 and self.args.attnnet != 'none')
        self.emb_sz = args.proj_embed_sz
        self.emb_names = sorted([name for name, _, _ in args.embeds])
        self.embedders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        for name, path, dim in args.embeds:
            self.embedders.update({name: SingleEmbedder(args, logger, path, dim)})
            self.projectors.update({name: nn.Linear(dim, args.proj_embed_sz).cuda()})
            nn_init(self.projectors[name], 'xavier')

        self.attn_0, self.attn_1 = None, None
        self.m_attn = None
        if self.n_emb > 1 and args.attnnet != 'none':
            if args.attnnet.startswith('dep_'):
                self.attn_0 = nn.LSTM(args.proj_embed_sz, 2, bidirectional=True)
                nn_init(self.attn_0, 'orthogonal')
                self.attn_1 = nn.Linear(2 * 2, 1)
                nn_init(self.attn_1, 'xavier')
            elif args.attnnet.startswith('no_dep_'):
                self.attn_0 = nn.Linear(args.proj_embed_sz, 2)
                nn_init(self.attn_0, 'xavier')
                self.attn_1 = nn.Linear(2, 1)
                nn_init(self.attn_1, 'xavier')

        self.dropout = nn.Dropout(p=args.emb_dropout)

    def forward(self, words):
        projected = [self.projectors[name](self.embedders[name](words)) for name in self.emb_names]

        if self.args.attnnet == 'none':
            out = sum(projected)
        else:
            projected_cat = torch.cat([p.unsqueeze(2) for p in projected], 2)
            s_len, b_size, _, emb_dim = projected_cat.size()
            attn_input = projected_cat

            if self.args.attnnet.startswith('dep_'):
                attn_input = attn_input.view(s_len, b_size * self.n_emb, -1)
                self.m_attn = self.attn_1(self.attn_0(attn_input)[0])
                self.m_attn = self.m_attn.view(s_len, b_size, self.n_emb)
            elif self.args.attnnet.startswith('no_dep_'):
                self.m_attn = self.attn_1(self.attn_0(attn_input)).squeeze(3)

            if self.args.attnnet.endswith('_gating'):
                self.m_attn = torch.sigmoid(self.m_attn)
            elif self.args.attnnet.endswith('_softmax'):
                self.m_attn = F.softmax(self.m_attn, dim=2)

            attended = projected_cat * self.m_attn.view(s_len, b_size, self.n_emb, 1).expand_as(projected_cat)
            out = attended.sum(2)

        if self.args.nonlin == 'relu':
            out = F.relu(out)
        if self.args.emb_dropout > 0.0:
            out = self.dropout(out)
        return out
