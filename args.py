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
from datetime import datetime
import os
import copy

from data_utils.embeddings import embeddings


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()), help='experiment name')
parser.add_argument('--task', type=str, default='snli', choices=['snli', 'multinli', 'allnli', 'sst2', 'flickr30k'],
                    help='task to train the model on')
# i/o paths
parser.add_argument('--datasets_root', type=str, default='./data/datasets', help='root path to dataset files')
parser.add_argument('--embeds_root', type=str, default='./data/embeddings', help='root path to embedding files')
parser.add_argument('--savedir', type=str, default='./saved', help='root path to checkpoint and caching files')
# optimization
parser.add_argument('--batch_sz', type=int, default=64, help='minibatch size')
parser.add_argument('--clf_dropout', type=float, default=0.5, help='dropout in classifier MLP')
parser.add_argument('--early_stop_patience', type=int, default=3, help='patience in early stopping criterion')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping threshold')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_min', type=float, default=1e-5, help='minimal learning rate')
parser.add_argument('--lr_shrink', type=float, default=2e-1, help='learning rate decaying factor')
parser.add_argument('--max_epochs', type=int, default=100, help='maximal number of epochs')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer')
parser.add_argument('--resume_from', type=str, default='__default_checkpoint__',
                    help='checkpoint file to resume training from (default is the one with same experiment name)')
parser.add_argument('--scheduler_patience', type=int, default=0, help='patience in learning rate scheduler')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
# embedder
parser.add_argument('--attnnet', type=str, default='none', choices=['none', 'no_dep_softmax', 'dep_softmax',
                                                                    'no_dep_gating', 'dep_gating'],
                    help='the attention type')
parser.add_argument('--emb_dropout', type=float, default=0.0, help='the dropout in embedder')
parser.add_argument('--proj_embed_sz', type=int, default=-1,
                    help='dimension of projected embeddings (default is the smallest dimension out of all embeddings)')
parser.add_argument('--embeds', type=str, default='fasttext2,glove', help='pre-trained embedding names')
parser.add_argument('--mixmode', type=str, default='proj_sum', choices=['cat', 'proj_sum'],
                    help='method of combining embeddings')
parser.add_argument('--nonlin', type=str, default='relu', choices=['none', 'relu'], help='nonlinearity in embedder')
# encoder
parser.add_argument('--rnn_dim', type=int, default=1024, help='dimension of RNN sentence encoder')
# classifier
parser.add_argument('--fc_dim', type=int, default=512, help='hidden layer size in classifier MLP')
# img cap retrieval
parser.add_argument('--img_cropping', type=str, default='1c', choices=['1c', 'rc'],
                    help='image cropping method (1c/rc: center/random cropping) in image caption retrieval task')
parser.add_argument('--img_feat_dim', type=int, default=2048, help='image feature size in image caption retrieval task')
parser.add_argument('--margin', type=float, default=0.2, help='margin in ranking loss for image caption retrieval task')

raw_args = parser.parse_known_args()[0]
args = copy.deepcopy(raw_args)


def get_available_embeddings(args):
    return [(emb_id, os.path.join(args.embeds_root, fn), dim) for emb_id, fn, dim, description, url in embeddings]


def preprocess(opt):
    assert os.path.exists(args.savedir)
    opt.cache_path = os.path.join(opt.savedir, 'cache')
    if not os.path.exists(opt.cache_path):
        os.makedirs(opt.cache_path)
    opt.checkpoint_path = os.path.join(opt.savedir, opt.name) + '.checkpoint'

    if opt.resume_from == '__default_checkpoint__' and os.path.exists(opt.checkpoint_path):
        opt.resume_from = opt.checkpoint_path
    elif os.path.exists(opt.resume_from):
        pass
    else:
        opt.resume_from = None

    available_embeds = get_available_embeddings(opt)
    selected_embeds = opt.embeds.split(',') if ',' in opt.embeds else [opt.embeds]
    assert all([(x in [y[0] for y in available_embeds]) for x in selected_embeds])
    opt.embeds = [x for x in available_embeds if x[0] in selected_embeds]

    if opt.proj_embed_sz < 0:
        opt.proj_embed_sz = min([x[2] for x in opt.embeds])


preprocess(args)
