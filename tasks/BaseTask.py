# Copyright 2017-present Facebook. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


def get_optimizer_scheduler(args, model_params):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model_params, lr=args.lr)
    else:
        optimizer = optim.SGD(model_params, lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=args.lr_shrink, min_lr=args.lr_min,
                                  patience=args.scheduler_patience)

    return optimizer, scheduler


def count_param_num(nn_module):
    return np.sum([np.prod(param.size()) for param in nn_module.parameters() if param.requires_grad])


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def init_weight(weight, method):
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


def nn_init(nn_module, method='xavier'):
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)


def normf(t, p=2, d=1):
    return t / t.norm(p, d, keepdim=True).expand_as(t)


class BaseTask(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.model = None
        self.criterion = None

    def reset_epoch_train_stats(self):
        pass

    def batch_forward(self, epoch, batch, batch_idx):
        pass

    def report_train_stats(self, epoch):
        pass

    def evaluate(self, data_iter, split, epoch):
        pass
