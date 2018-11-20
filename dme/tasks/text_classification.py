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

import torch
import torch.nn as nn

import numpy as np

from dme.encoders import SentEncoder
from dme.tasks.base import BaseTask, get_optimizer_scheduler, count_param_num, nn_init


class TextClassificationTask(BaseTask):
    def __init__(self, args, logger):
        super(TextClassificationTask, self).__init__(args, logger)

        self.model = TextClassificationModel(args, logger).cuda()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer, self.scheduler = get_optimizer_scheduler(args, self.model.parameters())

        self.epoch_train_stats = None
        self.reset_epoch_train_stats()

    def reset_epoch_train_stats(self):
        self.epoch_train_stats = {'loss': [], 'n_correct': 0, 'n_total': 0, 'acc': 0.0}

    def batch_forward(self, epoch, batch, batch_idx):
        if batch_idx == 0:
            self.reset_epoch_train_stats()

        answer = self.model(batch)
        loss = self.criterion(answer, (batch.label - 1))

        n_correct = (answer.max(1)[1].view(batch.label.size()).data == (batch.label.data - 1)).sum()
        self.epoch_train_stats['n_correct'] += float(n_correct.item())
        self.epoch_train_stats['n_total'] += batch.batch_size
        self.epoch_train_stats['loss'].append(loss.item() / batch.batch_size)

        return loss

    def report_train_stats(self, epoch):
        stats = self.get_train_stats()
        self.logger.info("Epoch %d training loss %.8f, acc: %.4f" % (epoch, stats['loss'], stats['acc']))
        return stats

    def get_train_stats(self):
        return {
            'loss': np.mean(self.epoch_train_stats['loss']),
            'acc': 100. * self.epoch_train_stats['n_correct'] / self.epoch_train_stats['n_total'],
        }

    def evaluate(self, data_iter, split, epoch):
        stats = {'n_correct': 0, 'n_total': 0, 'loss': 0.0, 'acc': 0.0}
        self.model.eval()
        data_iter.init_epoch()

        for batch_idx, batch in enumerate(data_iter):
            answer = self.model.forward(batch)
            n_correct = (torch.max(answer, 1)[1].view(batch.label.size()).data == (batch.label.data - 1)).sum()
            stats['n_correct'] += float(n_correct.item())
            stats['n_total'] += batch.batch_size
            loss = self.criterion(answer, (batch.label - 1))
            stats['loss'] += loss.item()
        stats['acc'] = 100. * stats['n_correct'] / stats['n_total']
        stats['loss'] = stats['loss'] / stats['n_total']
        self.logger.info("%s loss %.8f, acc: %.4f (%d/%d)" % (split, stats['loss'], stats['acc'], stats['n_correct'],
                                                              stats['n_total']))

        return stats['acc'], stats


class TextClassificationModel(nn.Module):
    def __init__(self, args, logger):
        super(TextClassificationModel, self).__init__()
        self.args = args
        self.encoder = SentEncoder(args, logger)
        self.classifier = nn.Sequential(
            nn.Linear(2 * args.rnn_dim, args.fc_dim),
            nn.ReLU(),
            nn.Dropout(p=args.clf_dropout),
            nn.Linear(args.fc_dim, args.fc_dim),
            nn.ReLU(),
            nn.Dropout(p=args.clf_dropout),
            nn.Linear(args.fc_dim, args.n_classes),
        )
        nn_init(self.classifier, 'xavier')

        self.report_model_size(logger)

    def forward(self, batch):
        return self.classifier(self.encoder(batch.text[0], batch.text[1]))

    def report_model_size(self, logger):
        logger.info('model size: {:,}'.format(count_param_num(self)))
