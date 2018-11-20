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

import copy
import math
import random

import numpy as np

import torch
from torchtext import data

from dme.args import args
from dme.logger import create_logger
from dme.tasks.base import get_lr
from dme.tasks.nli import NaturalLanguageInferenceTask
from dme.tasks.text_classification import TextClassificationTask
from dme.tasks.img_cap_retrieval import ImageCaptionRetrievalTask
from dme.embedders import get_embeds_vocab, get_emb_key
from dme.datasets.nli import SNLIDataset, MultiNLIDataset, AllNLIDataset
from dme.datasets.img_cap_retrieval import ImageCaptionDataset
from dme.datasets.sst import SST2Dataset

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def save_checkpoint(path, epoch, model, optimizer, scheduler, early_stopper, args):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'early_stopper_state_dict': early_stopper.state_dict(),
        'args': args
    }
    torch.save(checkpoint, path)


def resume_from_checkpoint(path, model, optimizer, scheduler, early_stopper):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    early_stopper.load_state_dict(checkpoint['early_stopper_state_dict'])
    return checkpoint['epoch']


def get_data_splits(args, logger, cache_root):
    logger.info('Loading data')

    emb_vocab = get_embeds_vocab(args, cache_root, logger)

    def filter_by_emb_vocab(x):
        return [w for w in x if get_emb_key(w, emb_vocab)[0] is not None]

    text_field = data.Field(include_lengths=True, init_token='<s>', eos_token='</s>', preprocessing=filter_by_emb_vocab)
    label_field = None
    if args.task in {'snli', 'multinli', 'allnli', 'sst2'}:
        label_field = data.Field(sequential=False)

    logger.info('Generating data split')
    if args.task == 'snli':
        train, dev, test = SNLIDataset.splits(text_field, label_field, args.datasets_root)
    elif args.task == 'multinli':
        train, dev, test = MultiNLIDataset.splits(text_field, label_field, args.datasets_root)
    elif args.task == 'allnli':
        train, dev, test = AllNLIDataset.splits(text_field, label_field, args.datasets_root)
    elif args.task == 'sst2':
        train, dev, test = SST2Dataset.splits(text_field, label_field, args.datasets_root)
    elif args.task == 'flickr30k':
        train, dev, test = ImageCaptionDataset.splits(text_field, args.datasets_root + '/flickr30k', 'images',
                                                      rand_crop=args.img_cropping == 'rc')

    logger.info('Building text/label vocab')
    text_field.build_vocab(train, dev, test, min_freq=1)
    if label_field is not None:
        label_field.build_vocab(train)
    return text_field, label_field, train, dev, test, emb_vocab


def get_data_iters(args, logger, train, dev, test):
    logger.info('Generating data iterator')
    if args.task in {'snli', 'multinli', 'allnli', 'sst2'}:
        train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_sz,
                                                                     device=torch.device(0), repeat=False)
        train_iter.shuffle = True
        dev_iter.shuffle = False
        test_iter.shuffle = False
    elif args.task in {'flickr30k'}:
        train_iter = train.get_dataloader(args.batch_sz, num_workers=4)
        dev_iter = dev.get_dataloader(args.batch_sz, num_workers=4)
        test_iter = test.get_dataloader(args.batch_sz, num_workers=4)
    return train_iter, dev_iter, test_iter


class EarlyStoppingCriterion(object):
    def __init__(self, patience, mode, min_delta=0.0):
        assert patience >= 0 and mode in {'min', 'max'} and min_delta >= 0.0
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self._count, self._best_score, self.is_improved = 0, None, None

    def step(self, cur_score):
        if self._best_score is None:
            self._best_score = cur_score
            return True
        else:
            if self.mode == 'max':
                self.is_improved = (cur_score >= self._best_score + self.min_delta)
            else:
                self.is_improved = (cur_score <= self._best_score - self.min_delta)

            if self.is_improved:
                self._count = 0
                self._best_score = cur_score
            else:
                self._count += 1
            return self._count <= self.patience

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def main():
    logger = create_logger('%s/%s.log' % (args.savedir, args.name))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items(), key=lambda x: x[0])))

    input_field, label_field, train, dev, test, emb_vocab = get_data_splits(args, logger, args.cache_path)
    args.vocab_stoi = input_field.vocab.stoi
    args.vocab_itos = input_field.vocab.itos
    args.tag_stoi = None if label_field is None else label_field.vocab.stoi
    args.tag_itos = None if label_field is None else label_field.vocab.itos
    args.emb_vocab = emb_vocab
    args.n_classes = train.n_classes
    train_iter, dev_iter, test_iter = get_data_iters(args, logger, train, dev, test)

    args.report_freq = int(math.ceil(len(train_iter) / 10))
    assert args.report_freq < len(train_iter)

    if args.task in {'snli', 'multinli', 'allnli'}:
        task = NaturalLanguageInferenceTask(args, logger)
    elif args.task in {'flickr30k'}:
        task = ImageCaptionRetrievalTask(args, logger)
    elif args.task in {'sst2'}:
        task = TextClassificationTask(args, logger)
    else:
        raise Exception('Unknown task')

    early_stopper = EarlyStoppingCriterion(args.early_stop_patience, mode='max')

    if args.resume_from:
        starting_epoch = resume_from_checkpoint(args.resume_from, task.model, task.optimizer, task.scheduler,
                                                early_stopper) + 1
        logger.info('resumed from %s' % args.resume_from)
    else:
        starting_epoch = 1

    for epoch in range(starting_epoch, args.max_epochs):
        logger.info('Current learning rate %.8f' % get_lr(task.optimizer))

        if hasattr(train_iter, 'init_epoch'):
            train_iter.init_epoch()
        task.model.train()
        for batch_idx, batch in enumerate(train_iter):
            task.optimizer.zero_grad()

            loss = task.batch_forward(epoch, batch, batch_idx)
            loss.backward()

            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(task.model.parameters(), args.grad_clip)

            task.optimizer.step()

            cur_train_stats = task.get_train_stats()
            if (batch_idx + 1) % args.report_freq == 0 and 'loss' in cur_train_stats:
                logger.info("[%.2f%%] Loss: %.8f" % (100. * (1 + batch_idx) / len(train_iter), cur_train_stats['loss']))

        train_stats = task.report_train_stats(epoch)

        dev_score, dev_stats = task.evaluate(dev_iter, 'valid', epoch)
        _, test_stats = task.evaluate(test_iter, 'test', epoch)
        stats = dict(
            [('train_' + k, v) for k, v in train_stats.items()] +
            [('dev_' + k, v) for k, v in dev_stats.items()] +
            [('test_' + k, v) for k, v in test_stats.items()])
        stats['epochs'] = epoch

        if early_stopper.step(dev_score):
            if early_stopper.is_improved:
                logger.info('Saved as: %s' % args.checkpoint_path)
                save_checkpoint(args.checkpoint_path, epoch, task.model, task.optimizer, task.scheduler, early_stopper,
                                args)
                best_model = copy.deepcopy(task.model.state_dict())
        else:
            logger.info('No more improvements.. reverting to best validation model')
            task.model.load_state_dict(best_model)

            if hasattr(test_iter.dataset, 'has_labels') and not test_iter.dataset.has_labels:
                _, final_stats = task.evaluate(dev_iter, 'valid', epoch)
                _, _ = task.evaluate(test_iter, 'test', epoch)
            else:
                _, final_stats = task.evaluate(test_iter, 'test', epoch)

            break

        task.scheduler.step(dev_score)

    logger.info('Finished')


if __name__ == '__main__':
    main()
