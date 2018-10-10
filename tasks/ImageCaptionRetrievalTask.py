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

from PIL import ImageFile

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models
from torch.autograd import Variable

from encoders import SentEncoder
from tasks.BaseTask import BaseTask, get_optimizer_scheduler, count_param_num, normf, nn_init

ImageFile.LOAD_TRUNCATED_IMAGES = True
cudnn.benchmark = True


class ImageCaptionRetrievalTask(BaseTask):
    def __init__(self, args, logger):
        super(ImageCaptionRetrievalTask, self).__init__(args, logger)

        self.args = args
        self.logger = logger

        self.model = ImageCaptionRetrievalModel(args, logger).cuda()
        self.criterion = VseLoss(args.margin)
        self.ranker = ImageCaptionRetrievalRanker(args, logger)

        self.optimizer, self.scheduler = get_optimizer_scheduler(args, self.model.parameters())

        self.epoch_train_stats = None
        self.reset_epoch_train_stats()

    def reset_epoch_train_stats(self):
        self.epoch_train_stats = {'loss': []}

    def batch_forward(self, epoch, batch, batch_idx):
        img_ids, img_1c_feats, images, cap_ids, caps, cap_lens = batch
        batch_size = len(img_ids)
        if batch_idx == 0:
            self.reset_epoch_train_stats()

        cap_features = self.model.cap_encoder(caps.cuda(), cap_lens.cuda())
        img_batch = images if self.args.img_cropping == 'rc' else img_1c_feats
        img_features = self.model.img_encoder(img_batch.cuda())

        loss = self.criterion(cap_features, img_features)
        self.epoch_train_stats['loss'].append(loss.item() / batch_size)
        return loss

    def report_train_stats(self, epoch):
        stats = self.get_train_stats()
        self.logger.info("Epoch %d training loss %.8f" % (epoch, stats['loss']))
        return stats

    def get_train_stats(self):
        return {'loss': np.mean(self.epoch_train_stats['loss'])}

    def evaluate(self, data_iter, split, epoch):
        stats = {
            'loss': 0.0,
            'i_1k_r1': 0.0, 'i_1k_r5': 0.0, 'i_1k_r10': 0.0, 'i_1k_mr': 0.0, 'i_1k_mrr': 0.0, 'i_1k_med_r': 0.0,
            'c_1k_r1': 0.0, 'c_1k_r5': 0.0, 'c_1k_r10': 0.0, 'c_1k_mr': 0.0, 'c_1k_mrr': 0.0, 'c_1k_med_r': 0.0,
            'i_r1': 0.0, 'i_r5': 0.0, 'i_r10': 0.0, 'i_mr': 0.0, 'i_mrr': 0.0, 'i_med_r': 0.0,
            'c_r1': 0.0, 'c_r5': 0.0, 'c_r10': 0.0, 'c_mr': 0.0, 'c_mrr': 0.0, 'c_med_r': 0.0,
        }

        self.model.eval()

        stats['loss'] = self.ranker.init(data_iter, self.model, self.criterion)
        self.logger.info("%s loss %.8f" % (split, stats['loss']))

        ranking_stats = self.ranker.rank(split, rank_1k=self.args.task == 'coco')
        for k, v in ranking_stats.items():
            stats[k] = v

        self.ranker.clear()

        score = stats['i_r1'] + stats['i_r5'] + stats['i_r10'] + stats['c_r1'] + stats['c_r5'] + stats['c_r10']
        return score, stats


class ImageCaptionRetrievalModel(nn.Module):
    def __init__(self, args, logger):
        super(ImageCaptionRetrievalModel, self).__init__()
        self.cap_encoder = CaptionEncoder(args, logger).cuda()
        self.img_encoder = ImageEncoder(args) if args.img_cropping == 'rc' else ImageFeatureEncoder(args)
        self.img_encoder.cuda()

        self.report_model_size(logger)

    def report_model_size(self, logger):
        logger.info('model size: {:,}'.format(count_param_num(self)))


class VseLoss(nn.Module):
    def __init__(self, margin):
        super(VseLoss, self).__init__()
        self.margin = margin

    def forward(self, normalized_images, normalized_captions):
        assert normalized_images.size(0) == normalized_captions.size(0)
        batch_size = normalized_images.size(0)
        scores = torch.mm(normalized_images, normalized_captions.t())
        diagonal = scores.diag().view(batch_size, 1)
        pos_caption_scores = diagonal.expand_as(scores)
        pos_image_scores = diagonal.t().expand_as(scores)

        cost_caption = (self.margin + scores - pos_caption_scores).clamp(min=0)
        cost_image = (self.margin + scores - pos_image_scores).clamp(min=0)

        mask = Variable(torch.eye(batch_size) > .5).cuda()
        cost_caption = cost_caption.masked_fill_(mask, 0.0)
        cost_image = cost_image.masked_fill_(mask, 0.0)

        return cost_caption.max(1)[0].sum() + cost_image.max(0)[0].sum()


class CaptionEncoder(nn.Module):
    def __init__(self, args, logger):
        super(CaptionEncoder, self).__init__()
        self.encoder = SentEncoder(args, logger)

    def forward(self, words, seq_lens):
        return normf(self.encoder(words, seq_lens))


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        resnet = torchvision.models.resnet152(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        for p in resnet.parameters():
            p.requires_grad = False
        self.resnet = nn.DataParallel(resnet).cuda()
        self.projector = nn.Linear(args.img_feat_dim, 2 * args.rnn_dim)
        nn_init(self.projector, 'xavier')

    def forward(self, images):
        features = normf(self.resnet(images).squeeze())
        return normf(self.projector(features))


class ImageFeatureEncoder(nn.Module):
    def __init__(self, args):
        super(ImageFeatureEncoder, self).__init__()
        self.projector = nn.Linear(args.img_feat_dim, 2 * args.rnn_dim)
        nn_init(self.projector, 'xavier')

    def forward(self, features):
        return normf(self.projector(normf(features)))


class ImageCaptionRetrievalRanker(nn.Module):
    def __init__(self, args, logger):
        super(ImageCaptionRetrievalRanker, self).__init__()
        self.args = args
        self.logger = logger

        self.inited = False
        self.img_features, self.cap_features = [], []
        self.cap_ix_to_img_ix, self.img_ix_to_cap_ix = {}, {}

    def init(self, data_iter, model, criterion):
        img_count, cap_count = 0, 0
        self.img_features, self.cap_features = [], []
        self.cap_ix_to_img_ix, self.img_ix_to_cap_ix = {}, {}
        image_ids_to_ix = {}
        loss = 0

        for batch in data_iter:
            img_ids, img_1c_feats, images, cap_ids, caps, cap_lens = batch
            cur_cap_features = model.cap_encoder(caps.cuda(), cap_lens.cuda()).data

            # assume pre-computed img features are l2-normalized
            cur_img_features = model.img_encoder.projector(img_1c_feats.cuda()).data
            cur_img_features = normf(cur_img_features)
            for i, (image_id, caption_id) in enumerate(zip(img_ids, cap_ids)):
                if image_id not in image_ids_to_ix:
                    image_ids_to_ix[image_id] = img_count
                    self.img_ix_to_cap_ix[img_count] = []
                    self.img_features.append(cur_img_features[i, :])
                    img_count += 1
                img_ix = image_ids_to_ix[image_id]
                self.img_ix_to_cap_ix[img_ix].append(cap_count)
                self.cap_ix_to_img_ix[cap_count] = img_ix
                cap_count += 1

            self.cap_features.append(cur_cap_features)
            loss += criterion(cur_img_features, cur_cap_features)

        self.img_features = torch.cat(self.img_features, 0).view(img_count, -1)
        self.cap_features = torch.cat(self.cap_features, 0).view(cap_count, -1)

        self.inited = True
        return float(loss.cpu().numpy()) / cap_count

    def clear(self):
        self.inited = False
        self.img_features, self.cap_features = [], []
        self.cap_ix_to_img_ix, self.img_ix_to_cap_ix = {}, {}

    @staticmethod
    def get_rank_stats(rank):
        r1, r5, r10 = [np.mean(rank <= r) * 100 for r in [1, 5, 10]]
        mrr = np.mean(rank ** -1)
        mr = np.mean(rank)
        med_r = np.median(rank)
        return {'r1': r1, 'r5': r5, 'r10': r10, 'mrr': mrr, 'mr': mr, 'med_r': med_r}

    def rank_i2c(self, img_features, cap_features, img_ix_to_cap_ix):
        scores = torch.mm(img_features, cap_features.t()).cpu()
        _, indices = torch.sort(scores, 1, descending=True)

        rank = np.zeros(img_features.size(0))
        for j in range(img_features.size(0)):
            for k in range(cap_features.size(0)):
                if indices[j, k] in img_ix_to_cap_ix[j][:5]:
                    rank[j] = k + 1
                    break

        return self.get_rank_stats(rank)

    def rank_c2i(self, img_features, cap_features, cap_ix_to_img_ix):
        scores = torch.mm(cap_features, img_features.t()).cpu()
        _, indices = torch.sort(scores, 1, descending=True)

        rank = np.zeros(cap_features.size(0))
        for j in range(cap_features.size(0)):
            for k in range(img_features.size(0)):
                if indices[j, k] == cap_ix_to_img_ix[j]:
                    rank[j] = k + 1
                    break

        return self.get_rank_stats(rank)

    def rank(self, split, rank_1k=False):
        if not self.inited:
            raise Exception('ranker is not initialized')

        stats = {
            'i_1k_r1': 0.0, 'i_1k_r5': 0.0, 'i_1k_r10': 0.0, 'i_1k_mr': 0.0, 'i_1k_mrr': 0.0, 'i_1k_med_r': 0.0,
            'c_1k_r1': 0.0, 'c_1k_r5': 0.0, 'c_1k_r10': 0.0, 'c_1k_mr': 0.0, 'c_1k_mrr': 0.0, 'c_1k_med_r': 0.0,
            'i_r1': 0.0, 'i_r5': 0.0, 'i_r10': 0.0, 'i_mr': 0.0, 'i_mrr': 0.0, 'i_med_r': 0.0,
            'c_r1': 0.0, 'c_r5': 0.0, 'c_r10': 0.0, 'c_mr': 0.0, 'c_mrr': 0.0, 'c_med_r': 0.0,
        }

        # 1k
        if rank_1k:
            batch_num = 5
            img_batch_size = int(self.img_features.size(0) / batch_num)
            for i in range(batch_num):
                img_batch = range(img_batch_size * i, img_batch_size * (i + 1))
                cap_batch = []
                for img_ix in img_batch:
                    cap_batch.extend(self.img_ix_to_cap_ix[img_ix])
                cap_batch = sorted(cap_batch)

                cap_ix_map = {cap_ix: local_ix for local_ix, cap_ix in enumerate(cap_batch)}
                i_offset = img_batch_size * i

                local_img_ix_to_cap_ix = {i - i_offset: [cap_ix_map[c] for c in self.img_ix_to_cap_ix[i]] for i in img_batch}
                local_cap_ix_to_img_ix = {cap_ix_map[c]: self.cap_ix_to_img_ix[c] - i_offset for c in cap_batch}

                ranking_scores = self.rank_i2c(self.img_features[torch.LongTensor(img_batch).cuda()],
                                               self.cap_features[torch.LongTensor(cap_batch).cuda()],
                                               local_img_ix_to_cap_ix)
                for k, v in ranking_scores.items():
                    stats['c_1k_%s' % k] += v

                ranking_scores = self.rank_c2i(self.img_features[torch.LongTensor(img_batch).cuda()],
                                               self.cap_features[torch.LongTensor(cap_batch).cuda()],
                                               local_cap_ix_to_img_ix)
                for k, v in ranking_scores.items():
                    stats['i_1k_%s' % k] += v

            for k in ranking_scores.keys():
                stats['c_1k_%s' % k] /= batch_num
                stats['i_1k_%s' % k] /= batch_num

            for which in ['caption', 'image']:
                w = which[0]
                scores = (split, which, stats['%s_1k_r1' % w], stats['%s_1k_r5' % w], stats['%s_1k_r10' % w],
                          stats['%s_1k_mrr' % w], stats['%s_1k_mr' % w], stats['%s_1k_med_r' % w])
                self.logger.info('1k %s %s R@1=%.4f R@5=%.4f R@10=%.4f MRR=%.4f MR=%.4f MedR=%.4f' % scores)

        # 5k
        ranking_scores = self.rank_i2c(self.img_features, self.cap_features, self.img_ix_to_cap_ix)
        for k, v in ranking_scores.items():
            stats['c_%s' % k] = v

        ranking_scores = self.rank_c2i(self.img_features, self.cap_features, self.cap_ix_to_img_ix)
        for k, v in ranking_scores.items():
            stats['i_%s' % k] = v

        for which in ['caption', 'image']:
            w = which[0]
            scores = (split, which, stats['%s_r1' % w], stats['%s_r5' % w], stats['%s_r10' % w], stats['%s_mrr' % w],
                      stats['%s_mr' % w], stats['%s_med_r' % w])
            self.logger.info('%s %s R@1=%.4f R@5=%.4f R@10=%.4f MRR=%.4f MR=%.4f MedR=%.4f' % scores)

        return stats
