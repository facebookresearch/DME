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
import argparse

from dme.embeddings import embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--embeds', type=str, default='fasttext,glove', help='embeddings to download')
args = parser.parse_known_args()[0]


def main():
    embeddings_root = os.path.join('data', 'embeddings')
    if not os.path.exists(embeddings_root):
        os.makedirs(embeddings_root)
    os.chdir(embeddings_root)

    specified_embeddings = {e.strip() for e in args.embeds.split(',')}
    for emb_id, fn, _, description, url in embeddings:
        if len(url) == 0 or emb_id not in specified_embeddings:
            continue
        if not os.path.exists(fn):
            print('Getting %s embeddings...' % description)
            os.system('wget %s' % url)
            zip_fn = url.split('/')[-1]
            if zip_fn.endswith('.zip'):
                os.system('unzip %s' % zip_fn)
                os.system('rm %s' % zip_fn)
            if zip_fn.endswith('.bz2'):
                os.system('bunzip2 %s' % zip_fn)


if __name__ == '__main__':
    main()
