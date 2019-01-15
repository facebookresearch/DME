# Dynamic Meta-Embeddings for Improved Sentence Representations
Code and models for the paper [Dynamic Meta-Embeddings for Improved Sentence Representations](https://arxiv.org/abs/1804.07983).

## Requirements

* Python 2.7 or 3.6+
* PyTorch >= 0.4.1
* torchtext >= 0.2.3
* torchvision >= 0.2.1
* Spacy >= 2.0.11
* NumPy >= 1.14.0
* jsonlines
* tqdm
* six

## Getting started

### Downloading the data
First, you should get pre-trained embeddings and pre-processed datasets in place. For embeddings, run
```bash
python get_embeddings.py --embeds fasttext,glove
```
(An example for fasttext and glove. Available embeddings are fasttext, fasttext_wiki, fasttext_opensubtitles, fasttext_torontobooks, glove, levy_bow2 and imagenet.)

For Flickr30k dataset, run
```bash
python get_flickr30k.py --flickr30k_root './data/flickr30k' --batch_size 32
```
with specified batch size for image feature extraction and Flickr30k root folder that includes `dataset_flickr30k.json` and `images` subfolder for all images.

For SNLI/MultiNLI/SST dataset, run `get_snli.py`, `get_multinli.py` and `get_sst2.py`, respectively.

The downloaded embedding and datasets will be located at `./data/embeddings` and `./data/datasets`, respectively.
### Training the models
Then, you can train the model by running `train.py`:
```bash
python train.py [arguments...]
```
Arguments are listed as follows:
```
  --name NAME           experiment name
  --task {snli,multinli,allnli,sst2,flickr30k}
                        task to train the model on
  --datasets_root DATASETS_ROOT
                        root path to dataset files
  --embeds_root EMBEDS_ROOT
                        root path to embedding files
  --savedir SAVEDIR     root path to checkpoint and caching files
  --batch_sz BATCH_SZ   minibatch size
  --clf_dropout CLF_DROPOUT
                        dropout in classifier MLP
  --early_stop_patience EARLY_STOP_PATIENCE
                        patience in early stopping criterion
  --grad_clip GRAD_CLIP
                        gradient clipping threshold
  --lr LR               learning rate
  --lr_min LR_MIN       minimal learning rate
  --lr_shrink LR_SHRINK
                        learning rate decaying factor
  --max_epochs MAX_EPOCHS
                        maximal number of epochs
  --optimizer {adam,sgd}
                        optimizer
  --resume_from RESUME_FROM
                        checkpoint file to resume training from (default is
                        the one with same experiment name)
  --scheduler_patience SCHEDULER_PATIENCE
                        patience in learning rate scheduler
  --seed SEED           random seed
  --attnnet {none,no_dep_softmax,dep_softmax,no_dep_gating,dep_gating}
                        the attention type
  --emb_dropout EMB_DROPOUT
                        the dropout in embedder
  --proj_embed_sz PROJ_EMBED_SZ
                        dimension of projected embeddings (default is the
                        smallest dimension out of all embeddings)
  --embeds EMBEDS       pre-trained embedding names
  --mixmode {cat,proj_sum}
                        method of combining embeddings
  --nonlin {none,relu}  nonlinearity in embedder
  --rnn_dim RNN_DIM     dimension of RNN sentence encoder
  --fc_dim FC_DIM       hidden layer size in classifier MLP
  --img_cropping {1c,rc}
                        image cropping method (1c/rc: center/random cropping)
                        in image caption retrieval task
  --img_feat_dim IMG_FEAT_DIM
                        image feature size in image caption retrieval task
  --margin MARGIN       margin in ranking loss for image caption retrieval
                        task
```
Here is an example for training SNLI model using fastText and glove embeddings:
```bash
python train.py --task snli \
--datasets_root data/datasets --embeds_root data/embeddings --savedir checkpoints \
--embeds fasttext,glove --mixmode proj_sum --attnnet no_dep_softmax \
--nonlin relu --rnn_dim 128 --fc_dim 128 \
--optimizer adam --lr 0.0004 --lr_min 0.00008 --batch_sz 64 --emb_dropout 0.2 --clf_dropout 0.2
```
### Allowing more types of embeddings
To allow using new types of embeddings in training, put the embedding files into `data/embeddings`. Then update the
 `embeddings` list in `dme/embeddings.py` with a new tuple per new type of embeddings. Each tuple will provide the id of the 
 embeddings, the embedding filename, the dimensionality, a description and the downloading URL (optional). 

### Pre-trained models
##### SNLI
```--batch_sz 64 --clf_dropout 0.2 --lr 0.0004 --lr_min 0.00008 --emb_dropout 0.2 --proj_embed_sz 256 --embeds fasttext,glove --rnn_dim 512 --fc_dim 1024```

[DME](https://dl.fbaipublicfiles.com/dme/snli_dme.checkpoint) (Accuracy: 86.9096%) / [CDME](https://dl.fbaipublicfiles.com/dme/snli_cdme.checkpoint) (Accuracy: 86.6042%)
##### MultiNLI
```--batch_sz 64 --clf_dropout 0.2 --lr 0.0004 --lr_min 0.00008 --emb_dropout 0.2 --proj_embed_sz 256 --embeds fasttext,glove --rnn_dim 512 --fc_dim 1024```

[DME](https://dl.fbaipublicfiles.com/dme/multinli_dme.checkpoint) (Accuracy: 74.3084%) / [CDME](https://dl.fbaipublicfiles.com/dme/multinli_cdme.checkpoint) (Accuracy: 74.7152%)
##### SST2
```--batch_sz 64 --clf_dropout 0.5 --lr 0.0004 --lr_min 0.00005 --emb_dropout 0.5 --proj_embed_sz 256 --embeds fasttext,glove --rnn_dim 512 --fc_dim 512``` 

[DME](https://dl.fbaipublicfiles.com/dme/sst2_dme.checkpoint) (Accuracy: 89.5113%) / [CDME](https://dl.fbaipublicfiles.com/dme/sst2_cdme.checkpoint) (Accuracy: 88.1933%)
##### Flickr30k
```--batch_sz 128 --clf_dropout 0.1 --early_stop_patience 5 --lr 0.0003 --lr_min 0.00005 --scheduler_patience 1 --emb_dropout 0.1 --proj_embed_sz 256 --embeds fasttext,imagenet --rnn_dim 1024 --fc_dim 512 --img_cropping rc```

[DME](https://dl.fbaipublicfiles.com/dme/flickr30k_dme.checkpoint) (Cap/Img R@1=47.3/33.12, R@10=80.9/73.44) / [CDME](https://dl.fbaipublicfiles.com/dme/flickr30k_cdme.checkpoint) (Cap/Img R@1=48.2/34.5, R@10=82.3/73.58) 

##### AllNLI
```--batch_sz 64 --clf_dropout 0.2 --lr 0.0004 --lr_min 0.00008 --emb_dropout 0.2 --proj_embed_sz 256 --embeds fasttext,glove --rnn_dim 2048 --fc_dim 1024```

[DME](https://dl.fbaipublicfiles.com/dme/allnli_dme.checkpoint) (Accuracy: 80.2757%) / [CDME](https://dl.fbaipublicfiles.com/dme/allnli_cdme.checkpoint) (Accuracy: 80.4742%)

## Reference
Please cite the following paper if you find this code useful in your research:

> D. Kiela, C. Wang, K. Cho, [*Dynamic Meta-Embeddings for Improved Sentence Representations*](https://arxiv.org/abs/1804.07983)


```
@inproceedings{kiela2018dynamic,
  title={Dynamic Meta-Embeddings for Improved Sentence Representations},
  author={Kiela, Douwe and Wang, Changhan and Cho, Kyunghyun},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  address={Brussels, Belgium},
  year={2018}
}
```

## License

This code is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 

We use [SNLI](https://nlp.stanford.edu/projects/snli/), [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/), 
[SST](https://nlp.stanford.edu/sentiment/) and [Flickr30k](http://hockenmaier.cs.illinois.edu/DenotationGraph/) datasets 
in the experiments. Please check their websites for license and citation information.


## Contact
This repo is maintained by Changhan Wang ([changhan@fb.com](mailto:changhan@fb.com)) and Douwe Kiela ([dkiela@fb.com](mailto:dkiela@fb.com)).
