embeddings = [
        ('fasttext', 'crawl-300d-2M.vec', 300, 'fastText CommonCrawl',
         'https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip'),
        ('fasttext_wiki', 'wiki.en.vec', 300, 'fastText Wiki',
         'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec'),
        ('fasttext_opensubtitles', 'opensubtitles.skipgram.defaults.vec', 300, 'fastText OpenTitles',
         'https://s3.amazonaws.com/fair-dme/opensubtitles.skipgram.defaults.vec.zip'),
        ('fasttext_torontobooks', 'torontobooks.skipgram.defaults.vec', 300, 'fastText TorontoBooks',
         'https://s3.amazonaws.com/fair-dme/torontobooks.skipgram.defaults.vec.zip'),
        ('glove', 'glove.840B.300d.txt', 300, 'glove', 'http://nlp.stanford.edu/data/glove.840B.300d.zip'),
        ('levy_bow2', 'bow2.words', 300, 'levy bow2', 'http://u.cs.biu.ac.il/~yogo/data/syntemb/bow2.words.bz2'),
        ('imagenet', 'imagenet_22k_r152.vec', 2048, 'ImageNet',
         'https://s3.amazonaws.com/fair-dme/imagenet_22k_r152.vec.zip'),
        ('specialized_lear', 'specialized_lear.vec', 300, 'specialized LEAR', ''),
        ('specialized_sentiment', 'specialized_sentiment.vec', 300, 'specialized sentiment', ''),
        ('hill_en_de', 'hill_translation_en_de.vec', 620, 'Hill En-De', ''),  # https://fh295.github.io/
        ('hill_en_fr', 'hill_translation_en_fr.vec', 620, 'Hill En-Fr', ''),  # https://fh295.github.io/
    ]
