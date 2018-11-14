import os
import time
import numpy as np
import pandas as pd

from gensim import models, corpora
from gensim.models import LdaModel
from gensim.test.utils import datapath
from gensim.corpora import Dictionary, MmCorpus


def train_lda(data, n_topics, modelname):
    """
    This function trains the lda model. You may select number of topics and play
    with other parameters such as chunksize, dictionary filtering, LDA model alpha and eta.
    """
    num_topics = n_topics
    chunksize = 300
    dictionary = corpora.Dictionary(data['processed'])

    # Possible filtering here:
    # dictionary.filter_extremes(no_below=5, no_above=0.8, keep_n=100000)  

    corpus = [dictionary.doc2bow(doc) for doc in data['processed']]

    print("Training model...")
    t1 = time.time()

    # Small alpha means each document is only represented by a small number of topics, and vice versa
    # Small eta means each topic is only represented by a small number of words, and vice versa

    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
    t2 = time.time()
    print("Time to train LDA model on ", len(data), "articles: ", (t2-t1)/60, "min")

    print("Saving model files...")

    # Saving model, dictionary, and corpus for visualisation and future use
    if not os.path.exists(r'./models/' + modelname):
        os.makedirs(r'./models/' + modelname)
    lda.save(r'./models/' + modelname + '/' + modelname + '.model')
    MmCorpus.serialize(r'./models/' + modelname + '/' + modelname + '.mm', corpus)
    dictionary.save(r'./models/' + modelname + '/' + modelname + '.dict')
    
    print("Model saved in models folder.")
    
    return dictionary,corpus,lda