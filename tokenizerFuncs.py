import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import torch,random,os
from torch.utils.data import DataLoader,Dataset
from itertools import permutations
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec,FastText
from glove import Glove, Corpus
from gensim.models.word2vec import LineSentence
from sklearn.preprocessing import OneHotEncoder

import logging,pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Tokenizer:
    def __init__(self, sequences, labels, seqMaxLen=512, multiLabel=False):
        print('Tokenizing the data...')
        cnt = 2
        id2tkn,tkn2id = ["[UNK]","[PAD]"],{"[UNK]":0,"[PAD]":1}
        for seq in tqdm(sequences):
            for tkn in seq:
                if tkn not in tkn2id:
                    tkn2id[tkn] = cnt
                    id2tkn.append(tkn)
                    cnt += 1
        self.id2tkn,self.tkn2id = id2tkn,tkn2id
        self.tknNum = cnt
        self.seqMaxLen = min(max([len(s) for s in sequences]), seqMaxLen)

        cnt = 0
        id2lab,lab2id = [],{}
        if multiLabel:
            for labs in tqdm(labels):
                for lab in labs:
                    if lab not in lab2id:
                        lab2id[lab] = cnt
                        id2lab.append(lab)
                        cnt += 1
        else:
            for lab in tqdm(labels):
                # for lab in labs:
                if lab not in lab2id:
                    lab2id[lab] = cnt
                    id2lab.append(lab)
                    cnt += 1
        labNum = cnt
        self.id2lab,self.lab2id = id2lab,lab2id
        self.labNum = labNum
        self.multiLabel = multiLabel
        if multiLabel:
            self.oh = OneHotEncoder().fit(np.arange(labNum).reshape(-1,1))
    def tokenize_sequences(self, sequences, eval=False, baseLen=1):
        if eval:
            seqMaxLen = self.seqMaxLen
        else:
            seqMaxLen = min(max([len(i) for i in sequences]), self.seqMaxLen)
        seqMaxLen = int((seqMaxLen//baseLen) * baseLen)
        return [[self.tkn2id[tkn] if tkn in self.tkn2id else self.tkn2id['[UNK]'] for tkn in seq[:seqMaxLen]]+[self.tkn2id['[PAD]']]*(seqMaxLen-len(seq)) for seq in sequences],[[1]*len(seq[:seqMaxLen])+[0]*(seqMaxLen-len(seq)) for seq in sequences]
    def tokenize_labels(self, labels):
        if self.multiLabel:
            return [self.oh.transform(np.array([self.lab2id[i] for i in labs]).reshape(-1,1)).toarray().sum(axis=0).astype('int32').tolist() if len(labs)>0 else np.zeros(self.labNum, dtype=np.int32) for labs in labels]
        else:
            return [self.lab2id[lab] for lab in labels]

    def vectorize(self, sequences, method=["skipgram"], embSize=64, window=7, iters=10, batchWords=10000,
                  workers=8, loadCache=True, suf=""):
        path = f'cache/{"-".join(method)}_d{embSize*len(method)}_{suf}.pkl'
        if os.path.exists(path) and loadCache:
            with open(path, 'rb') as f:
                self.embedding = pickle.load(f)
            print('Loaded cache from cache/%s'%path)
        else:
            corpus = [i+['[PAD]'] for i in sequences]
            embeddings = []
            if 'skipgram' in method:
                model = Word2Vec(corpus, min_count=0, window=window, vector_size=embSize, workers=workers, sg=1, epochs=iters, batch_words=batchWords)
                word2vec = np.zeros((self.tknNum, embSize), dtype=np.float32)
                for i in range(self.tknNum):
                    if self.id2tkn[i] in model.wv:
                        word2vec[i] = model.wv[self.id2tkn[i]]
                    else:
                        print('word %s not in word2vec.'%self.id2tkn[i])
                        word2vec[i] =  np.random.random(embSize)
                embeddings.append(word2vec)
            if 'cbow' in method:
                model = Word2Vec(corpus, min_count=0, window=window, vector_size=embSize, workers=workers, sg=0, epochs=iters, batch_words=batchWords)
                word2vec = np.zeros((self.tknNum, embSize), dtype=np.float32)
                for i in range(self.tknNum):
                    if self.id2tkn[i] in model.wv:
                        word2vec[i] = model.wv[self.id2tkn[i]]
                    else:
                        print('word %s not in word2vec.'%self.id2tkn[i])
                        word2vec[i] =  np.random.random(embSize)
                embeddings.append(word2vec)
            if 'glove' in method:
                gCorpus = Corpus()
                gCorpus.fit(corpus, window=window)
                model = Glove(no_components=embSize)
                model.fit(gCorpus.matrix, epochs=iters, no_threads=workers, verbose=True)
                model.add_dictionary(gCorpus.dictionary)
                word2vec = np.zeros((self.tknNum, embSize), dtype=np.float32)
                for i in range(self.tknNum):
                    if self.id2tkn[i] in model.dictionary:
                        word2vec[i] = model.word_vectors[model.dictionary[self.id2tkn[i]]]
                    else:
                        print('word %s not in word2vec.'%self.id2tkn[i])
                        word2vec[i] =  np.random.random(embSize)
                embeddings.append(word2vec)
            if 'fasttext' in method:
                model = FastText(corpus, vector_size=embSize, window=window, min_count=0, epochs=iters, sg=1, workers=workers, batch_words=batchWords)
                word2vec = np.zeros((self.tknNum, embSize), dtype=np.float32)
                for i in range(self.tknNum):
                    if self.id2tkn[i] in model.wv:
                        word2vec[i] = model.wv[self.id2tkn[i]]
                    else:
                        print('word %s not in word2vec.'%self.id2tkn[i])
                        word2vec[i] =  np.random.random(embSize)
                embeddings.append(word2vec)
            self.embedding = np.hstack(embeddings)

            with open(path, 'wb') as f:
                pickle.dump(self.embedding, f, protocol=4)