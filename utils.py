import numpy as np
import pandas as pd
import os,re
from tqdm import tqdm
from collections import Counter
import torch,random,json,pickle
from torch.utils.data import DataLoader,Dataset
from collections import OrderedDict

class MIMIC3(Dataset):
    def __init__(self, filePath, selectedLabels=None):
        self.filePath = filePath
        print('Loading the data...')
        data = pd.read_csv(filePath)
        self.sequences = [i.split() for i in tqdm(data['TEXT'].tolist())]
        self.labels = [sorted(list(set(i.split(';')))) for i in data['ICD9_CODE'].tolist()]
        if selectedLabels is not None:
            for i in range(len(self.labels)):
                self.labels[i] = [j for j in self.labels[i] if j in selectedLabels]
        self.sLens = [len(i) for i in self.sequences]
        self.ids = data['HADM_ID'].tolist()

        self.tokenizedSeqArr = [None]*len(data)
        self.maskPAD = [None]*len(data)
        self.tokenizedLabArr = [None]*len(data)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return self.getitem(index)
    def getitem(self, index):
        return {'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index],
                'tokenizedSeqArr':self.tokenizedSeqArr[index],
                'maskPAD':self.maskPAD[index],
                'tokenizedLabArr':self.tokenizedLabArr[index]}
    def cache_tokenized_input(self, tokenizer):
        self.tokenizedSeqArr,self.maskPAD = tokenizer.tokenize_sequences(tqdm(self.sequences))
        self.tokenizedLabArr = tokenizer.tokenize_labels(tqdm(self.labels))
    def __add__(self, other):
        self.filePath += "/"+other.filePath
        self.sequences += other.sequences
        self.labels += other.labels
        self.sLens += other.sLens
        self.ids += other.ids
        if hasattr(self, 'tokenizedSeqArr'):
            self.tokenizedSeqArr += other.tokenizedSeqArr
            self.maskPAD += other.maskPAD
        if hasattr(self, 'tokenizedLabArr'):
            self.tokenizedLabArr += other.tokenizedLabArr
        return self

class R8(Dataset):
    def __init__(self, filePath):
        self.filePath = filePath
        print('Loading the data...')
        data = pd.read_csv(filePath)
        self.sequences = [i.split() for i in tqdm(data['text'].tolist())]
        self.labels = data['intent'].tolist()
        self.sLens = [len(i) for i in self.sequences]
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index]}
    def __add__(self, other):
        self.filePath += "/"+other.filePath
        self.sequences += other.sequences
        self.labels += other.labels
        self.sLens += other.sLens
        return self

class IMDB(Dataset):
    def __init__(self, filePath):
        tknParser = re.compile("[A-Za-z0-9]+|[,;.!?()\'\"]", re.S)
        self.filePath = filePath
        print('Loading the data...')
        self.sequences,self.labels = [],[]
        for lab in ['pos','neg']:
            path = os.path.join(filePath, lab)
            for file in tqdm(np.sort(os.listdir(path))):
                tmp = open(os.path.join(path, file), encoding='utf8').readlines()
                assert len(tmp)==1
                self.sequences.append(tknParser.findall(tmp[0].lower()))
                self.labels.append(lab)
        self.sLens = [len(i) for i in self.sequences]
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index]}
    def __add__(self, other):
        self.filePath += "/"+other.filePath
        self.sequences += other.sequences
        self.labels += other.labels
        self.sLens += other.sLens
        return self

class ASTRAL_SCOPe(Dataset):
    def __init__(self, filePath):
        self.filePath = filePath
        with open(filePath, 'r') as f:
            ids,proteins,labels = [],[],[]
            fasta = ""
            for line in f.readlines():
                if line.startswith('>'):
                    line = line[1:].split()
                    ids.append(line[0])
                    labels.append('.'.join(line[1].split('.')[:2]))
                    if len(fasta)>0:
                        proteins.append(fasta)
                        fasta = ""
                else:
                    fasta += line.strip()
            if len(fasta)>0:
                proteins.append(fasta)
                fasta = ""
        self.ids = ids
        self.sequences = [list(i) for i in proteins]
        self.sLens = [len(i) for i in self.sequences]
        self.labels = labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index]}
    def __add__(self, other):
        self.filePath += "/"+other.filePath
        self.sequences += other.sequences
        self.labels += other.labels
        self.sLens += other.sLens
        return self
    def get_sub_set(self, indexs):
        obj = ASTRAL_SCOPe_(self.filePath, 
                            np.array(self.ids)[indexs].tolist(),
                            np.array(self.sequences)[indexs].tolist(),
                            np.array(self.sLens)[indexs].tolist(),
                            np.array(self.labels)[indexs].tolist())
        return obj
class ASTRAL_SCOPe_(ASTRAL_SCOPe):
    def __init__(self, filePath, ids, sequences, sLens, labels):
        self.filePath = filePath
        self.ids = ids
        self.sequences = sequences
        self.sLens = sLens
        self.labels = labels

from sklearn.datasets import fetch_20newsgroups
class Sklearn_20NG(Dataset):
    def __init__(self, filePath, type='train'):
        self.filePath = filePath
        data = fetch_20newsgroups(data_home=filePath, subset=type)
        tknParser = re.compile("[A-Za-z0-9]+|[,;.!?()\'\"]|\n", re.S)
        self.sequences = [tknParser.findall(i.lower()) for i in data.data]
        self.sLens = [len(i) for i in self.sequences]
        self.labels = [data.target_names[i] for i in data.target]
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index]}
    def __add__(self, other):
        self.filePath += "/"+other.filePath
        self.sequences += other.sequences
        self.labels += other.labels
        self.sLens += other.sLens
        return self

class BBC_News(Dataset):
    def __init__(self, filePath):
        self.filePath = filePath
        data = pd.read_csv(filePath)
        tknParser = re.compile("[A-Za-z0-9]+|[,;.!?()\'\"]", re.S)
        self.sequences = [tknParser.findall(i.lower()) for i in data['text']]
        self.sLens = [len(i) for i in self.sequences]
        self.labels = data['category'].tolist()
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index]}
    def __add__(self, other):
        self.filePath += "/"+other.filePath
        self.sequences += other.sequences
        self.labels += other.labels
        self.sLens += other.sLens
        return self
    def get_sub_set(self, indexs):
        obj = BBC_News_(self.filePath, 
                        np.array(self.sequences)[indexs].tolist(),
                        np.array(self.sLens)[indexs].tolist(),
                        np.array(self.labels)[indexs].tolist())
        return obj
class BBC_News_(BBC_News):
    def __init__(self, filePath, sequences, sLens, labels):
        self.filePath = filePath
        self.sequences = sequences
        self.sLens = sLens
        self.labels = labels

class CIFAR_100(Dataset):
    def __init__(self, filePath, seqMaxLen=1024):
        self.filePath = filePath
        with open(filePath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        tmp = (data[b'data'].reshape(-1,3,1024).transpose([0,2,1]) / 255.0).astype('float32')
        tmp = (tmp - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
        self.tokenizedSeqArr = tmp
        self.maskPAD = [[1]*seqMaxLen for i in range(len(self.tokenizedSeqArr))]
        self.sLens = [len(i) for i in self.tokenizedSeqArr]
        self.tokenizedLabArr = data[b'fine_labels']
        with open(os.path.split(filePath)[0]+'/meta', 'rb') as f:
            self.id2lab = pickle.load(f)['fine_label_names']
        self.labels = [self.id2lab[i] for i in self.tokenizedLabArr]
    def __len__(self):
        return len(self.tokenizedSeqArr)
    def __getitem__(self, index):
        return {'tokenizedSeqArr':self.tokenizedSeqArr[index],
                'maskPAD':self.maskPAD[index],
                'sLen':self.sLens[index],
                'label':self.labels[index],
                'tokenizedLabArr':self.tokenizedLabArr[index]}

class ListOps(Dataset):
    def __init__(self, filePath):
        self.filePath = filePath
        tmp = pd.read_table(filePath, sep='\t')
        self.sequences = [i.split() for i in tmp['Source'].tolist()]
        self.labels = tmp['Target'].tolist()
        self.sLens = [len(i) for i in self.sequences]
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index]}
    def __add__(self, other):
        self.filePath += "/"+other.filePath
        self.sequences += other.sequences
        self.labels += other.labels
        self.sLens += other.sLens
        return self

import tensorflow as tf
cpu=tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpu)
import tensorflow_datasets as tfds
def adapt_example(example):
    return {'Source': example['text'], 'Target': example['label']}
class ByteLevel_TextClassification(Dataset):
    def __init__(self, type='train'):
        with tf.device('cpu'):
            data = tfds.load('imdb_reviews')
        raw = data[type]
        tmp = raw.map(adapt_example)
        self.sequences = [list(i['Source'].numpy()) for i in tmp]
        self.labels = [i['Target'].numpy() for i in tmp]
        self.sLens = [len(i) for i in self.sequences]

    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index]}
    def __add__(self, other):
        self.filePath += "/"+other.filePath
        self.sequences += other.sequences
        self.labels += other.labels
        self.sLens += other.sLens
        return self

def decode(x):
    decoded = {
        'inputs':
            tf.cast(tf.image.rgb_to_grayscale(x['image']), dtype=tf.int32),
        'targets':
            x['label']
    }
    return decoded
class TFDS_CIFAR10(Dataset):
    def __init__(self, type='train', seqMaxLen=1024):
        with tf.device('cpu'):
            if type=='test':
                data = tfds.load('cifar10', split='test')
            else:
                data = tfds.load('cifar10', split='train[:90%]')
        tmp = data.map(decode)
        sequences = np.array([i['inputs'].numpy() for i in tmp], dtype=np.float32).reshape(-1,1024,1)
        self.tokenizedSeqArr = sequences/255 - 0.5
        self.maskPAD = [[1]*seqMaxLen for i in range(len(self.tokenizedSeqArr))]
        self.sLens = [len(i) for i in self.tokenizedSeqArr]
        self.tokenizedLabArr = [i['targets'].numpy() for i in tmp]
        self.id2lab,self.lab2id = None,None
    def __len__(self):
        return len(self.tokenizedSeqArr)
    def __getitem__(self, index):
        return {'tokenizedSeqArr':self.tokenizedSeqArr[index],
                'maskPAD':self.maskPAD[index],
                'sLen':self.sLens[index],
                # 'label':self.labels[index],
                'tokenizedLabArr':self.tokenizedLabArr[index]}