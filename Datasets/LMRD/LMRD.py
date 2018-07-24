import os
import sys
import math
from os import listdir
from os.path import isfile, join

from copy import deepcopy

import nltk
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Large Movie Review Dataset (using GloVe as word embeddings)
class LMRDDataset():
    def __init__(self, base_path='/home/m.nakhodnov/Samsung-Tasks/Datasets/', emb_size=50, kaggle=False, train=False, normalize=True):
        self.base_path = base_path
        self.emb_size = emb_size
        self.kaggle = kaggle
        self.train = train
        self.normalize = normalize

        self.test_path = base_path + 'LMRD/test/test/'
        self.train_neg_path = base_path + 'LMRD/train/train/neg/'
        self.train_pos_path = base_path + 'LMRD/train/train/pos/'

        self.emb_path = base_path + 'glove_embeddings/glove.6B.' + str(self.emb_size) + 'd.txt'
        self.emb_dict = dict()

        for line in open(self.emb_path):
            word = line[:line.find(' ')]
            emb = torch.Tensor(list(map(float, line[line.find(' '):].split())))
            self.emb_dict[word] = emb

        self.file_names = None
        self.neg_names = None
        self.pos_names = None

        self.embeddings = None
        self.neg_embeddings = None
        self.pos_embeddings = None

        if self.kaggle:
            self.file_names = [f for f in listdir(self.test_path) if isfile(join(self.test_path, f))]
            self.embeddings = torch.zeros([len(self.file_names), self.emb_size], dtype=torch.float)
            for idx, name in enumerate(self.file_names):
                text = open(self.test_path + name).read()
                self.embeddings[idx] = self.text_to_embedding(text)
        else:
            self.neg_names = [f for f in listdir(self.train_neg_path) if isfile(join(self.train_neg_path, f))]
            self.pos_names = [f for f in listdir(self.train_pos_path) if isfile(join(self.train_pos_path, f))]
            self.neg_embeddings = torch.zeros([len(self.neg_names), self.emb_size], dtype=torch.float)
            self.pos_embeddings = torch.zeros([len(self.pos_names), self.emb_size], dtype=torch.float)
            for idx, name in enumerate(self.neg_names):
                text = open(self.train_neg_path + name).read()
                self.neg_embeddings[idx] = self.text_to_embedding(text)
            for idx, name in enumerate(self.pos_names):
                text = open(self.train_pos_path + name).read()
                self.pos_embeddings[idx] = self.text_to_embedding(text)

        self.neg_train_len = None
        self.neg_test_len = None
        self.pos_train_len = None
        self.pos_test_len = None

        if not self.kaggle:
            self.neg_train_len = int(len(self.neg_embeddings) * 0.7)
            self.neg_test_len = len(self.neg_embeddings) - self.neg_train_len

            self.pos_train_len = int(len(self.pos_embeddings) * 0.7)
            self.pos_test_len = len(self.pos_embeddings) - self.neg_train_len

    def __len__(self):
        if self.kaggle:
            return len(self.embeddings)
        else:
            if self.train:
                return self.neg_train_len + self.pos_train_len
            else:
                return self.neg_test_len + self.pos_test_len

    def __getitem__(self, idx):
        if self.kaggle:
            return self.embeddings[idx]
        else:
            if self.train:
                if idx < self.neg_train_len:
                    return self.neg_embeddings[idx], torch.tensor(-1.)
                else:
                    return self.pos_embeddings[idx - self.neg_train_len], torch.tensor(1.)
            else:
                if idx < self.neg_test_len:
                    return self.neg_embeddings[self.neg_train_len + idx], torch.tensor(-1.)
                else:
                    return self.pos_embeddings[self.pos_train_len + idx - self.neg_test_len], torch.tensor(1.)

    def text_to_embedding(self, text):
        emb = torch.zeros([self.emb_size])
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        for word in map(str.lower, tokenizer.tokenize(text)):
            try:
                emb += self.emb_dict[word]
            except KeyError:
                pass
        return emb / len(text)