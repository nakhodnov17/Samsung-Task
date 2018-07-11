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

# import MNIST dataset with class selection
sys.path.insert(0, '/home/m.nakhodnov/Samsung-Tasks/Adding_one_neuron/utils/MyMNIST')
from MyMNIST import MNIST_Class_Selection


# generate two concentric circles with specified radius
class TwoCircleDataset(Dataset):
    def __init__(self, R1, R2, N, dim, train=False, normalize=True):
        self.R1 = R1
        self.R2 = R2
        self.N = N
        self.dim = dim
        self.train = train
        self.normalize = normalize

        self.dots = torch.zeros([N, dim])

        self.classes = torch.zeros([N])
        self.classes.bernoulli_(0.5)
        self.classes = self.classes * 2 - 1

        self.normal_generator = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        for idx in range(self.N):
            self.dots[idx] = self.normal_generator.sample_n(self.dim).view(1, -1)
            self.dots[idx] /= torch.sqrt(torch.sum(self.dots[idx] ** 2))
            self.classes[idx] = self.classes[idx]
            if self.classes[idx] == 1:
                self.dots[idx] *= self.R1
            else:
                self.dots[idx] *= self.R2

        self.mean = self.dots.mean(dim=0)
        self.std = self.dots.std(dim=0)

        if self.normalize:
            self.dots = (self.dots - self.mean.expand_as(self.dots)) / self.std.expand_as(self.dots)

    def __len__(self):
        return len(self.dots)

    def __getitem__(self, idx):
        return self.dots[idx], self.classes[idx]


# generate two samples from N-dim normal distribution with specified mean
class TwoGaussiansDataset(Dataset):
    def __init__(self, M1, M2, N, dim, train=False):
        self.M1 = M1
        self.M2 = M2
        self.N = N
        self.dim = dim
        self.train = train

        self.dots = torch.zeros([N, dim])

        self.classes = torch.zeros([N])
        self.classes.bernoulli_(0.5)
        self.classes = self.classes * 2 - 1

        self.normal_generator_1 = torch.distributions.normal.Normal(torch.tensor([M1]), torch.tensor([1.0]))
        self.normal_generator_2 = torch.distributions.normal.Normal(torch.tensor([M2]), torch.tensor([1.0]))
        for idx in range(self.N):
            if self.classes[idx] == 1:
                self.dots[idx] = self.normal_generator_1.sample_n(self.dim).view(1, -1)
            else:
                self.dots[idx] = self.normal_generator_2.sample_n(self.dim).view(1, -1)

    def __len__(self):
        return len(self.dots)

    def __getitem__(self, idx):
        return self.dots[idx], self.classes[idx]


# CIFAR dataset
class CIFARDataset(Dataset):
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __init__(self, classes=(1, 2), train=False, normalize=True):
        self.train = train
        self.normalize = normalize

        if self.train:
            n_batchs = ['1', '2', '3', '4', '5']

            for n_batch in n_batchs:
                batch = self.unpickle('cifar-10-batches-py/data_batch_' + n_batch)

                self.dots = np.array(batch[b'data'])[np.logical_or(np.array(batch[b'labels']) == classes[0],
                                                                   np.array(batch[b'labels']) == classes[1])]
                self.classes = np.array(batch[b'labels'])[np.logical_or(np.array(batch[b'labels']) == classes[0],
                                                                        np.array(batch[b'labels']) == classes[1])]
        else:
            batch = self.unpickle('cifar-10-batches-py/test_batch')

            self.dots = np.array(batch[b'data'])[np.logical_or(np.array(batch[b'labels']) == classes[0],
                                                               np.array(batch[b'labels']) == classes[1])]
            self.classes = np.array(batch[b'labels'])[np.logical_or(np.array(batch[b'labels']) == classes[0],
                                                                    np.array(batch[b'labels']) == classes[1])]

        self.dots = torch.FloatTensor(self.dots)
        self.classes = torch.FloatTensor(self.classes)

        self.classes[self.classes == classes[0]] = -1
        self.classes[self.classes == classes[1]] = 1

        self.mean = self.dots.mean(dim=0)
        self.std = self.dots.std(dim=0)

        if self.normalize:
            self.dots = (self.dots - self.mean.expand_as(self.dots)) / self.std.expand_as(self.dots)

    def __len__(self):
        return len(self.dots)

    def __getitem__(self, idx):
        return self.dots[idx], self.classes[idx]


# Large Movie Review Dataset (using GloVe as word embeddings)
class LMDRDataset():
    def __init__(self, base_path='/home/m.nakhodnov/Samsung-Tasks/Adding_one_neuron/', emb_size=50, kaggle=False, train=False, normalize=True):
        self.base_path = base_path
        self.emb_size = emb_size
        self.kaggle = kaggle
        self.train = train
        self.normalize = normalize

        self.test_path = base_path + 'LMDR/test/test/'
        self.train_neg_path = base_path + 'LMDR/train/train/neg/'
        self.train_pos_path = base_path + 'LMDR/train/train/pos/'

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