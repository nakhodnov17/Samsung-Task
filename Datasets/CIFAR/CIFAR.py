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


# CIFAR dataset
class CIFARDataset(Dataset):
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __init__(self, base_path = './', classes=(1, 2), train=False, normalize=True):
        self.base_path = base_path
        self.train = train
        self.normalize = normalize

        if self.train:
            n_batchs = ['1', '2', '3', '4', '5']

            for n_batch in n_batchs:
                batch = self.unpickle(self.base_path + 'cifar-10-batches-py/data_batch_' + n_batch)

                self.dots = np.array(batch[b'data'])[np.logical_or(np.array(batch[b'labels']) == classes[0],
                                                                   np.array(batch[b'labels']) == classes[1])]
                self.classes = np.array(batch[b'labels'])[np.logical_or(np.array(batch[b'labels']) == classes[0],
                                                                        np.array(batch[b'labels']) == classes[1])]
        else:
            batch = self.unpickle(self.base_path + 'cifar-10-batches-py/test_batch')

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