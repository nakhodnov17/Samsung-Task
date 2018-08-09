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
