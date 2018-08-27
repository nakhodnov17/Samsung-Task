import os
import sys
import math

from copy import deepcopy

from itertools import chain
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.linalg import orth

import seaborn as sns

from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST

# import MNIST dataset with class selection
sys.path.insert(0, '/home/m.nakhodnov/Samsung-Tasks/Datasets/MyMNIST')
from MyMNIST import MNIST_Class_Selection

# Ignore warnings
import warnings
import functools

warnings.filterwarnings("ignore")

use_cuda = False
device = None
os.environ["CUDA_VISIBLE_DEVICES"] = str(4)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_cuda = True


# Set DoubleTensor as a base type for computations
t_type = torch.float64


def print_plots(data, axis, labels, file_name=None):
    N_plots = len(data)
    plt.figure(figsize=(30, (N_plots // 3 + 1) * 10))

    for idx in range(len(data)):
        plt.subplot(N_plots // 3 + 1, 3, idx + 1)
        for jdx in range(len(data[idx])):
            plt.plot(data[idx][jdx], label=labels[idx][jdx])
        plt.xlabel(axis[idx][0], fontsize=16)
        plt.ylabel(axis[idx][1], fontsize=16)
        plt.legend(loc=0, fontsize=16)
    if file_name is not None:
        plt.savefig(file_name)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset_train = MNIST_Class_Selection('.', train=True, download=True, transform=transform)
dataset_test = MNIST_Class_Selection('.', train=False, transform=transform)

dataloader_train = DataLoader(dataset_train, batch_size=100, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=100, shuffle=False)

net = nn.Sequential(
    nn.Linear(28 * 28, 300),
    nn.Tanh(),
    nn.Linear(300, 100),
    nn.Tanh(),
    nn.Linear(100, 10)
).to(device=device).double()

optim = torch.optim.Adam(net.parameters())

checkpoint_file_name = './Checkpoints/' + 'e{0}-{1}_' + 'ml_est' + '.pth'
plots_file_name = './Plots/' + 'e{0}-{1}_' + 'ml_est' + '.png'
log_file_name = './Logs/' + 'ml_est' + '.txt'
if log_file_name is not None:
    log_file = open(log_file_name, 'a')
    log_file.write('\rNew run of training.\r')
    log_file.close()

### Train loss/accuracy
train_losses = []
train_accs = []
### Test loss/accuracy
test_losses = []
test_accs = []
try:
    for epoch in range(200):
        ### One update of particles via all dataloader_train
        for X, y in dataloader_train:
            X = X.double().to(device=device).view(X.shape[0], -1)
            y = y.to(device=device)

            predicts = net(X)
            loss = torch.nn.CrossEntropyLoss()(predicts, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        ### Evaluate cross entropy and accuracy over dataloader_train
        train_loss = 0.
        train_acc = 0.
        for X_train, y_train in dataloader_train:
            X_train = X_train.double().to(device=device).view(X_train.shape[0], -1)
            y_train = y_train.to(device=device)

            predicts = net(X_train)

            y_pred = torch.argmax(nn.Softmax()(predicts), dim=1)
            train_loss += nn.CrossEntropyLoss(reduction='sum')(predicts, y_train)
            train_acc += torch.sum(y_pred == y_train).float()
        train_loss /= (len(dataloader_train.dataset) + 0.)
        train_acc /= (len(dataloader_train.dataset) + 0.)

        ### Evaluate cross entropy and accuracy over dataloader_test
        test_loss = 0.
        test_acc = 0.
        for X_test, y_test in dataloader_test:
            X_test = X_test.double().to(device=device).view(X_test.shape[0], -1)
            y_test = y_test.to(device=device)

            predicts = net(X_test)

            y_pred = torch.argmax(nn.Softmax()(predicts), dim=1)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(predicts, y_test)
            test_acc += torch.sum(y_pred == y_test).double()
        test_loss /= (len(dataloader_test.dataset) + 0.)
        test_acc /= (len(dataloader_test.dataset) + 0.)


        ### Append evaluated losses and accuracies
        train_losses.append(train_loss.data[0].cpu().numpy())
        train_accs.append(train_acc.data[0].cpu().numpy())
        test_losses.append(test_loss.data[0].cpu().numpy())
        test_accs.append(test_acc.data[0].cpu().numpy())

        sys.stdout.write(
            ('\nEpoch {0}... ' +
             '\nEmpirical Loss (Train/Test): {1:.3f}/{2:.3f}' +
             '\nAccuracy (Train/Test): {3:.3f}/{4:.3f}\t'
             ).format(epoch,
                      train_loss, test_loss,
                      train_acc, test_acc,
                      )
        )

        if log_file_name is not None:
            log_file = open(log_file_name, 'a')
            log_file.write(
                ('\nEpoch {0}... ' +
                 '\nEmpirical Loss (Train/Test): {1:.3f}/{2:.3f}' +
                 '\nAccuracy (Train/Test): {3:.3f}/{4:.3f}\t'
                 ).format(epoch,
                          train_loss, test_loss,
                          train_acc, test_acc,
                          )
            )
            log_file.close()

except KeyboardInterrupt:
    pass
print_plots([[train_losses, test_losses],
             [train_accs, test_accs]],
            [['Epochs', ''],
             ['Epochs', '% * 1e-2']],
            [['Cross Entropy Loss (Train)', 'Cross Entropy Loss (Test)'],
             ['Accuracy (Train)', 'Accuracy (Test)']
             ],
            plots_file_name.format(0, 199)
    )
if checkpoint_file_name is not None:
    torch.save(net.state_dict(), checkpoint_file_name.format(0, 199))