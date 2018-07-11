import os
import sys
import math
from copy import deepcopy

import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

use_cuda = False
os.environ["CUDA_VISIBLE_DEVICES"]="7"
if torch.cuda.is_available():
    device = torch.cuda.device("cuda:7")
    use_cuda = True


# get pure data from tensor
def get_pure(args):
    return args.data.cpu().clone().numpy()


# get mean value of elements of list
def list_mean(lst):
    return sum(lst) / float(len(lst))


# print plots
def print_plots(data, axis, labels):
    N_plots = len(data)
    plt.figure(figsize=(30, (N_plots // 3 + 1) * 10))

    for idx in range(len(data)):
        plt.subplot(N_plots // 3 + 1, 3, idx + 1)
        for jdx in range(len(data[idx])):
            plt.plot(data[idx][jdx], label=labels[idx][jdx])
        plt.xlabel(axis[idx][0], fontsize=16)
        plt.ylabel(axis[idx][1], fontsize=16)
        plt.legend(loc=0, fontsize=16)


# Module to print shape of input data
class ShapePrint(nn.Module):
    def __init__(self):
        super(ShapePrint, self).__init__()
        pass

    def forward(self, X):
        print(X.shape)
        return X


# Explinear Block for neural networks
class ExpLinear(nn.Module):
    def __init__(self, in_size):
        super(ExpLinear, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_size, 1))
        self.alpha = nn.Sequential(nn.Linear(1, 1, bias=False))
        self.cloned_x = None

    def forward(self, x):
        self.cloned_x = self.alpha(torch.exp(self.linear(x))).clone().detach()
        return self.alpha(torch.exp(self.linear(x)))

    def get_alpha(self):
        return get_pure(dict(self.alpha.named_parameters())['0.weight'])[0][0]

    def init_weigth(self, mean, std):
        for name, param in super(ExpLinear, self).named_parameters():
            param.data.normal_(mean, std)


# Apply transformation: [x1, x2, ..., xn-1, xn] -> [BA(x1), BA(x2), ..., BA(xn-1), exp(xn)]
class AugActivation(nn.Module):
    def __init__(self, base_activation):
        super(AugActivation, self).__init__()
        self.base_activation = base_activation
        self.exp_activation = torch.exp

    def forward(self, x):
        y_1 = self.base_activation(x[:, :-1])
        y_2 = self.exp_activation(x[:, -1:])
        return torch.cat([y_1, y_2], dim=1)


# train strategy (compute loss and miscl rate via all samples from dataset)
def train_strategy_NO_STOCH(network, loss_func, optimizer, dataloader, reg_lambda):
    for batch_dots, batch_labels in dataloader:
        x = Variable(batch_dots)
        correct_y = Variable(batch_labels.float())
        if use_cuda:
            x = x.cuda()
            correct_y = correct_y.cuda()

        predict_y = network(x)
        loss, _ = loss_func(predict_y, correct_y, network, reg_lambda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if use_cuda:
        loss, reg_loss = loss_func(network(dataloader.dataset.dots.cuda()),
                                   dataloader.dataset.classes.cuda(),
                                   network, reg_lambda)
        misclass_rate = (np.where(get_pure(network(dataloader.dataset.dots.cuda())) < 0, -1, 1) !=
                         get_pure(dataloader.dataset.classes.cuda())).sum()
    else:
        loss, reg_loss = loss_func(network(dataloader.dataset.dots),
                                   dataloader.dataset.classes,
                                   network, reg_lambda)
        misclass_rate = (np.where(get_pure(network(dataloader.dataset.dots)) < 0, -1, 1) !=
                         get_pure(dataloader.dataset.classes)).sum()

    return get_pure(loss), misclass_rate, get_pure(reg_loss)


# test strategy (compute loss and miscl rate via all samples from dataset)
def test_strategy_NO_STOCH(network, loss_func, optimizer, dataloader, reg_lambda):
    if use_cuda:
        loss, reg_loss = loss_func(network(dataloader.dataset.dots.cuda()),
                                   dataloader.dataset.classes.cuda(),
                                   network, reg_lambda)
        misclass_rate = (np.where(get_pure(network(dataloader.dataset.dots.cuda())) < 0, -1, 1) !=
                         get_pure(dataloader.dataset.classes.cuda())).sum()
    else:
        loss, reg_loss = loss_func(network(dataloader.dataset.dots),
                                   dataloader.dataset.classes,
                                   network, reg_lambda)
        misclass_rate = (np.where(get_pure(network(dataloader.dataset.dots)) < 0, -1, 1) !=
                         get_pure(dataloader.dataset.classes)).sum()

    return get_pure(loss), misclass_rate, get_pure(reg_loss)


# train strategy (compute loss and miscl rate via subsamples from dataset)
def train_strategy_STOCH(network, loss_func, optimizer, dataloader, reg_lambda):
    losses_batch = []
    misscl_rate_batch = []
    reg_loss = None

    for batch_dots, batch_labels in dataloader:
        x = Variable(batch_dots)
        correct_y = Variable(batch_labels.float())
        if use_cuda:
            x = x.cuda()
            correct_y = correct_y.cuda()

            predict_y = network(x)
            loss, reg_loss = loss_func(predict_y, correct_y, network, reg_lambda)

            losses_batch.append(get_pure(loss))
            misscl_rate_batch.append((np.where(get_pure(predict_y) < 0, -1, 1) != get_pure(correct_y)).sum())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return list_mean(losses_batch), sum(misscl_rate_batch), get_pure(reg_loss)


# test strategy (compute loss and miscl rate via subsamples from dataset)
def test_strategy_STOCH(network, loss_func, optimizer, dataloader, reg_lambda):
    losses_batch = []
    misscl_rate_batch = []
    reg_loss = None

    for batch_dots, batch_labels in dataloader:
        x = Variable(batch_dots)
        correct_y = Variable(batch_labels.float())
        if use_cuda:
            x = x.cuda()
            correct_y = correct_y.cuda()

            predict_y = network(x)
            loss, reg_loss = loss_func(predict_y, correct_y, network, reg_lambda)

            losses_batch.append(get_pure(loss))
            misscl_rate_batch.append((np.where(get_pure(predict_y) < 0, -1, 1) != get_pure(correct_y)).sum())

    return list_mean(losses_batch), sum(misscl_rate_batch), get_pure(reg_loss)


# process of training network
def train_(network, loss_func,
           learning_rate, reinit_optim,
           train_strategy, test_strategy,
           reg_lambda,
           epochs,
           dataloader_train, dataloader_test,
           plot_graphs=True, verbose=False,
           epoch_hook=None):
    losses_train = []
    losses_test = []
    misscl_rate_train = []
    misscl_rate_test = []
    optimizer = None
    scheduler = None

    try:
        for epoch in range(epochs):
            if epoch == 0:
                optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, reinit_optim)
            scheduler.step()

            # train
            loss, misscl_rate, _ = train_strategy(network, loss_func, optimizer, dataloader_train, reg_lambda)
            losses_train.append(loss)
            misscl_rate_train.append(misscl_rate)

            # test
            loss, misscl_rate, _ = test_strategy(network, loss_func, optimizer, dataloader_test, reg_lambda)
            losses_test.append(loss)
            misscl_rate_test.append(misscl_rate)

            sys.stdout.write(
                '\rEpoch {0}... Empirical Loss/Misclassification Rate (Train): {1:.3f}/{2:.3f}\t Empirical Loss/Misclassification Rate (Test): {3:.3f}/{4:.3f}'.format(
                    epoch, losses_train[-1], misscl_rate_train[-1], losses_test[-1], misscl_rate_test[-1]))

            if epoch_hook:
                epoch_hook(network=network, loss_func=loss_func,
                           learning_rate=learning_rate, reinit_optim=reinit_optim,
                           train_strategy=train_strategy, test_strategy=test_strategy,
                           reg_lambda=reg_lambda,
                           epochs=epochs, epoch=epoch,
                           dataloader_train=dataloader_train, dataloader_test=dataloader_test,
                           plot_graphs=plot_graphs, verbose=verbose,
                           losses_train=losses_train, misscl_rate_train=misscl_rate_train,
                           losses_test=losses_test, misscl_rate_test=misscl_rate_test
                           )

            if losses_train[-1] > 10e6 or math.isnan(losses_train[-1]) or math.isinf(losses_train[-1]):
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    if plot_graphs:
        print_plots([[losses_train, losses_test],
                     [misscl_rate_train, misscl_rate_test]],
                    [['Epochs', 'Mean loss'],
                     ['Epochs', 'Number of objects']],
                    [['Loss (Train)', 'Loss (Test)'],
                     ['Misclassification Rate (Train)', 'Misclassification Rate (Test)']
                     ])
    if verbose:
        return losses_train, misscl_rate_train, losses_test, misscl_rate_test


# process of training network with regulariser
def train_EXP(network, loss_func,
              learning_rate, reinit_optim,
              train_strategy, test_strategy,
              reg_lambda,
              epochs,
              dataloader_train, dataloader_test,
              plot_graphs=True, verbose=False,
              epoch_hook=None):
    losses_train = []
    losses_test = []
    misscl_rate_train = []
    misscl_rate_test = []
    alpha = []
    reg_losses = []
    optimizer = None
    scheduler = None

    try:
        for epoch in range(epochs):
            if epoch == 0:
                optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, reinit_optim)
            scheduler.step()

            # train
            loss, misscl_rate, reg_loss = train_strategy(network, loss_func, optimizer, dataloader_train, reg_lambda)
            losses_train.append(loss)
            misscl_rate_train.append(misscl_rate)

            # test
            loss, misscl_rate, _ = test_strategy(network, loss_func, optimizer, dataloader_test, reg_lambda)
            losses_test.append(loss)
            misscl_rate_test.append(misscl_rate)

            alpha.append(network.get_alpha())
            reg_losses.append(reg_loss)

            sys.stdout.write(
                '\rEpoch {0}... Empirical Loss/Misclassification Rate (Train): {1:.3f}/{2:.3f}\t Empirical Loss/Misclassification Rate (Test): {3:.3f}/{4:.3f}'.format(
                    epoch, losses_train[-1], misscl_rate_train[-1], losses_test[-1], misscl_rate_test[-1]))

            if epoch_hook:
                epoch_hook(network=network, loss_func=loss_func,
                           learning_rate=learning_rate, reinit_optim=reinit_optim,
                           train_strategy=train_strategy, test_strategy=test_strategy,
                           reg_lambda=reg_lambda,
                           epochs=epochs, epoch=epoch,
                           dataloader_train=dataloader_train, dataloader_test=dataloader_test,
                           plot_graphs=plot_graphs, verbose=verbose,
                           losses_train=losses_train, misscl_rate_train=misscl_rate_train,
                           losses_test=losses_test, misscl_rate_test=misscl_rate_test
                           )

            if losses_train[-1] > 10e6 or math.isnan(losses_train[-1]) or math.isinf(losses_train[-1]):
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        pass
    if plot_graphs:
        print_plots([[losses_train, losses_test],
                     [misscl_rate_train, misscl_rate_test],
                     [alpha],
                     [reg_losses]],
                    [['Epochs', 'Mean loss'],
                     ['Epochs', 'Number of objects'],
                     ['Epochs', 'Alpha'],
                     ['Epochs', 'Mean reg loss']],
                    [['Loss (Train)', 'Loss (Test)'],
                     ['Misclassification Rate (Train)', 'Misclassification Rate (Test)'],
                     ['Alpha'],
                     ['Reg Loss']
                     ])

    if verbose:
        return losses_train, misscl_rate_train, losses_test, misscl_rate_test, alpha, reg_losses


# Cubic HingeLoss
def loss_func(predict_y, correct_y, network, reg_lambda):
    loss = None
    reg_loss = None
    if use_cuda:
        loss = torch.sum(torch.max(-correct_y * predict_y + torch.tensor(1.).cuda(), torch.tensor(0.).cuda()) ** 3)
        reg_loss = torch.tensor(0.).cuda()
    else:
        loss = torch.sum(torch.max(-correct_y * predict_y + torch.tensor(1.), torch.tensor(0.)) ** 3)
        reg_loss = torch.tensor(0.)

    return loss + reg_loss, reg_loss


# Cubic HingeLoss with regularisation for one exponential neuron
def loss_func_EXP(predict_y, correct_y, network, reg_lambda):
    loss = None
    reg_loss = None

    if use_cuda:
        loss = torch.sum(torch.max(-correct_y * predict_y + torch.tensor(1.).cuda(), torch.tensor(0.).cuda()) ** 3)
    else:
        loss = torch.sum(torch.max(-correct_y * predict_y + torch.tensor(1.), torch.tensor(0.)) ** 3)

    for param in network.explinear.alpha.parameters():
        reg_loss = float(reg_lambda) * param.norm(2)

    return loss + reg_loss, reg_loss


# Cubic HingeLoss with regularisation for network augmented with exp neuron on each layer
def loss_func_AUG(predict_y, correct_y, network, reg_lambda):
    loss = None
    reg_loss = None

    if use_cuda:
        loss = torch.sum(torch.max(-correct_y * predict_y + torch.tensor(1.).cuda(), torch.tensor(0.).cuda()) ** 3)
    else:
        loss = torch.sum(torch.max(-correct_y * predict_y + torch.tensor(1.), torch.tensor(0.)) ** 3)

    reg_loss = Variable(torch.tensor(0.)).cuda()
    for name, param in network.named_parameters():
        if name.find('linear') >= 0 and name.find('weight') >= 0:  # and name != 'linear.0.weight':
            reg_loss += reg_lambda * param[:, -1].norm(2)

    return loss + reg_loss, reg_loss