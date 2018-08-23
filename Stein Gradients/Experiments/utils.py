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

# Ignore warnings
import warnings
import functools

warnings.filterwarnings("ignore")

use_cuda = False
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_cuda = True

# Set DoubleTensor as a base type for computations
t_type = torch.float64

print('utils.py was imported.')

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


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


def plot_projections(dm=None, use_real=True, kernel='tri', pdf=None, N_plots_max=10):
    """
        Plot marginal kernel density estimation
    Args:
        dm (DistributionMover): class containing particles which define distribution
        use_real (bool): If set to True then apply transformation dm.lt.transform before creating plot
        kernel (str): Kernel type for kernel density estimation
        pdf (array_like, None): Samples from target distribution
        N_plots_max (int): Maximum number of plots
    """
    N_plots = None
    scale_factor = None

    if use_real:
        if not dm.use_latent:
            return
        N_plots = dm.lt.A.shape[0]
    else:
        N_plots = dm.particles.shape[0]
    if N_plots > 6:
        scale_factor = 15
    else:
        scale_factor = 5

    N_plots = min(N_plots, N_plots_max)

    plt.figure(figsize=(3 * scale_factor, (N_plots // 3 + 1) * scale_factor))

    for idx in range(N_plots):
        slice_dim = idx

        plt.subplot(N_plots // 3 + 1, 3, idx + 1)

        particles = None
        if use_real:
            particles = dm.lt.transform(dm.particles, n_particles_second=True).t()[:, slice_dim]
        else:
            particles = dm.particles.t()[:, slice_dim]

        if pdf is not None:
            plt.plot(np.linspace(-10, 10, len(pdf), dtype=np.float64), pdf)
        plt.plot(particles.data.cpu().numpy(), torch.zeros_like(particles).data.cpu().numpy(), 'ro')
        sns.kdeplot(particles.data.cpu().numpy(),
                    kernel=kernel, color='darkblue', linewidth=4)
    plt.show()


def plot_condition_distribution(dm, n_samples):
    """
    Args:
        dm (DistributionMover): object contains unconditioned density, linear manifold and particles
        n_samples (int): number of samples
    Return:
        (points, weight)
    """
    if not dm.use_latent:
        return

    points = torch.zeros([dm.n_hidden_dims, n_samples], dtype=t_type, device=device).uniform_(-10, 10)
    weight = dm.real_target_density(dm.lt.transform(points, n_particles_second=True))
    points = points.view(-1)

    plt.hist(points.data.cpu().numpy(), weights=weight.data.cpu().numpy(), density=True, bins=100, alpha=0.5,
             label='True conditional density')
    plt.plot(dm.particles.data.cpu().numpy(), torch.zeros_like(dm.particles).data.cpu().numpy(), 'ro')
    sns.kdeplot(dm.particles[0, :].data.cpu().numpy(),
                kernel='tri', color='darkblue', linewidth=4, label='Approximated conditional density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def pairwise_diffs(x, y, n_particles_second=True):
    """
    Args:
        if n_particles_second == True:
            Input: x is a dxN matrix
                   y is an optional dxM matrix
            Output: diffs is a dxNxM matrix where diffs[i,j] is the subtraction between x[:,i] and y[:,j]
            i.e. diffs[i,j] = x[:,i] - y[:,j]

        if n_particles_second == False:
            Input: x is a Nxd matrix
                   y is an optional Mxd matrix
            Output: diffs is a NxMxd matrix where diffs[i,j] is the subtraction between x[i,:] and y[j,:]
            i.e. diffs[i,j] = x[i,:]-y[j,:]
    """
    if n_particles_second:
        return x[:, :, np.newaxis] - y[:, np.newaxis, :]
    return x[:, np.newaxis, :] - y[np.newaxis, :, :]


def pairwise_dists(diffs=None, n_particles_second=True):
    """
    Args:
        if n_particles_second == True:
            Input: diffs is a dxNxM matrix where diffs[i,j] = x[:,i] - y[:,j]
            Output: dist is a NxM matrix where dist[i,j] is the square norm of diffs[i,j]
            i.e. dist[i,j] = ||x[:,i] - y[:,j]||

        if n_particles_second == False:
            Input: diffs is a NxMxd matrix where diffs[i,j] = x[i,:] - y[j, :]
            Output: dist is a NxM matrix where dist[i,j] is the square norm of diffs[i,j]
            i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    if n_particles_second:
        return torch.norm(diffs, dim=0)
    return torch.norm(diffs, dim=2)


class normal_density():
    """
        Multinomial normal density for independent random variables.
    """

    def __init__(self, n=1, mu=0., std=1., n_particles_second=True):
        """
        Args:
            n (int): number of dimensions of multinomial normal distribution
                Default: 1
            mu (float, array_like): mean of distribution
                Default: 0.
                if mu is float - use same mean across all dimensions
                if mu is 1D array_like - use different mean for each dimension but same for each particles dimension
                if mu is 2D array_like - use different mean for each dimension
            std (float, array_like): std of distribution
                Default: 1.
                if std is float - use same std across all dimensions
                if std is 1D array_like - use different std for each dimension but same for each particles dimension
                if std is 2D array_like - use different std for each dimension
            n_particles_second (bool): specify type of input
                Default: True
                if n_particles_second == True - input must has shape [n, n_particles]
                if n_particles_second == False - input must has shape [n_particles, n]
                Therefore the same mu and std are applied along particles axis
                Input will be reduced along all axis exclude particle axis
        """

        self.n = torch.tensor(n, dtype=t_type, device=device)
        self.n_particles_second = n_particles_second

        self.mu = mu
        self.std = std

        if isinstance(self.mu, float):
            self.mu = torch.tensor(self.mu, dtype=t_type, device=device).expand(n)
        if len(self.mu.shape) == 1:
            if self.mu.shape[0] == 1:
                self.mu = self.mu.view(1).expand(n)
            if self.n_particles_second:
                self.mu = torch.tensor(self.mu, dtype=t_type, device=device).view(n, 1)
            else:
                self.mu = torch.tensor(self.mu, dtype=t_type, device=device).view(1, n)
        elif len(self.mu.shape) == 2:
            self.mu = torch.tensor(self.mu, dtype=t_type, device=device)
        else:
            raise RuntimeError

        if isinstance(self.std, float):
            self.std = torch.tensor(self.std, dtype=t_type, device=device).expand(n)
        if len(self.std.shape) == 1:
            if len(self.std) == 1:
                self.std = self.std.view(1).expand(n)
            if self.n_particles_second:
                self.std = torch.tensor(self.std, dtype=t_type, device=device).view(n, 1)
            else:
                self.std = torch.tensor(self.std, dtype=t_type, device=device).view(1, n)
        elif len(self.std.shape) == 2:
            self.std = torch.tensor(self.std, dtype=t_type, device=device)
        else:
            raise RuntimeError

        self.zero = torch.tensor(0., dtype=t_type, device=device)
        self.one = torch.tensor(1., dtype=t_type, device=device)
        self.two = torch.tensor(2., dtype=t_type, device=device)
        self.pi = torch.tensor(math.pi, dtype=t_type, device=device)

        ### specify axis to reduce
        ### if n_particles_second == True - n_axis == 0
        ### if n_particles_second == False - n_axis == 1
        self.n_axis = 1 - int(self.n_particles_second)

    def __call__(self, x, n_axis=None):
        """
            Evaluate density in given point
        Args:
            x (torch.tensor): tensor which defines point where density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return (torch.pow(self.two * self.pi, -self.n / self.two) /
                torch.prod(self.std, dim=n_axis) *
                torch.exp(-self.one / self.two * torch.sum(torch.pow((x - self.mu) / self.std, self.two), dim=n_axis)))

    def unnormed_density(self, x, n_axis=None):
        """
            Evaluate unnormed density in given point
        Args:
            x (torch.tensor): tensor which defines point where unnormed density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return torch.exp(-self.one / self.two * torch.sum(torch.pow((x - self.mu) / self.std, self.two), dim=n_axis))

    def log_density(self, x, n_axis=None):
        """
            Evaluate log density in given point
        Args:
            x (torch.tensor): tensor which defines point where log density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return (-self.n / self.two * torch.log(self.two * self.pi) +
                torch.sum(torch.log(self.std), dim=n_axis) -
                self.one / self.two * torch.sum(torch.pow((x - self.mu) / self.std, self.two), dim=n_axis))

    def log_unnormed_density(self, x, n_axis=None):
        """
            Evaluate log unnormed density in given point
        Args:
            x (torch.tensor): tensor which defines point where log unnormed density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return -self.one / self.two * torch.sum(torch.pow((x - self.mu) / self.std, self.two), dim=n_axis)

    def get_sample(self):
        """
            Sample from normal distribution
        """
        sample = torch.normal(self.mu, self.std)
        if self.n_particles_second:
            return sample.view(-1, 1)
        else:
            return sample.view(1, -1)


class gamma_density():
    """
        Multinomial gamma density for independent random variables.
    """

    def __init__(self, n=1, alpha=1, betta=1, n_particles_second=True):
        """
        Args:
            n (int): number of dimensions of multinomial normal distribution
                Default: 1
            alpha (float, array_like): shape of distribution
                Default: 1.
                if alpha is float - use same shape across all dimensions
                if alpha is 1D array_like - use different shape for each dimension but same for each particles dimension
                if alpha is 2D array_like - use different shape for each dimension
            betta (float, array_like): rate of distribution
                Default: 1.
                if betta is float - use same rate across all dimensions
                if betta is 1D array_like - use different rate for each dimension but same for each particles dimension
                if betta is 2D array_like - use different rate for each dimension
            n_particles_second (bool): specify type of input
                Default: True
                if n_particles_second == True - input must has shape [n, n_particles]
                if n_particles_second == False - input must has shape [n_particles, n]
                Therefore the same mu and std are applied along particles axis
        """

        self.n = torch.tensor(n, dtype=t_type, device=device)
        self.n_particles_second = n_particles_second

        self.alpha = alpha
        self.betta = betta

        if isinstance(self.alpha, float):
            self.alpha = torch.tensor(self.alpha, dtype=t_type, device=device).expand(n)
        if len(self.alpha.shape) == 1:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha.view(1).expand(n)
            if self.n_particles_second:
                self.alpha = torch.tensor(self.alpha, dtype=t_type, device=device).view(n, 1)
            else:
                self.alpha = torch.tensor(self.alpha, dtype=t_type, device=device).view(1, n)
        elif len(self.alpha.shape) == 2:
            self.alpha = torch.tensor(self.alpha, dtype=t_type, device=device)
        else:
            raise RuntimeError

        if isinstance(self.betta, float):
            self.betta = torch.tensor(self.betta, dtype=t_type, device=device).expand(n)
        if len(self.betta.shape) == 1:
            if len(self.betta) == 1:
                self.betta = self.betta.view(1).expand(n)
            if self.n_particles_second:
                self.betta = torch.tensor(self.betta, dtype=t_type, device=device).view(n, 1)
            else:
                self.betta = torch.tensor(self.betta, dtype=t_type, device=device).view(1, n)
        elif len(self.std.shape) == 2:
            self.betta = torch.tensor(self.betta, dtype=t_type, device=device)
        else:
            raise RuntimeError

        self.one = torch.tensor(1., dtype=t_type, device=device)
        self.two = torch.tensor(2., dtype=t_type, device=device)

        ### specify axis to reduce
        ### if n_particles_second == True - n_axis == 0
        ### if n_particles_second == False - n_axis == 1
        self.n_axis = 1 - int(self.n_particles_second)

        ### log Г(alpha)
        self.lgamma = torch.lgamma(self.alpha)
        ### Г(alpha)
        self.gamma = torch.exp(self.lgamma)

    def __call__(self, x, n_axis=None):
        """
            Evaluate density in given point
        Args:
            x (torch.tensor): tensor which defines point where density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return (torch.prod(torch.pow(self.betta, self.alpha) / self.gamma * torch.pow(x, self.alpha - self.one),
                           dim=n_axis) *
                torch.exp(-torch.sum(self.betta * x, dim=n_axis)))

    def unnormed_density(self, x, n_axis=None):
        """
            Evaluate unnormed density in given point
        Args:
            x (torch.tensor): tensor which defines point where unnormed density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return (torch.prod(torch.pow(x, self.alpha - self.one), dim=n_axis) *
                torch.exp(-torch.sum(self.betta * x, dim=n_axis)))

    def log_density(self, x, n_axis=None):
        """
            Evaluate log density in given point
        Args:
            x (torch.tensor): tensor which defines point where log density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return (torch.sum(self.alpha * torch.log(self.betta) - self.lgamma + (self.alpha - self.one) * torch.log(x),
                          dim=n_axis) -
                torch.sum(self.betta * x, dim=n_axis))

    def log_unnormed_density(self, x, n_axis=None):
        """
            Evaluate log unnormed density in given point
        Args:
            x (torch.tensor): tensor which defines point where log unnormed density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return (torch.sum((self.alpha - self.one) * torch.log(x), dim=n_axis) -
                torch.sum(self.betta * x, dim=n_axis))

    def log_density_log_x(self, log_x, n_axis=None):
        """
            Evaluate log density in point log(x)
        Args:
            x (torch.tensor): tensor which defines point where log density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return (torch.sum(self.alpha * torch.log(self.betta) - self.lgamma + (self.alpha - self.one) * log_x,
                          dim=n_axis) -
                torch.sum(self.betta * torch.exp(log_x), dim=n_axis))

    def log_unnormed_density_log_x(self, log_x, n_axis=None):
        """
            Evaluate log unnormed density in point log(x)
        Args:
            x (torch.tensor): tensor which defines point where log unnormed density is evaluated
            n_axis (int): specify axis to reduce
                Default:
                    if n_particles_second == True - n_axis == 0
                    if n_particles_second == False - n_axis == 1
        """
        n_axis = self.n_axis if n_axis is None else n_axis
        return (torch.sum((self.alpha - self.one) * log_x, dim=n_axis) -
                torch.sum(self.betta * torch.exp(log_x), dim=n_axis))


class SteinLinear(nn.Module):
    """
        Custom full connected layer for Stein Gradient Neural Networks
        Transformation: y = xA + b
        Parameters prior:
            1 - p(w, a) = p(w|a)p(a) = П p(w_i|a_i)p(a_i)
                p(w_i|a_i) = N(w_i|0, a_i^(-1)); p(a_i) = G(1e-4, 1e-4)

            2 - p(w) = П p(w_i)
                p(w_i) = N(w_i|0, alpha^(-1))
    """

    def __init__(self, in_features, out_features, n_particles=1, use_bias=True, use_var_prior=True, alpha=1e-2):
        super(SteinLinear, self).__init__()
        """
        Args:
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            n_particles (int): number of particles
            use_bias (bool): If set to False, the layer will not learn an additive bias.
                Default: True
            use_var_prior (bool): If set to True, use Gamma prior distribution of weight variance
                Default: True
            alpha (float): If use_var_prior == False - defines weight variance
                Default: 1e-2
        """

        self.in_features = in_features
        self.out_features = out_features
        self.n_particles = n_particles
        self.use_bias = use_bias
        self.use_var_prior = use_var_prior

        ### if alpha is None use GLOROT prior
        if alpha is None:
            self.alpha_weight = (self.in_features + self.out_features) / 2.
            self.alpha_bias = (self.out_features) / 2.
        else:
            self.alpha_weight = alpha
            self.alpha_bias = alpha

        self.weight = torch.nn.Parameter(
            torch.zeros([in_features, out_features, n_particles], dtype=t_type, device=device))
        if self.use_var_prior:
            self.log_weight_alpha = torch.nn.Parameter(
                torch.zeros([in_features * out_features, n_particles], dtype=t_type, device=device))
        else:
            self.log_weight_alpha = torch.tensor([math.log(self.alpha_weight)], dtype=t_type, device=device,
                                                 requires_grad=False)

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros([1, out_features, n_particles], dtype=t_type, device=device))
            if self.use_var_prior:
                self.log_bias_alpha = torch.nn.Parameter(
                    torch.zeros([out_features, n_particles], dtype=t_type, device=device))
            else:
                self.log_bias_alpha = torch.tensor([math.log(self.alpha_bias)], dtype=t_type, device=device,
                                                   requires_grad=False)

        if self.use_var_prior:
            ### define prior on alpha p(a) = G(1e-4, 1e-4)
            self.weight_alpha_log_prior = lambda x: (gamma_density(n=self.log_weight_alpha.shape[0],
                                                                   alpha=1e-4,
                                                                   betta=1e-4,
                                                                   n_particles_second=True
                                                                   ).log_unnormed_density_log_x(x))
            if self.use_bias:
                self.bias_alpha_log_prior = lambda x: (gamma_density(n=self.log_bias_alpha.shape[0],
                                                                     alpha=1e-4,
                                                                     betta=1e-4,
                                                                     n_particles_second=True
                                                                     ).log_unnormed_density_log_x(x))

        self.one = torch.tensor(1., dtype=t_type, device=device)

        self.reset_parameters()

    ### useless function - all initialization defined in DistributionMover class
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.log_weight_alpha.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.log_bias_alpha.data.uniform_(-stdv, stdv)

    def forward(self, X):
        """
            Apply transformation: X_out[i, :, :] = X_in[i, :, :] * W + b[i]
        Args:
            X (torch.tensor): tensor
        Shape:
            Input: [n_particles, batch_size, in_features]
            Output: [n_particles, batch_size, out_features]
        """
        ### NEED SOME OPTIMIZATION TO OMMIT .permute
        if self.use_bias:
            return torch.bmm(X, self.weight.permute(2, 0, 1)) + self.bias.permute(2, 0, 1)
        return torch.bmm(X, self.weight.permute(2, 0, 1))

    def numel(self, trainable=True):
        """
            Count parameters in layer
        Args:
            trainable (bool): If set to False, the number of all trainable and not trainable parameters will be returned
                Default: True
        """
        if trainable:
            return sum(param.numel() for param in self.parameters() if param.requires_grad)
        else:
            return sum(param.numel() for param in self.parameters())

    def calc_log_prior(self):
        """
            Evaluate log prior of trainable parameters
        log p(w,a) = log p(w|a) + log p(a)
        """
        ### define prior on weight p(w|a) = N(0, 1 / alpha)
        weight_log_prior = lambda x: (normal_density(n=self.weight.numel() // self.n_particles,
                                                     mu=0.,
                                                     std=self.one / torch.exp(self.log_weight_alpha),
                                                     n_particles_second=True
                                                     ).log_unnormed_density(x))

        bias_log_prior = lambda x: (normal_density(self.bias.numel() // self.n_particles,
                                                   mu=0.,
                                                   std=self.one / torch.exp(self.log_bias_alpha),
                                                   n_particles_second=True
                                                   ).log_unnormed_density(x))

        if self.use_bias:
            if self.use_var_prior:
                return (weight_log_prior(self.weight.view(-1, self.n_particles)) + self.weight_alpha_log_prior(
                    self.log_weight_alpha) +
                        bias_log_prior(self.bias.view(-1, self.n_particles)) + self.bias_alpha_log_prior(
                            self.log_bias_alpha))
            return (weight_log_prior(self.weight.view(-1, self.n_particles)) +
                    bias_log_prior(self.bias.view(-1, self.n_particles)))

        if self.use_var_prior:
            return weight_log_prior(self.weight.view(-1, self.n_particles)) + self.weight_alpha_log_prior(
                self.log_weight_alpha)
        return weight_log_prior(self.weight.view(-1, self.n_particles))


class LinearTransform():
    """
        Class for various linear transformations
    """

    def __init__(self, n_dims, n_hidden_dims, use_identity=False, normalize=False, A=None, theta_0=None):
        """
        Args:
            n_dims (int): dimension of the space
            n_hidden_dims (int): dimension of the latent space
            use_identity (bool): If set to True, use 'eye' matrix for transformations
            normalize (bool): If set to True, columns of the transformation matrix is an orthonormal basis
            A (2D array_like, None): Initial value for transformation matrix
                If None then matrix will be sampled from uniform distribution and then orthonormate
                Default: None
            theta_0 (1D array_like, None): Initial value for bias
                If None then matrix will be sampled from uniform distribution
                Default: None
        """
        self.n_dims = n_dims
        self.n_hidden_dims = n_hidden_dims
        self.use_identity = use_identity
        self.normalize = normalize

        if self.use_identity:
            return

        self.A = A
        self.theta_0 = theta_0

        if self.A is None:
            self.A = torch.zeros([self.n_dims, self.n_hidden_dims], dtype=t_type, device=device)
            self.A.uniform_(-1., 1.)
            if self.normalize:
                ### normalize columns of matrix A
                self.A = torch.tensor(orth(self.A.data.cpu().numpy()), dtype=t_type, device=device)

        if self.theta_0 is None:
            self.theta_0 = torch.zeros([self.n_dims, 1], dtype=t_type, device=device)
            self.theta_0.uniform_(-1., 1.)

        ### A^(t)A
        self.AtA = torch.matmul(self.A.t(), self.A)
        ### (A^(t)A)^(-1)
        self.AtA_1 = torch.inverse(self.AtA)
        ### (A^(t)A)^(-1)A^(t)
        self.inverse_base = torch.matmul(self.AtA_1, self.A.t())

    def transform(self, theta, n_particles_second=True):
        """
            Transform thetas as follows:
                theta = Atheta` + theta_0
        """
        if self.use_identity:
            return theta
        if n_particles_second:
            return torch.matmul(self.A, theta) + self.theta_0
        return (torch.matmul(self.A, theta.t()) + self.theta_0).t()

    def inverse_transform(self, theta, n_particles_second=True):
        """
            Apply inverse transformation:
                theta` = (A^(t)A)^(-1)A^(t)(theta - theta_0)
        """
        if self.use_identity:
            return theta
        if n_particles_second:
            return torch.matmul(self.inverse_base, theta - self.theta_0)
        return torch.matmul(self.inverse_base, theta.t() - self.theta_0).t()

    def project_inverse(self, theta, n_particles_second=True):
        """
            Project and then apply inverse transform to theta - theta_0:
                theta_s_p_i = T^(-1)P(theta - theta_0)= (A^(t)A)^(-1)A^(t)theta
        """
        if self.use_identity:
            return theta
        ### This optimization severely reduces performance!!!!
        ### use solver trick: theta_s_p_i : A^(t)Atheta_s_p_i = A^(t)theta
        if n_particles_second:
            # return torch.gesv(torch.matmul(self.A.t(), theta), self.AtA)[0]
            return torch.matmul(self.inverse_base, theta)
        # return torch.gesv(torch.matmul(self.A.t(), theta.t()), self.AtA)[0].t()
        return torch.matmul(theta, self.inverse_base.t())

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.
        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Returns:
            dict:
                a dictionary containing a whole state of the module
        Example::
            >>> module.state_dict().keys()
            ['bias', 'weight']
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = dict(version=1)

        destination[prefix + 'A'] = self.A if keep_vars else self.A.data
        destination[prefix + 'theta_0'] = self.theta_0 if keep_vars else self.theta_0.data
        destination[prefix + 'n_dims'] = self.n_dims
        destination[prefix + 'n_hidden_dims'] = self.n_hidden_dims
        destination[prefix + 'use_identity'] = self.use_identity
        destination[prefix + 'normalize'] = self.normalize

        return destination

    def load_state_dict(self, state_dict, prefix=''):
        self.A.copy_(state_dict[prefix + 'A'])
        self.theta_0.copy_(state_dict[prefix + 'theta_0'])
        self.__init__(state_dict[prefix + 'n_dims'],
                      state_dict[prefix + 'n_hidden_dims'],
                      state_dict[prefix + 'use_identity'],
                      state_dict[prefix + 'normalize'],
                      self.A, self.theta_0
                      )


class RegressionDistribution(nn.Module):
    """
        Distribution over data for regression task: p(D|w) = N(y_predicted|y, )
    """

    def __init__(self, n_particles, use_var_prior=True, betta=1e-1):
        super(RegressionDistribution, self).__init__()
        """
        Args:
            n_particles (int): number of particles
            use_var_prior (bool): If set to True, use Gamma prior distribution of prediction variance
                Default: True
            betta (float): If use_var_prior == False - defines variance of prediction
                Default: 1e-1
        """

        self.n_particles = n_particles
        self.use_var_prior = use_var_prior
        self.betta = betta

        ### define betta - variance of data distribution for regression tasks (betta = 1 / std ** 2)
        if self.use_var_prior:
            self.log_betta = torch.nn.Parameter(torch.zeros([1, self.n_particles], dtype=t_type, device=device))
        else:
            self.log_betta = torch.tensor([math.log(self.betta)], dtype=t_type, device=device, requires_grad=False)

        if self.use_var_prior:
            ### define prior on betta p(betta)
            self.betta_log_prior = lambda x: (gamma_density(n=1,
                                                            alpha=1e-4,
                                                            betta=1e-4,
                                                            n_particles_second=True
                                                            ).log_unnormed_density_log_x(x))

        ### Support tensors for computations
        self.one = torch.tensor(1., dtype=t_type, device=device)

    def calc_log_data(self, X, y, y_predict, train_size):
        """
            Evaluate log p(theta)
        Args:
            X (array_like): batch of data
            y (array_like): batch of target values
            y_predict (array_like): batch of predicted values
            train_size (int) - size of training dataset
        Shapes:
            y.shape = [batch_size]
            y_predict.shape = [n_particles, batch_size, 1]
        """
        ### squeeze last axis because regression task is being solved
        y_predict.squeeze_(2)

        batch_size = torch.tensor(y.shape[0], dtype=t_type, device=device)
        train_size = torch.tensor(train_size, dtype=t_type, device=device)

        ### define distribution over data p(D|w)
        ### n_particles_second == False because y_predict has shape = [n_particles, batch_size]
        log_data_distr = None
        if self.use_var_prior:
            log_data_distr = lambda x: (normal_density(n=X.shape[0],
                                                       mu=y,
                                                       std=self.one / torch.sqrt(torch.exp(
                                                           self.log_betta.expand(X.shape[0], self.n_particles).t())),
                                                       n_particles_second=False
                                                       ).log_unnormed_density(x))
        else:
            log_data_distr = lambda x: (normal_density(n=X.shape[0],
                                                       mu=y,
                                                       std=self.one / torch.sqrt(torch.exp(self.log_betta)),
                                                       n_particles_second=False
                                                       ).log_unnormed_density(x))

        if self.use_var_prior:
            return train_size / batch_size * log_data_distr(y_predict) + self.betta_log_prior(self.log_betta)
        return train_size / batch_size * log_data_distr(y_predict)

    def modules(self):
        yield self

    def numel(self, trainable=True):
        """
            Count parameters in layer
        Args:
            trainable (bool): If set to False, the number of all trainable and not trainable parameters will be returned
                Default: True
        """
        if trainable:
            return sum(param.numel() for param in self.parameters() if param.requires_grad)
        else:
            return sum(param.numel() for param in self.parameters())


class ClassificationDistribution(nn.Module):
    def __init__(self, n_particles):
        super(ClassificationDistribution, self).__init__()
        """
        Args:
            n_particles (int): number of particles
        """
        self.n_particles = n_particles

    def calc_log_data(self, X, y, y_predict, train_size):
        """
            Evaluate log p(theta)
        Args:
            X (array_like): batch of data
            y (array_like): batch of target values
            y_predict (array_like): batch of predictions
            train_size (int): size of train dataset
        Shapes:
            X.shape = [batch_size, in_features]
            y.shape = [batch_size]
            y_predict.shape = [n_particles, batch_size, n_classes]
        """
        batch_size = torch.tensor(X.shape[0], dtype=t_type, device=device)
        train_size = torch.tensor(train_size, dtype=t_type, device=device)

        ### define distribution over data p(D|w)
        ### n_particles_second == False because y_predict has shape = [n_particles, batch_size]

        probas = nn.LogSoftmax(dim=2)(y_predict)
        probas_selected = torch.gather(input=probas, dim=2,
                                       index=y.view(1, -1, 1).expand(probas.shape[0], probas.shape[1], 1)).squeeze(2)
        log_data = torch.sum(probas_selected, dim=1)

        return train_size / batch_size * log_data

    def modules(self):
        yield self

    def numel(self, trainable=True):
        """
            Count parameters in layer
        Args:
            trainable (bool): If set to False, the number of all trainable and not trainable parameters will be returned
                Default: True
        """
        if trainable:
            return sum(param.numel() for param in self.parameters() if param.requires_grad)
        else:
            return sum(param.numel() for param in self.parameters())


class DistributionMover(nn.Module):
    def __init__(self,
                 task='app',
                 n_particles=None,
                 particles=None,
                 target_density=None,
                 n_dims=None,
                 n_hidden_dims=None,
                 use_latent=False,
                 net=None,
                 precomputed_params=None,
                 data_distribution=None
                 ):
        super(DistributionMover, self).__init__()
        """
        Args:
            task (str):
                'app' | 'net_reg' | 'net_class'
                - approximate target distribution
                - solve regression task using net
                - solve classification task using net
            n_particles (int): number of particles
            particles (2D array_like): array which contains initialized particles
            target_density (callable): computes probability density function of target distribution (only for 'app' task)
            n_dims (int): dimension of the space where optimization is performed
            n_hidden_dims (int): dimension of the latent space
            use_latent (bool): If set to True, Subspace Stein is used
            acr (list): List contains arcitecture of object which is used to make predictions (for 'net_reg' and' net_class' tasks)
            precomputed_params (1D array_like): Precomputed parameters, which will be used for particles initialization
            data_distribution (callable): computes probability over data p(D|w) (for 'net_reg' and' net_class' tasks)
        """

        self.task = task

        self.n_particles = n_particles
        self.particles = particles
        self.target_density = target_density
        self.n_dims = n_dims
        self.n_hidden_dims = n_hidden_dims
        self.use_latent = use_latent
        self.net = net
        self.precomputed_params = precomputed_params
        self.data_distribution = data_distribution

        if self.task == 'net_reg' or self.task == 'net_class':
            self.n_dims = self.numel() // self.n_particles
        if not self.use_latent:
            self.n_hidden_dims = self.n_dims

        ### Learnable samples from the target distribution
        self.particles = torch.zeros(
            [self.n_hidden_dims, self.n_particles],
            dtype=t_type,
            requires_grad=False,
            device=device).uniform_(-2., 2.)

        ### Class for performing linear transformations
        if self.use_latent:
            self.lt = LinearTransform(
                n_dims=self.n_dims,
                n_hidden_dims=self.n_hidden_dims,
                use_identity=False,
                normalize=True
            )
        else:
            self.lt = LinearTransform(
                n_dims=self.n_dims,
                n_hidden_dims=self.n_hidden_dims,
                use_identity=True,
                normalize=True
            )

        if self.precomputed_params is not None:
            self.particles = self.lt.inverse_transform(
                self.precomputed_params.unsqueeze(1).expand(self.n_dims, self.n_particles))

        ### Functions of probability density of target distribution
        if self.net is None:
            # use unnormed probability density to speedup computations
            if target_density is not None:
                self.target_density = target_density
                self.real_target_density = target_density
            else:
                self.target_density = lambda x, *args, **kwargs: (
                            0.3 * normal_density(self.n_dims, -2., 1., n_particles_second=True).unnormed_density(x,
                                                                                                                 *args,
                                                                                                                 **kwargs) +
                            0.7 * normal_density(self.n_dims, 2., 1., n_particles_second=True).unnormed_density(x,
                                                                                                                *args,
                                                                                                                **kwargs))

                self.real_target_density = lambda x, *args, **kwargs: (
                            0.3 * normal_density(self.n_dims, -2., 1., n_particles_second=True)(x, *args, **kwargs) +
                            0.7 * normal_density(self.n_dims, 2., 1., n_particles_second=True)(x, *args, **kwargs))

        ### Number of iterations since beginning
        self.iter = 0

        ### Adagrad parameters
        self.fudge_factor = torch.tensor(1e-6, dtype=t_type, device=device)
        self.step_size = torch.tensor(1e-2, dtype=t_type, device=device)
        self.auto_corr = torch.tensor(0.9, dtype=t_type, device=device)

        ### Gradient history term for adagrad optimization
        if self.use_latent:
            self.historical_grad = torch.zeros(
                [self.n_hidden_dims, n_particles], dtype=t_type, device=device)
            self.historical_grad_theta_0 = torch.zeros(
                [self.n_dims, 1], dtype=t_type, device=device)
        else:
            self.historical_grad = torch.zeros(
                [self.n_dims, n_particles], dtype=t_type, device=device)

        ### Factor from kernel
        self.h = torch.tensor(0., dtype=t_type, device=device)

        ### Support tensors for computations
        self.N = torch.tensor(self.n_particles, dtype=t_type, device=device)
        self.one = torch.tensor(1., dtype=t_type, device=device)
        self.two = torch.tensor(2., dtype=t_type, device=device)
        self.three = torch.tensor(3., dtype=t_type, device=device)

    def numel(self, trainable=True):
        """
            Count parameters in layer
        Args:
            trainable (bool): If set to False, the number of all trainable and not trainable parameters will be returned
                Default: True
        """
        cnt = 0
        for module in self.children():
            if 'numel' in dir(module):
                cnt += module.numel(trainable)
        return cnt

    def calc_kernel_term_latent(self, h_type, kernel_type='rbf', p=None):
        """
            Calculate k(*,*), grad(k(*,*))
        Args:
            h_type (int, float):
                If float then use h_type as kernel factor
                If int: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7:
                    0 - med(dist(theta-theta`)^2) / logN
                    1 - med(dist(theta-theta`)^2) / logN * n_dims
                    2 - med(dist(theta-theta`)) / logN * 2 * n_dims
                    3 - var(theta) / logN * 2 * n_dims
                    4 - var(diff(theta-theta`) / logN * n_dims
                    5 - med(dist(theta-theta`)^2) / (N^2 - 1)
                    6 - med(dist(theta-theta`)^2) / (N^(-1/p) - 1)
                    7 - -med(<Atheta`_i + theta_0, Atheta`_j + theta_0>) / logN
            kernel_type (std):
                'rbf' | 'imq' | 'exp' | 'rat'
                    - kernel[i, j] = exp(-1/h * ||A(theta`_i - theta`_j)||^2)
                    - kernel[i, j] = (1 + 1/h * ||A(theta`_i - theta`_j)||^2)^(-1/2)
                    - kernel[i, j] = exp(1/h * <Atheta`_i + theta_0, Atheta`_j + theta_0>)
                    - kernel[i, j] = (1 + 1/h * ||A(theta`_i - theta`_j)||^2)^(p)
                Default: 'rbf'
            p (double, None): power in rational kernel
                If kernel_type == 'rat' then p must be not None
                Default: None
        Shape:
            Output:
                ([n_particles, n_particles], [n_dims, n_particles, n_particles])
        """
        ### power for rational kernel
        p = torch.tensor(p, dtype=t_type, device=device) if p is not None else None

        ### theta = Atheta` + theta_0
        real_particles = self.lt.transform(self.particles, n_particles_second=True)
        ### diffs[i, j] = A(theta`_i - theta`_j)
        diffs = pairwise_diffs(real_particles, real_particles, n_particles_second=True)
        ### dists[i, j] = ||A(theta`_i - theta`_j)||
        dists = pairwise_dists(diffs=diffs, n_particles_second=True)
        ### sq_dists[i, j] = ||A(theta`_i - theta`_j)||^2
        sq_dists = torch.pow(dists, self.two)

        if type(h_type) == float:
            self.h = h_type
        elif h_type == 0:
            med = torch.median(sq_dists) + self.fudge_factor
            self.h = med / torch.log(self.N + 1)
        elif h_type == 1:
            med = torch.median(sq_dists) + self.fudge_factor
            self.h = med / torch.log(self.N + 1) * (self.n_dims)
        elif h_type == 2:
            med = torch.median(sq_dists) + self.fudge_factor
            self.h = med / torch.log(self.N + 1) * (2. * self.n_dims)
        elif h_type == 3:
            var = torch.var(self.particles) + self.fudge_factor
            self.h = var / torch.log(self.N + 1.) * (2. * self.n_dims)
        elif h_type == 4:
            var = torch.var(diffs) + self.fudge_factor
            self.h = var / torch.log(self.N + 1) * (self.n_dims)
        elif h_type == 5:
            med = torch.median(sq_dists) + self.fudge_factor
            self.h = med / (torch.pow(self.N, self.two) - self.one)
        elif h_type == 6:
            med = torch.median(sq_dists) + self.fudge_factor
            self.h = med / (torch.pow(self.N, -self.one / p) - self.one)
        elif h_type == 7:
            med = torch.median(torch.matmul(real_particles.t(), real_particles)) + self.fudge_factor
            self.h = med / torch.log(self.N + 1)

        kernel = None
        grad_kernel = None
        if kernel_type == 'rbf':
            ### RBF Kernel:
            ### kernel[i, j] = exp(-1/h * ||A(theta`_i - theta`_j)||^2)
            kernel = torch.exp(-self.one / self.h * sq_dists)
            ### grad_kernel[i, j] = -2/h * A(theta`_i - theta`_j) * kernel[i, j]
            grad_kernel = -self.two / self.h * kernel.unsqueeze(0) * diffs
        elif kernel_type == 'imq':
            ### IMQ Kernel:
            ### kernel[i, j] = (1 + 1/h * ||A(theta`_i - theta`_j)||^2)^(-1/2)
            kernel = torch.pow(self.one + self.one / self.h * sq_dists, -self.one / self.two)
            ### grad_kernel[i, j] = -1/h * A(theta`_i - theta`_j) * kernel^(3)[i, j]
            grad_kernel = -self.one / self.h * torch.pow(kernel, self.three).unsqueeze(0) * diffs
        elif kernel_type == 'exp':
            ### Exponential Kernel:
            ### kernel[i, j] = exp(1/h * <Atheta`_i + theta_0, Atheta`_j + theta_0>)
            kernel = torch.exp(self.one / self.h * torch.matmul(real_particles.t(), real_particles))
            ### grad_kernel[i, j] = 1/h * (Atheta`_j + theta_0) * kernel[i, j]
            grad_kernel = 1. / self.h * kernel.unsqueeze(0) * real_particles.unsqueeze(1)
        elif kernel_type == 'rat':
            ### RAT Kernel:
            ### kernel[i, j] = (1 + 1/h * ||A(theta`_i - theta`_j)||^2)^(p)
            kernel = torch.pow(self.one + self.one / self.h * sq_dists, p)
            ### grad_kernel[i, j] = p/h * A(theta`_i - theta`_j) * kernel^((p - 1)/p)[i, j]
            grad_kernel = p / self.h * torch.pow(kernel, (self.p - self.one) / p).unsqueeze(0) * diffs

        return kernel, grad_kernel

    def calc_kernel_term_latent_net(self, h_type, kernel_type='rbf', p=None):
        """
            Calculate k(*,*), grad(k(*,*))
        Args:
            h_type (int, float):
                If float then use h_type as kernel factor
                If int: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7:
                    0 - med(dist(theta-theta`)^2) / logN
                    1 - med(dist(theta-theta`)^2) / logN * n_dims
                    2 - med(dist(theta-theta`)) / logN * 2 * n_dims
                    3 - var(theta) / logN * 2 * n_dims
                    4 - var(diff(theta-theta`) / logN * n_dims
                    5 - med(dist(theta-theta`)^2) / (N^2 - 1)
                    6 - med(dist(theta-theta`)^2) / (N^(-1/p) - 1)
                    7 - -med(<Atheta`_i + theta_0, Atheta`_j + theta_0>) / logN
            kernel_type (std):
                'rbf' | 'imq' | 'exp' | 'rat'
                    - kernel[i, j] = exp(-1/h * ||A(theta`_i - theta`_j)||^2)
                    - kernel[i, j] = (1 + 1/h * ||A(theta`_i - theta`_j)||^2)^(-1/2)
                    - kernel[i, j] = exp(1/h * <Atheta`_i + theta_0, Atheta`_j + theta_0>)
                    - kernel[i, j] = (1 + 1/h * ||A(theta`_i - theta`_j)||^2)^(p)
                Default: 'rbf'
            p (double, None): power in rational kernel
                If kernel_type == 'rat' then p must be not None
                Default: None
        Shape:
            Output:
                ([n_particles, n_particles], [n_dims, n_particles, n_particles])
        """
        return self.calc_kernel_term_latent(h_type, kernel_type, p)

    def calc_log_prior_net(self):
        """
            Traverse all modules and evaluate weights log prior
        """
        log_prior = 0
        for module in self.children():
            if 'calc_log_prior' in dir(module):
                log_prior += module.calc_log_prior()
        return log_prior

    def calc_log_term_latent(self):
        """
            Calculate grad(log p(theta))
        Shape:
            Output: [n_dims, n_particles]
        """

        ### theta = A theta` + theta_0
        real_particles = self.lt.transform(self.particles.detach(), n_particles_second=True).requires_grad_(True)
        ### compute log data term log p(D|w)
        log_term = torch.log(self.target_density(real_particles))

        ### evaluate gradient with respect to trainable parameters
        for idx in range(self.n_particles):
            log_term[idx].backward(retain_graph=True)

        grad_log_term = real_particles.grad

        return grad_log_term

    def calc_log_term_latent_net(self, X, y, train_size):
        """
            Calculate grad(log p(theta))
        Args:
            X (torch.tensor): batch of data
            y (torch.tensor): batch of predictions
            train_size (int): size of train dataset
        Shape:
            Input:
                x.shape = [batch_size, in_features]
                y.shape = [batch_size, out_features]
            Output:
                [n_dims, n_particles]
        """
        log_data = torch.zeros([self.n_particles], dtype=t_type, device=device)
        log_prior = torch.zeros([self.n_particles], dtype=t_type, device=device)

        ### get real net parameters: theta_i = A theta`_i + theta_0
        real_particles = self.lt.transform(self.particles, n_particles_second=True)
        ### init net with real parameters
        self.vector_to_parameters(real_particles.view(-1), self.parameters_net())
        ### compute log prior of all weight in the net
        log_prior = self.calc_log_prior_net()

        ### get prediction for the batch of data
        y_predict = self.predict_net(X)
        ### compute log data term log p(D|w)
        log_data = self.data_distribution.calc_log_data(X, y, y_predict, train_size)

        ### log_term = log p(theta) = log p_prior(theta) + log p_data(D|theta)
        log_term = log_prior + log_data

        ### evaluate gradient with respect to trainable parameters
        for idx in range(self.n_particles):
            log_term[idx].backward(retain_graph=True)

        ### collect all gradients into one vector
        grad_log_term = self.parameters_grad_to_vector(self.parameters_net()).view(-1, self.n_particles)

        return grad_log_term

    def parameters_net(self):
        """
            Return all trainable parameters
        """
        return chain(self.net.parameters(), self.data_distribution.parameters())

    def predict_net(self, X, inference=False):
        """
            Use net to make predictions
            Args:
                X (array_like): batch of data
        """
        predictions = self.net(X.unsqueeze(0).expand(self.n_particles, *X.shape))
        if self.task == 'net_reg':
            if inference:
                return torch.mean(predictions, dim=0)
            else:
                return predictions
        elif self.task == 'net_class':
            if inference:
                return torch.log(torch.mean(torch.nn.Softmax(dim=2)(predictions), dim=0))
            else:
                return predictions

    def update_latent(self,
                      h_type, kernel_type='rbf', p=None,
                      step_size=None,
                      move_theta_0=False,
                      burn_in=False, burn_in_coeff=None,
                      epoch=None
                      ):
        self.step_size = step_size if step_size is not None else self.step_size
        self.epoch = epoch

        if burn_in:
            self.burn_in_coeff = torch.tensor(burn_in_coeff, dtype=t_type, device=device)
        else:
            self.burn_in_coeff = self.one

        self.iter += 1

        ### Compute additional terms
        kernel, grad_kernel = self.calc_kernel_term_latent(h_type, kernel_type, p)
        grad_log_term = self.calc_log_term_latent()

        ### Increase grad_log_term in burn_in_coeff times
        grad_log_term *= self.burn_in_coeff

        ### Compute value of step in functional space
        phi = (torch.matmul(grad_log_term, kernel) + torch.sum(grad_kernel, dim=1)) / self.N

        ### Transform phi from R^D space to R^d space: phi` = (A^(t)A)^(-1)A^(t)phi
        phi = self.lt.project_inverse(phi, n_particles_second=True)

        ### Update gradient history
        if self.iter == 1:
            self.historical_grad = self.historical_grad + phi * phi
        else:
            self.historical_grad = self.auto_corr * self.historical_grad + (self.one - self.auto_corr) * phi * phi

        ### Adjust gradient and make step
        adj_phi = phi / (self.fudge_factor + torch.sqrt(self.historical_grad))
        self.particles = self.particles + self.step_size * adj_phi

        ### Update theta_0 in LinearTransform
        if self.use_latent and move_theta_0:
            ### Compute value of step in functional space
            theta_0_update = torch.mean(grad_log_term, dim=1).view(-1, 1)

            ### Update gradient history
            if self.iter == 1:
                self.historical_grad_theta_0 = self.historical_grad_theta_0 + theta_0_update * theta_0_update
            else:
                self.historical_grad_theta_0 = self.auto_corr * self.historical_grad_theta_0 + (
                            self.one - self.auto_corr) * theta_0_update * theta_0_update

            ### Adjust gradient and make step
            adj_theta_0_update = theta_0_update / (self.fudge_factor + torch.sqrt(self.historical_grad_theta_0))
            self.lt.theta_0 = self.lt.theta_0 + self.step_size * adj_theta_0_update

    def update_latent_net(self,
                          h_type, kernel_type='rbf', p=None,
                          X_batch=None, y_batch=None, train_size=None,
                          step_size=None,
                          move_theta_0=False,
                          burn_in=False, burn_in_coeff=None,
                          epoch=None
                          ):
        self.step_size = step_size if step_size is not None else self.step_size
        self.epoch = epoch

        if burn_in:
            self.burn_in_coeff = torch.tensor(burn_in_coeff, dtype=t_type, device=device)
        else:
            self.burn_in_coeff = self.one

        self.iter += 1
        self.net.zero_grad()
        self.data_distribution.zero_grad()

        ### Compute additional terms
        kernel, grad_kernel = self.calc_kernel_term_latent_net(h_type, kernel_type, p)
        grad_log_term = self.calc_log_term_latent_net(X_batch, y_batch, train_size)

        ### Increase grad_log_term in burn_in_coeff times
        grad_log_term *= self.burn_in_coeff

        ### Compute value of step in functional space
        phi = (torch.matmul(grad_log_term, kernel) + torch.sum(grad_kernel, dim=1)) / self.N

        ### Transform phi from R^D space to R^d space: phi` = (A^(t)A)^(-1)A^(t)phi
        phi = self.lt.project_inverse(phi, n_particles_second=True)

        ### Update gradient history
        if self.iter == 1:
            self.historical_grad = self.historical_grad + phi * phi
        else:
            self.historical_grad = self.auto_corr * self.historical_grad + (self.one - self.auto_corr) * phi * phi

        ### Adjust gradient and make step
        adj_phi = phi / (self.fudge_factor + torch.sqrt(self.historical_grad))
        self.particles = self.particles + self.step_size * adj_phi

        ### Update theta_0 in LinearTransform
        if self.use_latent and move_theta_0:
            ### Compute value of step in functional space
            theta_0_update = torch.mean(grad_log_term, dim=1).view(-1, 1)

            ### Update gradient history
            if self.iter == 1:
                self.historical_grad_theta_0 = self.historical_grad_theta_0 + theta_0_update * theta_0_update
            else:
                self.historical_grad_theta_0 = self.auto_corr * self.historical_grad_theta_0 + (
                            self.one - self.auto_corr) * theta_0_update * theta_0_update

            ### Adjust gradient and make step
            adj_theta_0_update = theta_0_update / (self.fudge_factor + torch.sqrt(self.historical_grad_theta_0))
            self.lt.theta_0 = self.lt.theta_0 + self.step_size * adj_theta_0_update

    @staticmethod
    def vector_to_parameters(vec, parameters):
        pointer = 0
        for param in parameters:
            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.data = vec[pointer:pointer + num_param].view_as(param).data
            # Increment the pointer
            pointer += num_param

    @staticmethod
    def parameters_to_vector(parameters):
        vec = []
        for param in parameters:
            vec.append(param.view(-1))
        return torch.cat(vec)

    @staticmethod
    def parameters_grad_to_vector(parameters):
        vec = []
        for param in parameters:
            vec.append(param.grad.view(-1))
        return torch.cat(vec)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.
        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Returns:
            dict:
                a dictionary containing a whole state of the module
        """
        destination = super(DistributionMover, self).state_dict(destination, prefix, keep_vars)

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = dict(version=self._version)

        destination[prefix + 'particles'] = self.particles if keep_vars else self.particles.data
        destination[prefix + 'historical_grad'] = self.historical_grad if keep_vars else self.historical_grad.data
        if self.use_latent:
            destination[
                prefix + 'historical_grad_theta_0'] = self.historical_grad_theta_0 if keep_vars else self.historical_grad_theta_0.data
            self.lt.state_dict(destination, prefix + 'lt' + '.', keep_vars=keep_vars)
        destination[prefix + 'step_size'] = self.step_size
        destination[prefix + 'iter'] = self.iter
        destination[prefix + 'epoch'] = self.epoch

        return destination

    def load_state_dict(self, state_dict, prefix=''):
        super(DistributionMover, self).load_state_dict(state_dict, prefix)

        self.particles.copy_(state_dict[prefix + 'particles'])
        self.historical_grad.copy_(state_dict[prefix + 'historical_grad'])
        self.step_size = state_dict[prefix + 'step_size']
        self.iter = state_dict[prefix + 'iter']
        self.epoch = state_dict[prefix + 'epoch']

        if self.use_latent:
            self.historical_grad_theta_0.copy_(state_dict[prefix + 'historical_grad_theta_0'])
            self.lt.load_state_dict(state_dict, prefix + 'lt' + '.')


class LRStrategy:
    def __init__(self, step_size, factor=0.1, n_epochs=1, patience=10):
        """
            Multiply @step_size by factor each @n_epochs epochs
            Freeze @step_size after @patience epochs
        """
        self.step_size = step_size
        self.factor = factor
        self.n_epochs = n_epochs
        self.patience = patience
        self.iter = 0

    def step(self):
        self.iter += 1
        if self.iter < self.patience and self.iter % self.n_epochs == 0:
            self.step_size *= self.factor


### Add some methods to nn.Sequential to make code clear

setattr(nn.Sequential, "numel", DistributionMover.numel)
setattr(nn.Sequential, "calc_log_prior", DistributionMover.calc_log_prior_net)


def train(dm,
          dataloader_train, dataloader_test,
          lr_str, start_epoch, end_epoch, n_epochs_save=20, n_epochs_log=1,
          move_theta_0=False, plot_graphs=True, verbose=False,
          checkpoint_file_name=None, plots_file_name=None, log_file_name=None,
          n_warmup_epochs=16, n_previous=10
          ):
    ### Get all y_test in one tensor
    y_test_all = torch.tensor([], dtype=torch.int64, device=device)
    for _, y_test in dataloader_test:
        y_test = y_test.to(device=device)
        y_test_all = torch.cat([y_test_all, y_test.data.detach().clone()], dim=0)
    ### WARNING: May be incorrect if output features of dm.net != n_classes
    n_classes = len(dataloader_train.dataset.class_nums)

    ### Train loss/accuracy
    train_losses = []
    train_accs = []
    ### Test loss/accuracy
    test_losses = []
    test_accs = []
    ### Mean loss/accuracy from @n_warmup_epochs to current epoch
    test_losses_mean = []
    test_accs_mean = []
    predictions_test_cummulative = torch.zeros([1, 1], dtype=t_type, device=device)
    ### Mean loss/accuracy from (current epoch - n_previous)  to current epoch
    test_losses_mean_previous = []
    test_accs_mean_previous = []
    predictions_test_previous = torch.zeros([n_previous, y_test_all.shape[0], n_classes], dtype=t_type, device=device)
    ### Index of 'oldest' element in predictions_test_previous
    pointer_to_the_back = 0

    if log_file_name is not None:
        log_file = open(log_file_name, 'a')
        log_file.write('\rNew run of training.\r')
        log_file.close()

    try:
        for epoch in range(start_epoch, end_epoch):
            epoch_since_start = epoch - start_epoch

            ### One update of particles via all dataloader_train
            for X, y in dataloader_train:
                X = X.double().to(device=device).view(X.shape[0], -1)
                y = y.to(device=device)
                burn_in_coeff = max(1. - (1. - 1.) / 20. * epoch, 1.)
                dm.update_latent_net(h_type=0, kernel_type='rbf', p=None,
                                     X_batch=X, y_batch=y,
                                     train_size=len(dataloader_train.dataset),
                                     step_size=lr_str.step_size,
                                     move_theta_0=move_theta_0,
                                     burn_in=True, burn_in_coeff=burn_in_coeff,
                                     epoch=epoch
                                     )

            ### Evaluate cross entropy and accuracy over dataloader_train
            train_loss = 0.
            train_acc = 0.
            for X_train, y_train in dataloader_train:
                X_train = X_train.double().to(device=device).view(X.shape[0], -1)
                y_train = y_train.to(device=device)

                net_pred = dm.predict_net(X_train, inference=True)
                y_pred = torch.argmax(net_pred, dim=1)

                train_loss -= torch.sum(torch.gather(net_pred, 1, y_train.view(-1, 1)))
                train_acc += torch.sum(y_pred == y_train).float()
            train_loss /= (len(dataloader_train.dataset) + 0.)
            train_acc /= (len(dataloader_train.dataset) + 0.)

            ### Evaluate cross entropy and accuracy over dataloader_test
            test_loss = 0.
            test_acc = 0.
            predictions_test_current = torch.tensor([], dtype=t_type, device=device)
            for X_test, y_test in dataloader_test:
                X_test = X_test.double().to(device=device).view(X.shape[0], -1)
                y_test = y_test.to(device=device)

                ### Get output of net before Softmax, mean and log, Shape = [n_particles, batch_size, output_features]
                net_pred_pure = dm.predict_net(X_test, inference=False)
                net_pred_pure = torch.mean(torch.nn.Softmax(dim=2)(net_pred_pure), dim=0)
                predictions_test_current = torch.cat([predictions_test_current, net_pred_pure.data.detach().clone()],
                                                     dim=0)

                net_pred = torch.log(net_pred_pure)
                y_pred = torch.argmax(net_pred, dim=1)

                test_loss -= torch.sum(torch.gather(net_pred, 1, y_test.view(-1, 1)))
                test_acc += torch.sum(y_pred == y_test).float()
            test_loss /= (len(dataloader_test.dataset) + 0.)
            test_acc /= (len(dataloader_test.dataset) + 0.)

            ### Evaluate cross entropy and accuracy over dataloader_test using
            ### all predictions from previous (@epoch_since_start - @n_warmup_epochs) epochs
            test_loss_mean = 0.
            test_acc_mean = 0.
            if epoch_since_start >= n_warmup_epochs:
                predictions_test_cummulative = (
                        predictions_test_cummulative * (epoch_since_start - n_warmup_epochs) / (
                            epoch_since_start - n_warmup_epochs + 1.) +
                        predictions_test_current / (epoch_since_start - n_warmup_epochs + 1.))
                log_predictions_test = torch.log(predictions_test_cummulative)
                y_pred_all = torch.argmax(log_predictions_test, dim=1)

                test_loss_mean = -torch.sum(torch.gather(log_predictions_test, 1, y_test_all.view(-1, 1)))
                test_acc_mean = torch.sum(y_pred_all == y_test_all).float()
                test_loss_mean /= (len(dataloader_test.dataset) + 0.)
                test_acc_mean /= (len(dataloader_test.dataset) + 0.)

            ### Evaluate cross entropy and accuracy over dataloader_test using
            ### all predictions from previous @n_previous epochs
            test_loss_mean_previous = 0.
            test_acc_mean_previous = 0.
            predictions_test_previous[pointer_to_the_back] = predictions_test_current
            if pointer_to_the_back + 1 == n_previous:
                pointer_to_the_back = 0
            else:
                pointer_to_the_back += 1
            if epoch_since_start + 1 >= n_previous:
                log_predictions_test = torch.log(torch.mean(predictions_test_previous, dim=0))
                y_pred_all = torch.argmax(log_predictions_test, dim=1)
                test_loss_mean_previous = -torch.sum(torch.gather(log_predictions_test, 1, y_test_all.view(-1, 1)))
                test_acc_mean_previous = torch.sum(y_pred_all == y_test_all).float()
                test_loss_mean_previous /= (len(dataloader_test.dataset) + 0.)
                test_acc_mean_previous /= (len(dataloader_test.dataset) + 0.)

            ### Append evaluated losses and accuracies
            train_losses.append(train_loss.data[0].cpu().numpy())
            train_accs.append(train_acc.data[0].cpu().numpy())
            test_losses.append(test_loss.data[0].cpu().numpy())
            test_accs.append(test_acc.data[0].cpu().numpy())
            if epoch_since_start >= n_warmup_epochs:
                test_losses_mean.append(test_loss_mean.data[0].cpu().numpy())
                test_accs_mean.append(test_acc_mean.data[0].cpu().numpy())
            else:
                test_losses_mean.append(None)
                test_accs_mean.append(None)
            if epoch_since_start + 1 >= n_previous:
                test_losses_mean_previous.append(test_loss_mean_previous.data[0].cpu().numpy())
                test_accs_mean_previous.append(test_acc_mean_previous.data[0].cpu().numpy())
            else:
                test_losses_mean_previous.append(None)
                test_accs_mean_previous.append(None)

            ### Print log into console and file
            if epoch % n_epochs_log == 0:
                sys.stdout.write(
                    ('\nEpoch {0}... \t Step Size {1:.3f}\t Kernel factor: {2:.3f}\t Burn-in Coeff: {3:.3f}' +
                     '\nEmpirical Loss (Train/Test/Test (Mean (All))/Test (Mean (n_prev))): {4:.3f}/{5:.3f}/{6:.3f}/{7:.3f}' +
                     '\nAccuracy (Train/Test/Test (Mean (All))/Test (Mean (n_prev))): {8:.3f}/{9:.3f}/{10:.3f}/{11:.3f}\t'
                     ).format(epoch, lr_str.step_size, dm.h, dm.burn_in_coeff,
                              train_loss, test_loss, test_loss_mean, test_loss_mean_previous,
                              train_acc, test_acc, test_acc_mean, test_acc_mean_previous
                              )
                )
                if log_file_name is not None:
                    log_file = open(log_file_name, 'a')
                    log_file.write(
                        ('\nEpoch {0}... \t Step Size {1:.3f}\t Kernel factor: {2:.3f}\t Burn-in Coeff: {3:.3f}' +
                         '\nEmpirical Loss(Train/Test/Test (Mean (All))/Test (Mean (n_prev))): {4:.3f}/{5:.3f}/{6:.3f}/{7:.3f}' +
                         '\nAccuracy(Train/Test/Test (Mean (All))/Test (Mean (n_prev))): {8:.3f}/{9:.3f}/{10:.3f}/{11:.3f}\t'
                         ).format(epoch, lr_str.step_size, dm.h, dm.burn_in_coeff,
                                  train_loss, test_loss, test_loss_mean, test_loss_mean_previous,
                                  train_acc, test_acc, test_acc_mean, test_acc_mean_previous
                                  )
                    )
                    log_file.close()

            if epoch % n_epochs_save == 0 and epoch > start_epoch and checkpoint_file_name is not None:
                torch.save(dm.state_dict(), checkpoint_file_name.format(start_epoch, epoch))

            ### Update step_size
            lr_str.step()

    except KeyboardInterrupt:
        pass
    if plot_graphs:
        print_plots([[train_losses, test_losses, test_losses_mean, test_losses_mean_previous],
                     [train_accs, test_accs, test_accs_mean, test_accs_mean_previous]],
                    [['Epochs', ''],
                     ['Epochs', '% * 1e-2']],
                    [['Cross Entropy Loss (Train)', 'Cross Entropy Loss (Test)', 'Cross Entropy Loss (Test (Mean))',
                      'Cross Entropy Loss (Test (Mean (n_prev)))'],
                     ['Accuracy (Train)', 'Accuracy (Test)', 'Accuracy (Mean)', 'Accuracy (Mean (n_prev))']
                     ],
                    plots_file_name.format(start_epoch, epoch)
                    )
    if checkpoint_file_name is not None:
        torch.save(dm.state_dict(), checkpoint_file_name.format(start_epoch, epoch))

    if verbose:
        return (train_losses, test_losses, test_losses_mean, test_losses_mean_previous,
                train_accs, test_accs, test_accs_mean, test_accs_mean_previous)