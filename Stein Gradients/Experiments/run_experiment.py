import os
import torch
import argparse
import importlib

parser = argparse.ArgumentParser(description='Stein Gradient Experiments')
parser.add_argument('--config_name', type=str, default='config',
                    help='input name of config file (default: \'config\')')
parser.add_argument('--version', type=int, default=-1,
                    help='specify checkpoint to continue training process (default: None)')
args = parser.parse_args()

config = importlib.import_module('Configs.' + args.config_name)


use_cuda = False
device = None
if config.use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_device_id)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        use_cuda = True

from utils import *

# import MNIST dataset with class selection
sys.path.insert(0, '/home/m.nakhodnov/Samsung-Tasks/Datasets/MyMNIST')
from MyMNIST import MNIST_Class_Selection

# Set DoubleTensor as a base type for computations
t_type = torch.float64

if config.dataset == 'MNIST':
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])
    dataset_train = MNIST_Class_Selection('.', train=True, download=True, transform=transform)
    dataset_test = MNIST_Class_Selection('.', train=False, transform=transform)

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
else:
    raise RuntimeError

if config.net_arc == 'fc-300-100':
    net = nn.Sequential(
        SteinLinear(28 * 28, 300, config.n_particles, use_var_prior=config.use_var_prior, alpha=config.alpha),
        nn.Tanh(),
        SteinLinear(300, 100, config.n_particles, use_var_prior=config.use_var_prior, alpha=config.alpha),
        nn.Tanh(),
        SteinLinear(100, 10, config.n_particles, use_var_prior=config.use_var_prior, alpha=config.alpha)
    ).to(device=device)
elif config.net_arc == 'fc-18-14':
    net = nn.Sequential(
        SteinLinear(28 * 28, 18, config.n_particles, use_var_prior=config.use_var_prior, alpha=config.alpha),
        nn.Tanh(),
        SteinLinear(18, 14, config.n_particles, use_var_prior=config.use_var_prior, alpha=config.alpha),
        nn.Tanh(),
        SteinLinear(14, 10, config.n_particles, use_var_prior=config.use_var_prior, alpha=config.alpha)
    ).to(device=device)
else:
    raise RuntimeError

own_name = config.experiment_name
version = args.version if args.version > 0 else config.version
checkpoint_file_name = './Checkpoints/' + 'e{0}-{1}_' + own_name + '.pth'
plots_file_name = './Plots/' + 'e{0}-{1}_' + own_name + '.png'
log_file_name = './Logs/' + own_name + '.txt'

data_distr = ClassificationDistribution(config.n_particles)
lr_str = LRStrategy(step_size=0.03, factor=0.97, n_epochs=1, patience=80)

# If checkpoint exists do not create and perform any unnecessary computations in DistributionMover class
use_checkpoint = os.path.exists(checkpoint_file_name.format(0, version))
if use_checkpoint:
    dm = DistributionMover(
        task='net_class',
        n_particles=config.n_particles,
        n_hidden_dims=config.n_hidden_dims,
        use_latent=config.use_latent,
        net=net,
        data_distribution=data_distr,
        dummy=True
    )
    dm.load_state_dict(torch.load(checkpoint_file_name.format(0, version)))
    lr_str.step_size = dm.step_size
    lr_str.iter = dm.epoch + 1
else:
    dm = DistributionMover(
        task='net_class',
        n_particles=config.n_particles,
        n_hidden_dims=config.n_hidden_dims,
        use_latent=config.use_latent,
        net=net,
        data_distribution=data_distr,
        dummy=False
    )

try:
    if config.use_initializer and not use_checkpoint:
        initializer = torch.load(config.initializer_name)
        if config.init_theta_0:
            if 'particles' in initializer.keys():
                dm.lt.theta_0.data = initializer['particles'].data
            else:
                theta_0 = torch.tensor([], dtype=t_type, device=device)
                for name, weight in initializer.items():
                    if len(weight.shape) > 1:
                        # SteinLinear layer has weight matrix with shape [in_features, out_features]
                        # while nn.Linear has [out_features, in_features]
                        theta_0 = torch.cat([theta_0, weight.transpose(1, 0).contiguous().view(-1)], dim=0)
                    else:
                        theta_0 = torch.cat([theta_0, weight.view(-1)], dim=0)
                dm.lt.theta_0.data = theta_0.data.view(-1, 1)
                del theta_0
        # free memory
        del initializer
except:
    print('Initialization failed!')
    raise RuntimeError


train(dm=dm,
      dataloader_train=dataloader_train, dataloader_test=dataloader_test,
      lr_str=lr_str, start_epoch=lr_str.iter, end_epoch=lr_str.iter + config.n_epochs,
      n_epochs_save=config.n_epochs_save, n_epochs_log=config.n_epochs_log,
      move_theta_0=config.move_theta_0, plot_graphs=True, verbose=False,
      checkpoint_file_name=checkpoint_file_name, plots_file_name=plots_file_name, log_file_name=log_file_name,
      n_warmup_epochs=config.n_warmup_epochs, n_previous=config.n_previous,
      h_type=config.h_type, kernel_type=config.kernel_type, p=config.p
      )
