import os
import math
import torch
import argparse
import importlib
import xlsxwriter
from collections import namedtuple

use_cuda = False
device = None
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_cuda = True

from utils import *
from adv_utils import *

# import MNIST dataset with class selection
sys.path.insert(0, '/home/m.nakhodnov/Samsung-Tasks/Datasets/MyMNIST')
from MyMNIST import MNIST_Class_Selection

# Set DoubleTensor as a base type for computations
t_type = torch.float64


# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('Results Add.xlsx')
worksheets = [workbook.add_worksheet(), workbook.add_worksheet()]

# Name and colour of the corresponding row
experiments = [
    ('model_1', None),
    ('model_2', None),
    ('model_3', None),
    ('model_4', None),
    ('model_5', None),
    ('model_6', None),
    ('model_7', None),
    ('model_8', None),
    ('model_9', None),
    ('model_10', 'A5A4A4'),         # 146 epochs
    ('model_11', '#74FC74'),        # aka map estimate
    ('model_12', '#FCF567'),
    ('model_13', '#FCF567'),
    ('model_14', '#FCF567'),
    ('model_15', '#FCF567'),
    ('model_16', None),
    ('model_17', None),
    ('model_18', None),
    ('model_19', None),
    ('model_20', '#FCAC67'),
    ('model_21', '#FCAC67'),
    ('model_22', '#FCAC67'),
    ('model_23', '#FCAC67'),
    ('model_24', None),
    ('model_25', None),
    ('model_26', None),
    ('ml_est', '#FC6C67'),          # ml estimation
    ('ml_ensemble', '#FC6C67'),     # ensemble of 5 ml estimators
    ('model_30', '#FCAC67'),
    ('model_31', '#FCAC67'),
    ('model_28', '#FCAC67'),
    ('model_27', '#FCAC67'),
    ('model_29', '#FCAC67'),
    ('model_33', '#FCAC67'),
    ('model_34', '#FCAC67')
]

column_names = [
    'experiment_name',
    'dataset',
    'batch_size',
    'net_arc',
    'use_var_prior',
    'alpha',
    'n_particles',
    'use_latent',
    'n_hidden_dims',
    'n_epochs',
    'move_theta_0',
    'init_theta_0',
    'Error Rate (1 particle)',
    'Error Rate (all particles)',
    'Error Rate (augmentation)',
    'Cross Entropy (original)',
    'Cross Entropy (1 particle)',
    'Cross Entropy (all particles)',
    'Cross Entropy (augmentation)',
    'Mean Var (latent)',
    'Mean Var (original)',
    'Comment'
]

squeezed_column_names = [
    'experiment_name',
    'n_particles',
    'use_latent',
    'n_hidden_dims',
    'move_theta_0',
    'init_theta_0',
    'Error Rate (1 particle)',
    'Error Rate (all particles)',
    'Error Rate (augmentation)',
    'Cross Entropy (original)',
    'Cross Entropy (1 particle)',
    'Cross Entropy (all particles)',
    'Cross Entropy (augmentation)',
    'Mean Var (latent)',
    'Mean Var (original)',
    'Comment'
]


def to_field(name):
    return name.replace(' ', '_').replace('(', '').replace(')', '')


field_names = [to_field(name) for name in column_names]

Result = namedtuple('Result', field_names, verbose=False)
data = []


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset_m_test = MNIST_Class_Selection('.', train=False, transform=transform)
dataloader_m_test = DataLoader(dataset_m_test, batch_size=100, shuffle=False)

transform_aug = transforms.Compose([
    transforms.RandomAffine(degrees=90.,
                            translate=(0.25, 0.25),
                            scale=(0.8, 1.2)
                            ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset_m_test_aug = MNIST_Class_Selection('.', train=False, transform=transform_aug)
dataloader_m_test_aug = DataLoader(dataset_m_test_aug, batch_size=100, shuffle=False)


# Get data from config and log files
for exp_name, _ in experiments:

    config = importlib.import_module('Configs.config_' + exp_name)
    checkpoint_file_name = ('./Checkpoints/' + 'e{0}-{1}_' + config.experiment_name + '.pth').format(0, 199)
    if not os.path.exists(checkpoint_file_name):
        data.append(
            Result(
                config.experiment_name,
                config.dataset,
                config.batch_size,
                config.net_arc,
                config.use_var_prior,
                config.alpha,
                config.n_particles,
                config.use_latent,
                config.n_hidden_dims,
                config.n_epochs,
                config.move_theta_0,
                config.init_theta_0,
                None, None, None,
                None, None, None, None,
                None, None,
                config.comment
            )
        )
        continue

    print(exp_name)

    if exp_name.find('model') >= 0:
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

        data_distr = ClassificationDistribution(config.n_particles)
        dm = DistributionMover(task='net_class',
                               n_particles=config.n_particles,
                               n_hidden_dims=config.n_hidden_dims,
                               use_latent=config.use_latent,
                               net=net,
                               data_distribution=data_distr,
                               dummy=True
                               )
        lr_str = LRStrategy(step_size=0.03, factor=0.97, n_epochs=1, patience=80)

        dm.load_state_dict(torch.load(checkpoint_file_name))

        mv_l, mv_o = (torch.mean(torch.var(dm.particles, dim=1)),
                      torch.mean(torch.var(dm.lt.transform(dm.particles, n_particles_second=True), dim=1))
                      )
        mv_l, mv_o = float(mv_l.data[0].cpu()), float(mv_o.data[0].cpu())

        obj = dm
    elif exp_name.find('ensemble') >= 0:
        state_dict = torch.load(checkpoint_file_name)
        nets = []
        for _ in range(len(state_dict)):
            nets.append(
                nn.Sequential(
                    nn.Linear(28 * 28, 300),
                    nn.Tanh(),
                    nn.Linear(300, 100),
                    nn.Tanh(),
                    nn.Linear(100, 10)
                ).to(device=device).double()
            )
            nets[-1].load_state_dict(state_dict[_])

        params = torch.cat([torch.nn.utils.parameters_to_vector(net.parameters()).unsqueeze(0) for net in nets])

        mv_l, mv_o = [torch.mean(torch.var(params, dim=0))] * 2
        mv_l, mv_o = float(mv_l.data[0].cpu()), float(mv_o.data[0].cpu())

        obj = nets
    else:
        net = nn.Sequential(
            nn.Linear(28 * 28, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, 10)
        ).to(device=device).double()

        net.load_state_dict(torch.load(checkpoint_file_name))

        mv_l, mv_o = None, None

        obj = net

    er_1, ce_o, ce_1 = perform_adv_attack(dataloader_m_test, obj, modify_sample, eps=0.01, n_iters=20, use_full=False)
    er_a, _, ce_a = perform_adv_attack(dataloader_m_test, obj, modify_sample, eps=0.01, n_iters=20, use_full=True)
    er_au, _, ce_au = perform_aug_attack(dataloader_m_test, dataloader_m_test_aug, obj)
    er_1, er_a, er_au = round(er_1, 3), round(er_a, 3), round(er_au, 3)
    ce_o, ce_1, ce_a, ce_au = round(ce_o, 4), round(ce_1, 4), round(ce_a, 4), round(ce_au, 4)

    if mv_l is None or math.isinf(mv_l) or math.isnan(mv_l):
        mv_l = None
    else:
        mv_l = round(mv_l, 4)
    if mv_o is None or math.isinf(mv_o) or math.isnan(mv_o):
        mv_o = None
    else:
        mv_o = round(mv_o, 4)

    del obj

    data.append(
        Result(
            config.experiment_name,
            config.dataset,
            config.batch_size,
            config.net_arc,
            config.use_var_prior,
            config.alpha,
            config.n_particles,
            config.use_latent,
            config.n_hidden_dims,
            config.n_epochs,
            config.move_theta_0,
            config.init_theta_0,
            er_1, er_a, er_au,
            ce_o, ce_1, ce_a, ce_au,
            mv_l, mv_o,
            config.comment
        )
    )
    print(data[-1])

# Iterate over the data and write it out row by row.
for idx, result in enumerate(data):
    if experiments[idx][1] is not None:
        colorize = workbook.add_format({'bg_color': experiments[idx][1]})
    else:
        colorize = workbook.add_format({})

    if experiments[idx][1] is not None:
        right_align = workbook.add_format({'align': 'right', 'bg_color': experiments[idx][1]})
    else:
        right_align = workbook.add_format({'align': 'right'})

    for jdx, name in enumerate(column_names):
        if type(getattr(result, to_field(name))) == bool:
            bool_str = str(getattr(result, to_field(name))).upper()
            worksheets[0].write(idx + 1, jdx, bool_str, right_align)
        else:
            worksheets[0].write(idx + 1, jdx, getattr(result, to_field(name)), colorize)
    for jdx, name in enumerate(squeezed_column_names):
        if type(getattr(result, to_field(name))) == bool:
            bool_str = str(getattr(result, to_field(name))).upper()
            worksheets[1].write(idx + 1, jdx, bool_str, right_align)
        else:
            worksheets[1].write(idx + 1, jdx, getattr(result, to_field(name)), colorize)

# Write head of the table
for idx, name in enumerate(column_names):
    worksheets[0].set_column(idx, idx, len(name))
    worksheets[0].write(0, idx, name)
for idx, name in enumerate(squeezed_column_names):
    worksheets[1].set_column(idx, idx, len(name))
    worksheets[1].write(0, idx, name)

workbook.close()
