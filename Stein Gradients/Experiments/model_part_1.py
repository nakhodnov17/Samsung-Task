use_cuda = False
device = None
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_cuda = True

from utils import *

# import MNIST dataset with class selection
sys.path.insert(0, '/home/m.nakhodnov/Samsung-Tasks/Datasets/MyMNIST')
from MyMNIST import MNIST_Class_Selection

# Ignore cuda
# use_cuda = False
# device = None

# Set DoubleTensor as a base type for computations
t_type = torch.float64

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                    ])
dataset_m_train = MNIST_Class_Selection('.', train=True, download=True, transform=transform)
dataset_m_test = MNIST_Class_Selection('.', train=False, transform=transform)


dataloader_m_train = DataLoader(dataset_m_train, batch_size=100, shuffle=True)
dataloader_m_test = DataLoader(dataset_m_test, batch_size=100, shuffle=False)

### MAP estimate for MNIST classification task
net_1 = nn.Sequential(SteinLinear(28 * 28, 18, 1, use_var_prior=False),
                      nn.Tanh(),
                      SteinLinear(18, 14, 1, use_var_prior=False),
                      nn.Tanh(),
                      SteinLinear(14, 10, 1, use_var_prior=False)
                     ).to(device=device)
data_distr_1 = ClassificationDistribution(1)
dm_1 = DistributionMover(task='net_class', n_particles=1, use_latent=False, net=net_1, data_distribution=data_distr_1)
lr_str_1 = LRStrategy(step_size=0.03, factor=0.97, n_epochs=1, patience=80)

own_name_1 = sys.argv[0][2:sys.argv[0].find('.py')]
version_1 = 0
checkpoint_file_name_1 = './Checkpoints/' + 'e{0}_' + own_name_1 + '.pth'
plots_file_name_1 = './Plots/' + own_name_1 + '.png'
log_file_name_1 = './Logs/' + own_name_1 + '.txt'
if os.path.exists(checkpoint_file_name_1.format(version_1)):
    dm_1.load_state_dict(torch.load(checkpoint_file_name_1.format(version_1)))
    lr_str_1.step_size = dm_1.step_size
    lr_str_1.iter = dm_1.epoch + 1

train(dm=dm_1,
      dataloader_train=dataloader_m_train, dataloader_test=dataloader_m_test,
      lr_str=lr_str_1, start_epoch=lr_str_1.iter, end_epoch=lr_str_1.iter + 16, n_epochs_save=20,
      checkpoint_file_name=checkpoint_file_name_1, plots_file_name=plots_file_name_1, log_file_name=log_file_name_1,
      n_warmup_epochs=1
     )

