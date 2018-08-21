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


net_4 = nn.Sequential(SteinLinear(28 * 28, 18, 20, use_var_prior=False),
                      nn.Tanh(),
                      SteinLinear(18, 14, 20, use_var_prior=False),
                      nn.Tanh(),
                      SteinLinear(14, 10, 20, use_var_prior=False)
                     ).to(device=device)
data_distr_4 = ClassificationDistribution(20)
dm_4 = DistributionMover(task='net_class',
                         n_particles=20,
                         n_hidden_dims=700,
                         use_latent=True,
                         net=net_4,
                         data_distribution=data_distr_4
                        )
lr_str_4 = LRStrategy(step_size=0.03, factor=0.97, n_epochs=1, patience=80)

own_name = sys.argv[0][:sys.argv[0].find('.py')]
checkpoint_file_name_4 = './Checkpoints/' + own_name + '.pth'
plots_file_name_4 = './Plots/' + own_name + '.png'
log_file_name_4 = './Logs/' + own_name + '.txt'
if os.path.exists(checkpoint_file_name_4):
    dm_4.load_state_dict(torch.load(checkpoint_file_name_4))
    lr_str_4.step_size = dm_4.step_size
    lr_str_4.n_epochs = dm_4.epoch

train(dm=dm_4,
      dataloader_train=dataloader_m_train, dataloader_test=dataloader_m_test,
      lr_str=lr_str_4, epochs=200,
      checkpoint_file_name=checkpoint_file_name_4, plots_file_name=plots_file_name_4, log_file_name=log_file_name_4
      )