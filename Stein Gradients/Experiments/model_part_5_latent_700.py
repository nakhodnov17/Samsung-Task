use_cuda = False
device = None
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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


net_2 = nn.Sequential(SteinLinear(28 * 28, 18, 5, use_var_prior=False),
                      nn.Tanh(),
                      SteinLinear(18, 14, 5, use_var_prior=False),
                      nn.Tanh(),
                      SteinLinear(14, 10, 5, use_var_prior=False)
                     ).to(device=device)
data_distr_2 = ClassificationDistribution(5)
dm_2 = DistributionMover(task='net_class',
                         n_particles=5,
                         n_hidden_dims=700,
                         use_latent=True,
                         net=net_2,
                         data_distribution=data_distr_2
                        )
lr_str_2 = LRStrategy(step_size=0.03, factor=0.97, n_epochs=1, patience=80)

own_name_2 = sys.argv[0][2:sys.argv[0].find('.py')]
version_2 = 0
checkpoint_file_name_2 = './Checkpoints/' + 'e{0}_' + own_name_2 + '.pth'
plots_file_name_2 = './Plots/' + own_name_2 + '.png'
log_file_name_2 = './Logs/' + own_name_2 + '.txt'
if os.path.exists(checkpoint_file_name_2.format(version_2)):
    dm_2.load_state_dict(torch.load(checkpoint_file_name_2.format(version_2)))
    lr_str_2.step_size = dm_2.step_size
    lr_str_2.iter = dm_2.epoch + 1

train(dm=dm_2,
      dataloader_train=dataloader_m_train, dataloader_test=dataloader_m_test,
      lr_str=lr_str_2, start_epoch=lr_str_2.iter, end_epoch=lr_str_2.iter + 100, n_epochs_save=20,
      checkpoint_file_name=checkpoint_file_name_2, plots_file_name=plots_file_name_2, log_file_name=log_file_name_2,
      n_warmup_epochs=16
     )