"""
Params:
    use_cuda (bool)
    cuda_device_id (int): id of selected gpu
    dataset (str): name of dataset to train
        Default: 'MNIST'
    batch_size (int)
        Default: 100
    net_arc (str): description of network
        'fc-18-14' | 'fc-300-100'
        Default: 'fc-18-14'
    use_var_prior (bool): if set to True, use Gamma prior distribution of weight variance
        Default: False
    alpha (float, None): specify alpha for weight prior ~N(w|0, 1 / alpha)
        If alpha == None then use GLOROT prior
    n_particles (int): number of particles in stein gradient
    use_latent (bool): if True subspace Stein is used
    n_hidden_dims (int): dim of subspace
    experiment_name (str): name of experiment
    version (int): specify checkpoint to continue training process
    n_epochs (int): model will be trained exactly @n_epochs epochs
    n_epochs_save (int): specify period of making checkpoints
        Default: 20
    n_epochs_log (int): specify period of writing log
        Default: 1
    move_theta_0 (bool): if True and use_latent == True then theta_0 in LinearTransform will be update
    n_warmup_epochs (int): specify number of epoch since that loss and accuracy will be averaged
    n_previous (int): specify number of epoch that loss/accuracy will be averaged over last @n_previous epochs
    h_type (int, float): specify way to compute kernel factor
    kernel_type (std)
    p (double, None): power in rational kernel
    use_initializer (bool): if True then some fields may be initializer from checkpoint @initializer_name
    initializer_name (str): checkpoint name which is used for initialization
    init_theta_0 (bool): if True then use theta_0 from @initializer_name checkpoint
"""

use_cuda = True
cuda_device_id = 4
dataset = 'MNIST'
batch_size = 100
net_arc = 'fc-300-100'
use_var_prior = False
alpha = None
n_particles = 5
use_latent = True
n_hidden_dims = 500
experiment_name = 'model_23'
version = 0
n_epochs = 200
n_epochs_save = 100
n_epochs_log = 1
move_theta_0 = False
n_warmup_epochs = 10
n_previous = 6
h_type = 0
kernel_type = 'rbf'
p = None
use_initializer = True
initializer_name = ('./Checkpoints/' + 'e{0}-{1}_' + 'ml_est' + '.pth').format(0, 35)
init_theta_0 = True
comment = 'Use ML initialization'
