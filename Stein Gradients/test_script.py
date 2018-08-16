from utils import *

### Bostor housing dataset for regression task

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# boston = load_boston()
#
# X = boston['data']
# y = boston['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# X_train, X_test, y_train, y_test = (torch.tensor(X_train, dtype=t_type, device=device),
#                                     torch.tensor(X_test, dtype=t_type, device=device),
#                                     torch.tensor(y_train, dtype=t_type, device=device),
#                                     torch.tensor(y_test, dtype=t_type, device=device))
#
# ### Linear Regression baseline
#
# from sklearn.linear_model import LinearRegression
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# print(torch.nn.MSELoss()(torch.tensor(model.predict(X_test), dtype=t_type, device=device), y_test),
#       torch.nn.MSELoss()(torch.tensor(model.predict(X_train), dtype=t_type, device=device), y_train))
#
# print(torch.nn.MSELoss()(torch.mean(y_train).expand(y_test.shape[0]), y_test),
#       torch.nn.MSELoss()(torch.mean(y_train).expand(y_train.shape[0]), y_train))
#
#
# net = nn.Sequential(SteinLinear(13, 1, 700, use_var_prior=False, alpha=1e-2, use_bias=True))
# data_distr = RegressionDistribution(700, use_var_prior=False, betta=1e-1)
# dm = DistributionMover(task='net_reg', n_particles=700, use_latent=False, net=net, data_distribution=data_distr)
#
#
# log_file = open('log.txt', 'a')
#
# try:
#     if os.path.getsize('particles.txt'):
#         part_file_read = open('particles.txt', 'rb')
#         dm.particles = torch.load(part_file_read).cuda(device=device)
# except FileNotFoundError:
#     pass
#
# try:
#     step_size = 0.00025
#     dm.historical_grad.zero_()
#     for _ in range(100000):
#         dm.update_latent_net(h_type=1, kernel_type='rbf', p=-1, X_batch=X_train, y_batch=y_train,
#                              train_size=X_train.shape[0], step_size=step_size)
#         train_loss = torch.nn.MSELoss()(dm.predict_net(X_train, inference=True).view(-1), y_train)
#         test_loss = torch.nn.MSELoss()(dm.predict_net(X_test, inference=True).view(-1), y_test)
#
#         if _ % 1 == 0:
#             clear_output()
#
#             sys.stdout.write(
#                 '\rEpoch {0}... Empirical Loss(Train): {1:.3f}\t Empirical Loss(Test): {2:.3f}\t Kernel factor: {3:.3f}'.format(
#                     _, train_loss, test_loss, dm.h))
#
#             log_file.write( '\rEpoch {0}... Empirical Loss(Train): {1:.3f}\t Empirical Loss(Test): {2:.3f}\t Kernel factor: {3:.3f}'.format(
#                     _, train_loss, test_loss, dm.h))
#
#         if _ % 10 == 0:
#             part_file_write = open('particles.txt', 'wb')
#             torch.save(dm.particles.cpu(), part_file_write)
#             part_file_write.close()
#
# except KeyboardInterrupt:
#     log_file.close()
#     part_file_write.close()
#
# log_file.close()

net = nn.Sequential(SteinLinear(28 * 28, 200, 10), nn.ReLU(), SteinLinear(200, 200, 10), nn.ReLU(), SteinLinear(200, 10, 10))
data_distr = ClassificationDistribution(10)
dm = DistributionMover(task='net_class',
                       n_particles=10,
                       n_hidden_dims=700,
                       use_latent=False,
                       net=net,
                       data_distribution=data_distr
                      )
import dill
import pickle
with open('tmp_file.pl', 'wb') as tmp_file:
    torch.save(dm, tmp_file)