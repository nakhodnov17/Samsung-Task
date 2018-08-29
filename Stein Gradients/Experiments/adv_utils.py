import os
import torch
import argparse
import importlib

from utils import *

use_cuda = False
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_cuda = True
    
# Set DoubleTensor as a base type for computations
t_type = torch.float64


def modify_sample(obj, samples, real_labels, eps=1e-2, n_iters=1, use_full=False):
    for idx in range(n_iters):
        samples = samples.detach().clone().requires_grad_(True)

        if isinstance(obj, DistributionMover):
            predictions = obj.predict_net(samples, inference=False)
            if use_full:
                predictions = predictions[:]
            else:
                predictions = predictions[0:1]
        elif isinstance(obj, list):
            if use_full:
                predictions = torch.cat([net(samples).unsqueeze(0) for net in obj])
            else:
                predictions = obj[0](samples).unsqueeze(0)
        else:
            predictions = obj(samples).unsqueeze(0)
        predictions = torch.log(torch.mean(torch.nn.Softmax(dim=2)(predictions), dim=0))

        train_loss = -torch.sum(torch.gather(predictions, 1, real_labels.view(-1, 1)))

        train_loss.backward()
        perturbation = torch.sign(samples.grad)
        samples = torch.clamp(samples + eps * perturbation, -0.42, 2.8)

    return samples.detach()


def _calc_attack_values(obj, samples, modified_samples, real_labels):
    if isinstance(obj, DistributionMover):
        # evaluate crossentropy on real samples
        pred = obj.predict_net(samples, inference=False)
        pred_slice = pred[:]
        log_p = torch.log(torch.mean(torch.nn.Softmax(dim=2)(pred_slice), dim=0))
        cross_entr = -torch.sum(torch.gather(log_p, 1, real_labels.view(-1, 1))).data[0].cpu().numpy()

        # evaluate crossentropy on modified samples
        pred = obj.predict_net(modified_samples, inference=False)
        pred_slice = pred[:]
        log_p = torch.log(torch.mean(torch.nn.Softmax(dim=2)(pred_slice), dim=0))
        cross_entr_modified = -torch.sum(torch.gather(log_p, 1, real_labels.view(-1, 1))).data[0].cpu().numpy()

        # evaluate error rate
        pred = torch.argmax(obj.predict_net(samples, inference=True), dim=1)
        pred_modified = torch.argmax(obj.predict_net(modified_samples, inference=True), dim=1)
        cnt = len(np.argwhere(pred - pred_modified != 0).view(-1))
    elif isinstance(obj, nn.Sequential):
        # evaluate crossentropy on real samples
        pred = obj(samples)
        log_p = torch.log(torch.nn.Softmax()(pred))
        cross_entr = -torch.sum(torch.gather(log_p, 1, real_labels.view(-1, 1))).data[0].cpu().numpy()

        # evaluate crossentropy on modified samples
        pred = obj(modified_samples)
        log_p = torch.log(torch.nn.Softmax()(pred))
        cross_entr_modified = -torch.sum(torch.gather(log_p, 1, real_labels.view(-1, 1))).data[0].cpu().numpy()

        # evaluate error rate
        pred = torch.argmax(obj(samples), dim=1)
        pred_modified = torch.argmax(obj(modified_samples), dim=1)
        cnt = len(np.argwhere(pred - pred_modified != 0).view(-1))
    elif isinstance(obj, list):
        # evaluate crossentropy on real samples
        pred = torch.cat([net(samples).unsqueeze(0) for net in obj])
        pred_slice = pred[:]
        log_p = torch.log(torch.mean(torch.nn.Softmax(dim=2)(pred_slice), dim=0))
        cross_entr = -torch.sum(torch.gather(log_p, 1, real_labels.view(-1, 1))).data[0].cpu().numpy()

        # evaluate crossentropy on modified samples
        pred = torch.cat([net(modified_samples).unsqueeze(0) for net in obj])
        pred_slice = pred[:]
        log_p = torch.log(torch.mean(torch.nn.Softmax(dim=2)(pred_slice), dim=0))
        cross_entr_modified = -torch.sum(torch.gather(log_p, 1, real_labels.view(-1, 1))).data[0].cpu().numpy()

        # evaluate error rate
        pred = torch.cat([net(samples).unsqueeze(0) for net in obj])
        pred = torch.argmax(torch.mean(nn.Softmax(dim=2)(pred), dim=0), dim=1)

        pred_modified = torch.cat([net(modified_samples).unsqueeze(0) for net in obj])
        pred_modified = torch.argmax(torch.mean(nn.Softmax(dim=2)(pred_modified), dim=0), dim=1)

        cnt = len(np.argwhere(pred - pred_modified != 0).view(-1))
    else:
        raise RuntimeError

    return cross_entr, cross_entr_modified, cnt


def perform_adv_attack(dataloader, obj, modifier, **modifier_kwargs):
    cnt = 0.
    cross_entr = 0.
    cross_entr_modified = 0.
    for samples, real_labels in dataloader:
        samples = samples.double().to(device=device).view(samples.shape[0], -1)
        real_labels = real_labels.to(device=device)

        # modify samples
        modified_samples = modifier(obj, samples=samples, real_labels=real_labels, **modifier_kwargs)

        _cross_entr, _cross_entr_modified, _cnt = _calc_attack_values(obj, samples, modified_samples, real_labels)

        cnt += _cnt
        cross_entr += _cross_entr
        cross_entr_modified += _cross_entr_modified

    return (cnt / len(dataloader.dataset),
            cross_entr / len(dataloader.dataset),
            cross_entr_modified / len(dataloader.dataset)
            )


def perform_aug_attack(dataloader, dataloader_aug, obj):
    cnt = 0.
    cross_entr = 0.
    cross_entr_modified = 0.

    iterator = iter(dataloader)
    iterator_aug = iter(dataloader_aug)
    for _ in range(len(dataloader)):
        samples, real_labels = next(iterator)
        modified_samples, _ = next(iterator_aug)

        samples = samples.double().to(device=device).view(samples.shape[0], -1)
        modified_samples = modified_samples.double().to(device=device).view(modified_samples.shape[0], -1)
        real_labels = real_labels.to(device=device)

        _cross_entr, _cross_entr_modified, _cnt = _calc_attack_values(obj, samples, modified_samples, real_labels)

        cnt += _cnt
        cross_entr += _cross_entr
        cross_entr_modified += _cross_entr_modified

    return (cnt / len(dataloader.dataset),
            cross_entr / len(dataloader.dataset),
            cross_entr_modified / len(dataloader.dataset)
            )
