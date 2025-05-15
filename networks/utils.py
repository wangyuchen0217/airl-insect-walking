import math
import torch
from torch import nn


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    
    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


# def reparameterize(means, log_stds):
#     noises = torch.randn_like(means)
#     us = means + noises * log_stds.exp()
#     actions = torch.tanh(us)
#     return actions, calculate_log_pi(log_stds, noises, actions)

def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    pre_tanh = means + noises * stds
    actions = torch.tanh(pre_tanh)

    # 修正后的 log_prob，包含 tanh 的导数项
    log_probs = -0.5 * ((noises ** 2) + 2 * log_stds + torch.log(torch.tensor(2 * torch.pi)))
    log_probs = log_probs.sum(dim=-1)

    # tanh 的 Jacobian 修正
    log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)

    return actions, log_probs


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


# def evaluate_lop_pi(means, log_stds, actions):
#     noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
#     return calculate_log_pi(log_stds, noises, actions)

def evaluate_lop_pi(means, log_stds, actions):
    # 反向变换动作
    pre_tanh = atanh(torch.clamp(actions, -0.999, 0.999))
    stds = log_stds.exp()

    noises = (pre_tanh - means) / stds
    log_probs = -0.5 * ((noises ** 2) + 2 * log_stds + torch.log(torch.tensor(2 * torch.pi)))
    log_probs = log_probs.sum(dim=-1)

    # Jacobian 修正项
    log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)

    return log_probs

