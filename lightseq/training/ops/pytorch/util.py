import torch


def copy_para(x):
    return torch.nn.Parameter(torch.empty_like(x).copy_(x))
