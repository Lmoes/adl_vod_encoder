"""
Contains custom validation metrics
"""

import torch


def calc_rsquared(a, b):
    """
    Calculates the correlation coefficient between tensors a and b.
    :param a: tensor, same shape as b
    :param b: tensor, same shape as a
    :return: correlation coefficient of tensors a and b
    """
    vx = a - torch.mean(a)
    vy = b - torch.mean(b)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))