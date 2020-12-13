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
    vx = a - torch.mean(a, 0)
    vy = b - torch.mean(b, 0)
    return torch.sum(vx * vy, 0) / (torch.sqrt(torch.sum(vx ** 2, 0), 0) * torch.sqrt(torch.sum(vy ** 2, 0)), 0)


def calc_TSS(A):
    """
    Calculate total dispersion according to section 1.1.1 from
    https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    :param A: some (batch x N x p) matrix where N = observations and p = features
    :return: The total dispersion of the matrix
    """
    A_mean = A.mean(1)
    c_enc = A - A_mean[:, None]
    T = torch.matmul(torch.transpose(c_enc, 1, 2), c_enc)
    tss = T.diagonal(0, 1, 2).sum(1)
    return tss


def within_group_scatter(A):
    """
    Calculate the within cluster dispersion
    Source: section 1.1.2. from https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    :param A: some (batch x N x p) matrix where N = observations and p = features
    :return:
    """
    WGSS = calc_TSS(A).sum()
    return WGSS


def between_group_scatter(A):
    """
    Calculate the between cluster dispersion
    Source: section 1.1.3. from https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    :param A: some (batch x N x p) matrix where N = observations and p = features
    :return:
    """
    total_mean = A.mean((0, 1))
    cluster_means = A.mean(1)
    B = cluster_means - total_mean

    BG = torch.matmul(B.t(), B)
    BGSS = BG.diag().sum()
    return BGSS


def normalized_scatter_ratio(A):
    """
    Normalized "between group scatter" - "within group scatter".
    Ranges from -1 to 1, where
        - negative values mean that the within-group scatter is greater than the between-groups scatter
        - positive values mean that the between-group scatter is greater than the within-groups scatter
        - 0 means that the scatter between groups is equal to the scatter within groups
    :param A: some (batch x N x p) matrix where N = observations and p = features
    :return:
    """
    BGSS = between_group_scatter(A)
    WGSS = within_group_scatter(A)
    normalized_diff = (BGSS - WGSS)/(BGSS + WGSS)
    return normalized_diff
