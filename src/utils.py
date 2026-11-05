import torch
import matplotlib.pyplot as plt
from scipy.special import factorial, hermite

from .config import SIGMA, X_SIZE, DX, x_coord, device


_GAUS_NORM = (2 * torch.pi * SIGMA**2) ** -0.25


def gaus(loc=0):
    return (_GAUS_NORM * torch.exp(-((x_coord - loc) ** 2) / (4 * SIGMA**2))).to(
        torch.complex64
    )


def hg(q=0):
    gaus_kernel = gaus(0)
    if q == 0:
        return gaus_kernel

    _x = x_coord / (2**0.5 * SIGMA)
    term1 = (2**q * factorial(q)) ** -0.5
    if q == 1:
        term2 = 2 * _x
    else:
        H = hermite(q)
        term2 = torch.tensor(H(as_arr(_x)), device=device)
    return (term1 * term2 * gaus_kernel).to(torch.complex64)


def delta(loc=0):
    field = torch.zeros(X_SIZE)
    idx = X_SIZE // 2 + round(loc / DX)
    field[idx] = 1
    return field.to(torch.complex64)


def proj(field, *modes):
    return torch.stack(
        [intens(torch.sum(fmt(mode).conj() * fmt(field)) * DX) for mode in modes]
    )


def as_arr(tensor):
    return tensor.detach().cpu().numpy().squeeze()


def intens(field):
    return torch.abs(field) ** 2


def power(field):
    return intens(field).sum(-1) * DX


def fmt(tensor):
    l = len(tensor.shape)
    assert l <= 3
    for _ in range(3 - l):
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=device, dtype=torch.complex64)


def merge(*fields, dim):
    fields = [fmt(field) for field in fields]
    return torch.cat(fields, dim=dim)


def vis(*inputs, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots(1, 1)
    for input_ in inputs:
        ax.plot(as_arr(x_coord), (as_arr(input_)))
    ax.set_xlabel(r"$x~(\rm \mu m)$" if xlabel is None else xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax
