import torch
from torch import nn

from .utils import fmt
from .config import DX, WAVELENGTH, F, x_coord, device


class Lens(nn.Module):
    def __init__(self):
        super().__init__()
        phase = -2j * torch.pi * x_coord[:, None] * x_coord[None, :] / (WAVELENGTH * F)
        scale = DX / torch.sqrt(torch.tensor(1j * WAVELENGTH * F, device=device))
        dft_mat = (scale * torch.exp(phase)).to(torch.complex64)
        self.register_buffer("U", dft_mat)

    def forward(self, field):
        field = fmt(field)
        return field @ self.U.T
