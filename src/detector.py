import torch
from torch import nn

from .utils import intens, fmt
from .config import PHOTONS, NOISE, DX, device


class Detector(nn.Module):
    def __init__(self, method="gaus"):
        super().__init__()
        assert method.lower() in ("gaus", "pois", "direct")
        self.method = method.lower()

    def forward(self, field):
        field = fmt(field)
        pmf = intens(field).sum(1) * DX
        self.total = pmf.sum(-1)
        # if not torch.allclose(total, torch.ones_like(total), atol=0.05):
        #     print(f"warning: pmf.sum(-1) = {total}")

        mu = PHOTONS * pmf + NOISE
        if self.method == "gaus":
            returns = mu + torch.sqrt(mu) * torch.randn(mu.shape, device=device)
        elif self.method == "pois":
            returns = torch.poisson(mu)
        else:
            returns = mu

        return returns
