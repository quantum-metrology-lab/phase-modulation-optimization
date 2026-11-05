import torch
from torch import nn
from scipy.signal.windows import dpss

from .utils import fmt
from .config import DX, X_SIZE, device


class PSLM(nn.Module):
    def __init__(self, bandwidth, init_std, learnable=True):
        super().__init__()
        nw = X_SIZE * bandwidth * DX
        self.num_coeffs = round(2 * nw)
        basis = dpss(X_SIZE, NW=nw, Kmax=self.num_coeffs)
        self.register_buffer(
            "basis",
            torch.from_numpy(basis.copy()).to(device=device, dtype=torch.float32),
        )

        if learnable:
            self.coeffs = nn.Parameter(
                init_std * torch.randn(len(self.basis), device=device)
            )
        else:
            self.register_buffer(
                "coeffs",
                init_std * torch.randn(len(self.basis), device=device),
            )

    @property
    def phase(self):
        return torch.pi * torch.tanh(self.coeffs @ self.basis)

    def forward(self, field):
        field = fmt(field)
        return field * torch.exp(1j * self.phase)


# class PSLM(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.coeffs = nn.Parameter(0.01 * torch.randn(X_SIZE, device=device))

#     @property
#     def phase(self):
#         return torch.pi * torch.tanh(self.coeffs)

#     def forward(self, field):
#         return fmt(field) * torch.exp(1j * self.phase)
