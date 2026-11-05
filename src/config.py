import torch

X_SIZE = 512
DX = 4.6
WAVELENGTH = 770e-3
F = 150e3
SIGMA = 100

PHOTONS = 1000
NOISE = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
x_coord = ((torch.arange(X_SIZE) - X_SIZE // 2) * DX).to(device)
