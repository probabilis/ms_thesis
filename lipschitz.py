import torch
import math

from pattern_formation import dtype_real, fourier_multiplier, device

th = 1.0
N = 40
Lx = 1.0
h = Lx / N

gamma = 0.005
eps = 0.05

from pattern_formation import double_well_potential



x = torch.arange(-2, +2, 0.01)
y = double_well_potential(x, 9/32)
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()



kx = torch.fft.fftfreq(N, d=h).to(device) * 2 *  torch.pi
ky = torch.fft.fftfreq(N, d=h).to(device) * 2 * torch.pi
KX, KY = torch.meshgrid(kx, ky, indexing='ij')
K = torch.sqrt(KX**2 + KY**2)

# Discrete Laplacian eigenvalues (periodic 2D)
# indices 0..N-1
idx = torch.arange(N)
kx = math.pi * idx / N  # kx*h/2 = π i/N => we store θ = kx*h/2
ky = math.pi * idx / N
KX, KY = torch.meshgrid(kx, ky, indexing="ij")

lambda_lap = -4.0 / h**2 * (torch.sin(KX)**2 + torch.sin(KY)**2)
rho_lap = torch.max(torch.abs(lambda_lap))  # ≈ 8/h^2

sigma_hat = fourier_multiplier(th * K).to(dtype_real).to(device)
# Nonlocal operator norm
rho_sigma = torch.max(torch.abs(sigma_hat))  # sigma_hat(kx,ky)

# Nonlinear double-well bound (on u in [-1,1])
Wpp_max = 2.0

L_A = gamma * eps * rho_lap + rho_sigma
L_g = gamma * Wpp_max / eps
L_total = L_A + L_g

eta_safe = 1.0 / L_total
print("Lipschitz upper bound L =", L_total.item())
print("Safe step size eta ~", eta_safe.item())
