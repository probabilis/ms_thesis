import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pattern_formation import fourier_multiplier, energy_value, dtype_real, device

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

folder_path = r"out/"

L = 1.0
N = 128
epsilon = 1/20
gamma = 1/200
c0 = 9/32

alpha = 1e-4       # step size for plain GD
tau   = 5e-3       # step size for prox-based methods
num_iter = 20_000
th = 1.0

# ---------------------------------------------------------------
# Fourier multipliers
# ---------------------------------------------------------------
k = torch.cat([
    torch.arange(0, N // 2, dtype=dtype_real, device=device),
    torch.arange(-N // 2, 0, dtype=dtype_real, device=device)
])
xi, eta = torch.meshgrid(k, k, indexing='ij')
modk2 = (xi**2 + eta**2).to(dtype_real)
modk = torch.sqrt(modk2)

sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
M_k = sigma_k + gamma * epsilon * modk2

# ---------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------
def initialize_u0_random(N):
    amplitude = 0.1
    return amplitude * (2 * torch.rand(N, N, dtype=dtype_real, device=device) - 1)

u0 = initialize_u0_random(N)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def fft2_real(x): return torch.fft.fft2(x)
def ifft2_real(x): return torch.fft.ifft2(x).real

def grad_g(u):
    Fu = fft2_real(u)
    grad_hat = M_k * Fu
    return ifft2_real(grad_hat)

from gradient_descent import double_well_prime

def prox_h(v, tau, gamma=gamma, eps=epsilon, c0=c0, maxiter=20, tol=1e-8):
    lam = tau * (gamma / eps) * c0
    x = v.clone()
    for _ in range(maxiter):
        grad = x - v - 4.0 * lam * x * (1.0 - x * x)
        hess = 1.0 + lam * (4.0 * (x * x - 1) + 8.0 * x * x)
        hess_safe = torch.where(torch.abs(hess) < 1e-12, torch.sign(hess) * 1e-12, hess)
        step = grad / hess_safe
        x_new = x - step
        if torch.max(torch.abs(x_new - x)) < tol:
            return x_new
        x = x_new
    return x

# ---------------------------------------------------------------
# Methods
# ---------------------------------------------------------------

def run_gradient_descent(u0, alpha, num_iter):
    u = u0.clone()
    energies = []
    for _ in tqdm(range(num_iter), desc="GD"):
        # Fourier transform with consistent scaling
        Fu = fft2_real(u) / (N**2)

        # linear + nonlocal part (consistent with energy_value)
        grad_lin = ifft2_real((sigma_k + gamma * epsilon * modk2) * Fu) * (N**2)

        # nonlinear double well term (with gamma factor)
        grad_double = (gamma / epsilon) * double_well_prime(u, c0)

        # total gradient
        grad_E = grad_lin + grad_double

        # GD update
        u -= alpha * grad_E

        energies.append(energy_value(gamma, epsilon, N, u, th, modk, modk2, c0))
    return energies


def run_proximal_gradient(u0, tau, num_iter):
    u = u0.clone()
    energies = []
    for _ in tqdm(range(num_iter), desc="ProxGD"):
        ggrad = grad_g(u)
        v = u - tau * ggrad
        u = prox_h(v, tau)
        energies.append(energy_value(gamma, epsilon, N, u, th, modk, modk2, c0))
    return energies

def run_nesterov(u0, tau, num_iter):
    u_prev = u0.clone()
    u_curr = u0.clone()
    t_prev = 1.0
    energies = []
    for _ in tqdm(range(num_iter), desc="Nesterov"):
        # momentum extrapolation
        t_curr = 0.5 * (1.0 + (1.0 + 4.0 * t_prev**2)**0.5)
        beta = (t_prev - 1.0) / t_curr
        y = u_curr + beta * (u_curr - u_prev)
        # forward-backward
        ggrad = grad_g(y)
        v = y - tau * ggrad
        u_next = prox_h(v, tau)
        # update
        u_prev, u_curr, t_prev = u_curr, u_next, t_curr
        energies.append(energy_value(gamma, epsilon, N, u_curr, th, modk, modk2, c0))
    return energies

# ---------------------------------------------------------------
# Run all methods
# ---------------------------------------------------------------
energies_gd = run_gradient_descent(u0, alpha, num_iter)
energies_prox = run_proximal_gradient(u0, tau, num_iter)
energies_nest = run_nesterov(u0, tau, num_iter)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.semilogy(energies_gd, label="Gradient Descent")
plt.semilogy(energies_prox, label="Proximal Gradient Descent")
plt.semilogy(energies_nest, label="Nesterov + Prox")
plt.xlabel("Iteration")
plt.ylabel("Energy (log scale)")
plt.title("Convergence comparison")
plt.legend()
plt.grid(True)
plt.savefig(folder_path + f"energy_convergence_comparison_N={N}_nmax={num_iter}_gamma={gamma}_eps={epsilon}.png")
plt.show()
