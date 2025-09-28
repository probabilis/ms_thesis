import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pattern_formation import *
from env_utils import PATHS, plotting_style

plotting_style()
# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

folder_path = PATHS.COMPARISON

gridsize = 1.0
N = 128
epsilon = 1/20
gamma = 1/200
c0 = 9/32

alpha = 1e-4       # step size for plain GD
tau   = 5e-3       # step size for prox-based methods
num_iter = 3_000
th = 1.0

# ---------------------------------------------------------------
# Fourier multipliers
# ---------------------------------------------------------------
x, k, modk, modk2 = define_spaces(N, gridsize)

sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
M_k = sigma_k + gamma * epsilon * modk2

# ---------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------
u0 = initialize_u0_random(N, REAL = True)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

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
        ggrad = grad_g(u, M_k)
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
        ggrad = grad_g(y, M_k)
        v = y - tau * ggrad
        u_next = prox_h(v, tau)
        # update
        u_prev, u_curr, t_prev = u_curr, u_next, t_curr
        energies.append(energy_value(gamma, epsilon, N, u_curr, th, modk, modk2, c0))
    return energies

def run_crank_nicolson():
    LIVE_PLOT = False
    DATA_LOG = False
    dt = 1/10
    max_it_fixpoint = 10
    max_it = num_iter
    tol = 1e-6
    stop_limit = 1e-9
    from crank_nicolson import adapted_crank_nicolson

    energies = adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0)
    return energies
# ---------------------------------------------------------------
# Run all methods
# ---------------------------------------------------------------
energies_cn = run_crank_nicolson()

energies_gd = run_gradient_descent(u0, alpha, num_iter)
energies_prox = run_proximal_gradient(u0, tau, num_iter)
energies_nest = run_nesterov(u0, tau, num_iter)


def relative(x):
    if len(x) > 1:
        return [abs(i-x[0]) for i in x]
    else:
        return []
    
energies_cn = relative(energies_cn)
energies_gd = relative(energies_gd)
energies_prox = relative(energies_prox)
energies_nest = relative(energies_nest)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.semilogy(energies_cn, label="Crank Nicolson")
plt.semilogy(energies_gd, label="Gradient Descent")
plt.semilogy(energies_prox, label="Proximal Gradient Descent")
plt.semilogy(energies_nest, label="Nesterov + Prox")
plt.xlabel("Iteration")
plt.ylabel("Energy (log scale)")
plt.title("Convergence comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_path / f"energy_convergence_comparison_N={N}_nmax={num_iter}_gamma={gamma}_eps={epsilon}.png")
plt.show()
