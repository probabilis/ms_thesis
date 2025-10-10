import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pattern_formation import *
from env_utils import PATHS, plotting_style

# ---------------------------------------------------------------

plotting_style()
folder_path = PATHS.COMPARISON


# ---------------------------------------------------------------
gridsize = 1.0
N = 256
epsilon = 1/20
gamma = 1/200
c0 = 9/32
th = 1.0

alpha = 1e-4       # step size for plain GD
tau   = 5e-3       # step size for prox-based methods


num_iter = 10_000          # number of iterations for all methods
ENERGY_STOP_TOL = 1e-8  # energy stopping tolerance if STOP_BY_TOL is true for methods 


# ---------------------------------------------------------------
x, k, modk, modk2 = define_spaces(gridsize, N)

sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
M_k = sigma_k + gamma * epsilon * modk2

# ---------------------------------------------------------------
u0 = initialize_u0_random(N, REAL = True)

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


def run_gradient_descent(u0, alpha, num_iter, STOP_BY_TOL = False):

    u = u0.clone()
    energies = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]

    for _ in tqdm(range(num_iter), desc="GD"):
        # Fourier transform normalized
        Fu = fft2_real(u) / (N**2)

        # local term 1 (laplcian term)
        laplacian_fourier_space = gamma * epsilon * modk2

        # linear + nonlocal part (FM part + laplacian)
        grad_lin = ifft2_real((sigma_k + laplacian_fourier_space) * Fu) * (N**2)

        # nonlinear / local part (double well term)
        grad_double = (gamma / epsilon) * double_well_prime(u, c0)

        # total gradient
        grad_E = grad_lin + grad_double

        # GD update
        u -= alpha * grad_E

        curr_energy = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
        energy_diff = energies[-1] - curr_energy
        energies.append(curr_energy)

        if STOP_BY_TOL and energy_diff < ENERGY_STOP_TOL:
            break

    return energies

def run_proximal_gradient(u0, tau, num_iter, STOP_BY_TOL = False):
    u = u0.clone()
    energies = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]
    for _ in tqdm(range(num_iter), desc="ProxGD"):
        ggrad = grad_g(u, M_k)
        v = u - tau * ggrad
        u = prox_h(v, tau)

        curr_energy = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
        energy_diff = energies[-1] - curr_energy
        energies.append(curr_energy)
        if STOP_BY_TOL and energy_diff < ENERGY_STOP_TOL:
            break

    return energies

def run_nesterov(u0, tau, num_iter, STOP_BY_TOL = False):
    u_prev = u0.clone()
    u_curr = u0.clone()
    t_prev = 1.0
    energies = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]
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

        curr_energy = energy_value(gamma, epsilon, N, u_curr, th, modk, modk2, c0)
        energy_diff = energies[-1] - curr_energy
        energies.append(curr_energy)
        if STOP_BY_TOL and energy_diff < ENERGY_STOP_TOL:
            break


    return energies

def run_crank_nicolson(_ls, STOP_BY_TOL = False):
    LIVE_PLOT = False
    DATA_LOG = False
    dt = 1/10
    max_it_fixpoint = 1
    max_it = num_iter
    tol = 1e-6
    stop_limit = ENERGY_STOP_TOL
    from crank_nicolson import adapted_crank_nicolson
    
    energies_cn = []
    for max_it_fixpoint in _ls:
        energies = adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0, STOP_BY_TOL)
        energies_cn.append(energies)
    return energies_cn



max_it_fixpoint_ls = [1, 2, 5]
colors_cn = ['cornflowerblue', 'royalblue','blue']
colors_gd = ['lightcoral', 'mediumseagreen', 'red']

A = False
B = True

if A:

    # ---------------------------------------------------------------

    energies_cn = run_crank_nicolson(max_it_fixpoint_ls)
    energies_gd = run_gradient_descent(u0, alpha, num_iter)
    energies_prox = run_proximal_gradient(u0, tau, num_iter)
    energies_nest = run_nesterov(u0, tau, num_iter)

    # ---------------------------------------------------------------

    fig, ax = plt.subplots(1,1, figsize = (12,10))

    for ii, _cn in enumerate(energies_cn):
        ax.plot(_cn, label= fr"Crank Nicolson $N_{{fixpoint}}$ = {max_it_fixpoint_ls[ii]}", linewidth = 2, color = colors_cn[ii])

    ax.plot(energies_gd, label="Gradient Descent", linewidth = 3, color =  colors_gd[0])
    ax.plot(energies_prox, label="Proximal Gradient Descent", linewidth = 3, color = colors_gd[1])
    ax.plot(energies_nest, label="Nesterov Proximal GD", linewidth = 3, color = colors_gd[2])

    #ax.set_yscale('log')
    ax.set_xlabel("Iteration $i$")
    ax.set_ylabel("Energy $E_i$")
    ax.set_title("Energy convergence comparison")


    ax.legend(loc = "upper right")
    ax.grid(True)
    fig.tight_layout()

    plt.savefig(folder_path / f"energy_convergence_comparison_N={N}_nmax={num_iter}_gamma={gamma}_eps={epsilon}.png", dpi = 300)
    plt.show()

    # ---------------------------------------------------------------
    # relative (E_i - E_0) plot

    fig, ax = plt.subplots(1,1, figsize = (12,10))

    def relative(x, REL = True):
        if len(x) > 1 and REL:
            return [abs(i-x[0]) for i in x]
        else:
            return x

    energies_gd = relative(energies_gd)
    energies_prox = relative(energies_prox)
    energies_nest = relative(energies_nest)

    for ii, _cn in enumerate(energies_cn):
        _diff = relative(_cn)
        ax.plot(_diff, label= fr"Crank Nicolson $N_{{fixpoint}}$ = {max_it_fixpoint_ls[ii]}", linewidth = 2, color = colors_cn[ii])

    ax.plot(energies_gd, label="Gradient Descent", linewidth = 3, color =  colors_gd[0])
    ax.plot(energies_prox, label="Proximal Gradient Descent", linewidth = 3, color = colors_gd[1])
    ax.plot(energies_nest, label="Nesterov Proximal GD", linewidth = 3, color = colors_gd[2])

    ax.set_xlabel("Iteration $i$")
    ax.set_ylabel("Energy difference $|E_i - E_0|$")
    ax.set_title("Energy convergence comparison / $E_0$ is initial energy")


    ax.legend(loc = "lower right")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(folder_path / f"energy_convergence_comparison_relative_N={N}_nmax={num_iter}_gamma={gamma}_eps={epsilon}.png", dpi = 300)
    plt.show()


# ---------------------------------------------------------------

if B:
    STOP_BY_TOL = True

    energies_cn = run_crank_nicolson(max_it_fixpoint_ls, STOP_BY_TOL)
    energies_gd = run_gradient_descent(u0, alpha, num_iter, STOP_BY_TOL)
    energies_prox = run_proximal_gradient(u0, tau, num_iter, STOP_BY_TOL)
    energies_nest = run_nesterov(u0, tau, num_iter, STOP_BY_TOL)


    fig, ax = plt.subplots(1,1, figsize = (12,10))

    for ii, _cn in enumerate(energies_cn):
        ax.plot(_cn, label= fr"Crank Nicolson $N_{{fixpoint}}$ = {max_it_fixpoint_ls[ii]}", linewidth = 2, color = colors_cn[ii])

    ax.plot(energies_gd, label="Gradient Descent", linewidth = 3, color =  colors_gd[0])
    ax.plot(energies_prox, label="Proximal Gradient Descent", linewidth = 3, color = colors_gd[1])
    ax.plot(energies_nest, label="Nesterov Proximal GD", linewidth = 3, color = colors_gd[2])

    #ax.set_yscale('log')

    ax.set_xlabel("Iteration $i$")
    ax.set_ylabel("Energy $E_i$")
    ax.set_title(f"Energy convergence comparison / stop at tolerance = {ENERGY_STOP_TOL}")

    ax.legend(loc = "upper right")
    ax.grid(True)
    fig.tight_layout()

    plt.savefig(folder_path / f"energy_convergence_comparison_stop-by-limit={ENERGY_STOP_TOL}_N={N}_nmax={num_iter}_gamma={gamma}_eps={epsilon}.png", dpi = 300)
    plt.show()

# ---------------------------------------------------------------
# 