import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict

from pattern_formation import *
from params import labyrinth_data_params, pgd_sim_params, get_DataParameters
from env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic, log_data


# ---------------------------------------------------------------

def prox_h(v, tau, gamma, eps, c0, maxiter, tol):
    # --- proximal operator for h(x) = (gamma/epsilon) * c0 * (1 - x^2)^2 ---
    # via vectorized Newton method, returns prox evaluated elementwise
    # Ref.: https://stackoverflow.com/questions/30191851/vectorize-a-newton-method-in-python-numpy
    # minimize 0.5*(x-v)^2 + tau*(gamma/eps)*c0*(1-x^2)^2

    lam = tau * (gamma / eps) * c0
    x = v.clone()

    for i in range(maxiter):

        grad = x - v - 4.0 * lam * x * (1.0 - x * x)
        hess = 1.0 + lam * (4.0 * (x * x - 1) + 8.0 * x * x)
        hess_safe = torch.where(torch.abs(hess) < 1e-12, torch.sign(hess) * 1e-12, hess)
        step = grad / hess_safe # ratio for newtons method

        # damped update (clamp step to avoid runaway)
        # use backtracking-like damping factor to ensure phi decreases (simple safeguard)
        # Ref.: claude.ai + stackoverflow
        alpha = 1.0
        x_new = x - alpha * step

        max_jump = 0.5
        delta = x_new - x
        overshoot = torch.abs(delta) > max_jump
        if overshoot.any():
            # scale down the step where overshooting
            scale = max_jump / (torch.abs(delta) + 1e-16)
            x_new = x + delta * torch.where(overshoot, scale, torch.ones_like(scale))

        # check convergence (max abs difference)
        if torch.max(torch.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    return x

# ---------------------------------------------------------------

def gradient_descent_proximal(u, LIVE_PLOT, DATA_LOG, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton):


    x, k, modk, modk2 = define_spaces(gridsize, N)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
    M_k = sigma_k + gamma * epsilon * modk2 # M_k multiplier for the quadratic term

    # --- main proximal-gradient loop ---
    energies = []
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
    fig2, ax2 = plt.subplots(1,1, figsize=(5,5))
    plt.ion()

    try:
        for n in tqdm(range(num_iters)):

            # forward step (gradient of smooth part)
            ggrad = grad_g(u, M_k)     
            v = u - tau * ggrad

            # backward/prox step: solve pointwise prox
            u = prox_h(v, tau, gamma=gamma, eps=epsilon, c0=c0,maxiter=prox_newton_iters, tol=tol_newton)

            # u = torch.clamp(u, -10.0, 10.0) # keep u within reasonable bounds to avoid blowup
            try:
                E = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
            except Exception as e:
                E = None
            energies.append(E)

            if LIVE_PLOT and (n % 1000) == 0:
                plotting_schematic(folder_path, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)
                plt.pause(1)

    except KeyboardInterrupt:
        print("Exit.")  
    
    plt.ioff()

    if DATA_LOG:
        log_data(folder_path, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(folder_path, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)
        plt.pause(1)

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    folder_path = PATHS.PATH_PGD
    
    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log


    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u = initialize_u0_random(N, REAL = True)

    print_bars()
    print(labyrinth_data_params)
    print(pgd_sim_params)
    print_bars()

    gradient_descent_proximal(u, LIVE_PLOT, DATA_LOG,**asdict(labyrinth_data_params),**asdict(pgd_sim_params))


