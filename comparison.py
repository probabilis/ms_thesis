import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pattern_formation import *
from env_utils import PATHS, plotting_style

# algorithms
from crank_nicolson import adapted_crank_nicolson
from gradient_descent import gradient_descent
from gd_proximal import gradient_descent_proximal
from gd_nesterov import gradient_descent_nesterov


def relative(x, REL = True):
    if len(x) > 1 and REL:
        return [abs(i-x[0]) for i in x]
    else:
        return x

def convergence_comparison():

    STOP_BY_TOL = False # which means full iterations are done, independent of energy convergence value
    
    # multiple CN instances
    energies_cn = []
    for max_it_fixpoint in max_it_fixpoint_ls:
        max_it = num_iters / max_it_fixpoint
        energies = adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0, STOP_BY_TOL)
        energies_cn.append(energies)

    energies_gd = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, c0, alpha, num_iters, STOP_BY_TOL)

    energies_prox = gradient_descent_proximal(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL)
    
    energies_nest = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL)

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

    plt.savefig(FOLDER_PATH / f"energy_convergence_comparison_N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}.png", dpi = 300)
    plt.show()

    # ---------------------------------------------------------------
    # relative (E_i - E_0) plot

    fig, ax = plt.subplots(1,1, figsize = (12,10))

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
    plt.savefig(FOLDER_PATH / f"energy_convergence_comparison_relative_N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}.png", dpi = 300)
    plt.show()


def convegence_comparison_by_tol():
    
    STOP_BY_TOL = True

    # multiple CN instances
    energies_cn = []
    for max_it_fixpoint in max_it_fixpoint_ls:
        max_it = num_iters / max_it_fixpoint
        energies = adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0, STOP_BY_TOL)
        energies_cn.append(energies)

    energies_gd = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, c0, alpha, num_iters, STOP_BY_TOL)
    energies_prox = gradient_descent_proximal(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL)
    energies_nest = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL)


    fig, ax = plt.subplots(1,1, figsize = (12,10))

    y_min = min(energies_nest)
    y_max = max(energies_nest)

    for ii, _cn in enumerate(energies_cn):
        ax.plot(_cn, label= fr"Crank Nicolson $N_{{fixpoint}}$ = {max_it_fixpoint_ls[ii]}", linewidth = 2, color = colors_cn[ii])
        ax.vlines(len(_cn)-1, y_min, y_max, linestyle = "--",linewidth = 3, color =  colors_cn[ii] )

    ax.plot(energies_gd, label="Gradient Descent", linewidth = 3, color =  colors_gd[0])
    ax.vlines(len(energies_gd)-1, y_min, y_max, linestyle = "--", linewidth = 3, color =  colors_gd[0] )
    ax.plot(energies_prox, label="Proximal Gradient Descent", linewidth = 3, color = colors_gd[1])
    ax.vlines(len(energies_prox)-1, y_min, y_max, linestyle = "--",linewidth = 3, color =  colors_gd[1] )
    ax.plot(energies_nest, label="Nesterov Proximal GD", linewidth = 3, color = colors_gd[2])
    ax.vlines(len(energies_nest)-1, y_min, y_max, linestyle = "--",linewidth = 3, color =  colors_gd[2] )

    #ax.set_yscale('log')

    ax.set_xlabel("Iteration $i$")
    ax.set_ylabel("Energy $E_i$")
    ax.set_title(f"Energy convergence comparison / stop at tolerance = {ENERGY_STOP_TOL}")

    ax.legend(loc = "upper right")
    ax.grid(True)
    fig.tight_layout()

    plt.savefig(FOLDER_PATH / f"energy_convergence_comparison_stop-by-limit={ENERGY_STOP_TOL}_N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}.png", dpi = 300)
    plt.show()


if __name__ == "__main__":

    # ---------------------------------------------------------------

    plotting_style()
    FOLDER_PATH = PATHS.PATH_COMPARISON
    LIVE_PLOT = False
    DATA_LOG = False

    # ---------------------------------------------------------------

    gridsize = 1.0
    N = 256
    epsilon = 1/20
    gamma = 1/200
    c0 = 9/32
    th = 1.0

    alpha = 1e-4       # step size for plain GD
    tau   = 5e-3       # step size for prox-based methods 
            # number of iterations for all methods
    ENERGY_STOP_TOL = 1e-8  # energy stopping tolerance if STOP_BY_TOL is true for methods 
    stop_limit = ENERGY_STOP_TOL
    tol = 1e-6

    prox_newton_iters = 50
    tol_newton = 1e-6

    max_it_fixpoint_ls = [1, 2, 5]
    dt = 1/10

    num_iters = 1_000 

    # ---------------------------------------------------------------
    x, k, modk, modk2 = define_spaces(gridsize, N)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
    M_k = sigma_k + gamma * epsilon * modk2

    # ---------------------------------------------------------------
    u0 = initialize_u0_random(N, REAL = True)
    # ---------------------------------------------------------------

    
    colors_cn = ['cornflowerblue', 'royalblue','blue']
    colors_gd = ['lightcoral', 'mediumseagreen', 'red']


    convergence_comparison()
    convegence_comparison_by_tol()