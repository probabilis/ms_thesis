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



def convergence_comparison():

    STOP_BY_TOL = False # which means full iterations are done, independent of energy convergence value
    
    # multiple CN instances
    energies_cn = []
    for max_it_fixpoint in max_it_fixpoint_ls:
        for dt in dt_ls:
            max_it = num_iters
            _, energies = adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0, STOP_BY_TOL)
            energies_cn.append(energies)

    _, energies_gd = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, c0, alpha, num_iters, LAPLACE_SPECTRAL=True, STOP_BY_TOL = STOP_BY_TOL)

    _, energies_prox = gradient_descent_proximal(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL)
    
    _, energies_nest = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, LAPLACE_SPECTRAL=True, STOP_BY_TOL = STOP_BY_TOL)

    # ---------------------------------------------------------------

    fig, ax = plt.subplots(1,1, figsize = (12,10))

    colors_cn = ['cornflowerblue', 'royalblue','blue', 'black']
    colors_gd = ['lightcoral', 'mediumseagreen', 'red']

    ii = 0
    for max_it_fixpoint in max_it_fixpoint_ls:
            for dt in dt_ls:
                
                ax.plot(energies_cn[ii], label= fr"Crank Nicolson $N_{{fixpoint}}$ = {max_it_fixpoint} | $dt$ = {dt}", linewidth = 2, color = colors_cn[ii])
                ii += 1

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

    #plt.savefig(FOLDER_PATH / f"energy_convergence_comparison_N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}.png", dpi = 300)
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
    N = 64
    epsilon = 1/100
    gamma = 1/2000

    c0 = 9/32
    th = 1.0

    alpha = 0.1       # step size for plain GD
    tau   = 0.1       # step size for prox-based methods 
            
    ENERGY_STOP_TOL = 1e-12  # energy stopping tolerance if STOP_BY_TOL is true for methods 
    stop_limit = ENERGY_STOP_TOL
    tol = 1e-4

    prox_newton_iters = 10
    tol_newton = 1e-6

    max_it_fixpoint_ls = [1, 5]
    dt_ls = [0.1, 1.0]

    num_iters = 5_000  # number of iterations for all methods

    # ---------------------------------------------------------------
    u0 = initialize_u0_random(N, REAL = True)
    # ---------------------------------------------------------------

    



    convergence_comparison()
    #convegence_comparison_by_tol()