import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict

from pattern_formation import *
from params import labyrinth_data_params, pgd_sim_params, get_DataParameters
from env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic, log_data
# ---------------------------------------------------------------

def gradient_descent_proximal(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL = True, ENERGY_STOP_TOL = 1e-6):

    x, k, modk, modk2 = define_spaces(gridsize, N)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
    M_k = sigma_k + gamma * epsilon * modk2 # M_k multiplier for the quadratic term

    # --- main proximal-gradient loop ---
    energies = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]
    
    if LIVE_PLOT or DATA_LOG:
        fig1, ax1 = plt.subplots(figsize = (14,12))
        fig2, ax2 = plt.subplots(figsize = (10,10))
        plt.ion()

    u = u0.clone()

    try:
        for n in tqdm(range(num_iters), desc= "GD Proximal"):

            # forward step (gradient of smooth part (laplacian + FM) )
            ggrad = grad_g(u, M_k)     
            v = u - tau * ggrad

            # backward/prox step: solve pointwise prox
            u = prox_h(v, tau, gamma=gamma, eps=epsilon, c0=c0,maxiter=prox_newton_iters, tol=tol_newton)

            try:
                E = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
            except Exception as e:
                E = None

            energy_diff = energies[-1] - E
            energies.append(E)

            if LIVE_PLOT and (n % 1000) == 0:
                plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)
                plt.pause(1)

            if STOP_BY_TOL and energy_diff < ENERGY_STOP_TOL:
                print("dE[ii-1,ii]", abs(energy_diff) )
                break

    except KeyboardInterrupt:
        print("Exit.")  
    
    plt.ioff()

    if DATA_LOG:
        log_data(FOLDER_PATH, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)

    return u, energies

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    FOLDER_PATH = PATHS.PATH_PGD
    
    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log


    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u = initialize_u0_random(N, REAL = True)

    print_bars()
    print(labyrinth_data_params)
    print(pgd_sim_params)
    print_bars()

    gradient_descent_proximal(u, LIVE_PLOT, DATA_LOG, FOLDER_PATH,**asdict(labyrinth_data_params),**asdict(pgd_sim_params))


