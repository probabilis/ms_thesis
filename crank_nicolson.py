from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pandas as pd
from dataclasses import asdict
from scipy.ndimage import gaussian_filter

from pattern_formation import *
from params import labyrinth_data_params, cn_sim_params, get_DataParameters, get_SimulationParamters, sin_data_params
from env_utils import PATHS, get_args, plotting_style, plotting_schematic, log_data, print_bars

# ---------------------------------------------------------------

def adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0, STOP_BY_TOL = True):
    # Adapted Crank-Nicolson Schematic (Reference Condette Paper)

    x, k, modk, modk2 = define_spaces(gridsize, N)
    
    L = gamma * epsilon * modk2 + fourier_multiplier(th * modk) # (2 * np.pi)**2 ... not correct for normalization
    L[0, 0] = fourier_multiplier(torch.tensor(0.0))

    time_vector = [0.0]
    energies = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]

    u_n = u0

    energy_diff = 1000
    ii = 0

    if LIVE_PLOT or DATA_LOG:
        fig1, ax1 = plt.subplots(figsize = (14,12))
        fig2, ax2 = plt.subplots(figsize = (10,10))
        plt.ion()

    fp_iterations = []

    pbar = tqdm(total=max_it, desc = "Crank Nicolson")

    try:
        while ii < max_it:

            if STOP_BY_TOL and energy_diff <= stop_limit: # only when STOP_BY_TOL is True, max_iterations will be cut
                break

            ii_fp, u_np1, err, conv = fixpoint(u_n, L, dt, N, epsilon, gamma, max_it_fixpoint, tol, c0)
            fp_iterations.append(ii_fp)
            
            if conv:
                curr_energy = energy_value(gamma, epsilon, N, u_np1, th, modk, modk2, c0)
                u_diff = torch.max(torch.abs(u_np1 - u_n)).item()

                energy_diff = energies[-1] - curr_energy
                
                energies.append(curr_energy)
                _time = time_vector[-1] + dt
                time_vector.append(_time)

                u_n = u_np1
                ii += 1
                
                if LIVE_PLOT and ii % 100 == 0:
                    plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u_n, energies, N, max_it, gamma, epsilon, ii)
                    plt.pause(1)
                
            else:
                dt = dt / 4
                if dt < 1e-12:
                    print("exit.")
                    raise RuntimeError("Time step too small. Exiting.")
                
            pbar.update(1)

    except KeyboardInterrupt:
        print("Exit.")


    pbar.close()
    plt.ioff()

    if DATA_LOG:
        log_data(FOLDER_PATH, u_n, energies, N, max_it, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u_n, energies, N, max_it, gamma, epsilon, ii)

    return energies

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    FOLDER_PATH = PATHS.PATH_CN

    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log


    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u0 = initialize_u0_random(N)
    #u0 = initialize_u0_sin(N, x)
    #u0 = initialize_u0_image('input_test.png')

    print_bars()
    print(labyrinth_data_params)
    print(cn_sim_params)
    print_bars()

    adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params), **asdict(cn_sim_params))

    