import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from pattern_formation import initialize_u0_random

from params import labyrinth_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params

from env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic, log_data

from gd_nesterov import gradient_descent_nesterov
from params import sim_config


if __name__ == "__main__":

    plotting_style()
    FOLDER_PATH = PATHS.PATH_COMPARISON

    LIVE_PLOT = False
    DATA_LOG = False

    #gamma_start_fd = 0.00008
    #gamma_start_spectral = 0.00014
    
    labyrinth_data_params = replace(labyrinth_data_params, N = 64, gamma = 0.0008)


    #gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    N = 64
    u0 = initialize_u0_random(N, REAL = True)
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    fig, axs = plt.subplots(2,2)

    _types = ["Finite Differences", "Spectral"]

    for ii in range(0, 2):
        if ii == 1:
            sim_config = replace(sim_config, LAPLACE_SPECTRAL = True)
        
        u, e = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), **asdict(sim_config))
        
        axs[ii, 0].imshow(u)
        axs[ii, 0].set_title(f"LaPlace: {_types[ii]}")

        axs[ii, 1].plot(e)
        axs[ii, 1].set_title("Energy evolution")

    plt.savefig(FOLDER_PATH / "laplace_evaluation_comparison.png", dpi = 300)
    plt.show()