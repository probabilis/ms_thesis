import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from pattern_formation import initialize_u0_random

from params import labyrinth_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params

from env_utils import PATHS, print_bars, plotting_style, log_data

from gd_nesterov import gradient_descent_nesterov
from params import sim_config



if __name__ == "__main__":

    plotting_style()

    FOLDER_PATH = PATHS.PATH_PARAMS_STUDY

    LIVE_PLOT = False
    DATA_LOG = False

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    N = 64

    ngd_sim_params = replace(ngd_sim_params, num_iters = 5_000)
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print(sim_config)
    print_bars()

    N_est = 1

    gamma_ls = np.linspace(1, 0, 50)
    #gamma_ls = np.array([1/500, 1/800, 1/1000, 1/1500, 1/2000, 1/3000, 1/4000, 1/5000, 1/8000, 1/12000])
    energies_ls = []

    for ii in range(N_est):
        for gamma in gamma_ls:
            print(gamma)
            u0 = initialize_u0_random(N, REAL = True)

            labyrinth_data_params = replace(labyrinth_data_params, N = 64, gamma = gamma)
            
            u, energies = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), **asdict(sim_config))
            
            energies_ls.append( energies[-1] )

    
    def algebraic_scaling(gamma):
        return gamma**(1/2)

    plt.loglog(gamma_ls, energies_ls, label = "exp.")
    plt.loglog(gamma_ls, algebraic_scaling(gamma_ls), linestyle = "--", label = "theor.")

    plt.grid(color = "gray")
    plt.legend(loc = "lower right")
    plt.savefig(FOLDER_PATH / "asymptotic_energy.png", dpi = 300)
    plt.show()