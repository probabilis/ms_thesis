from gd_nesterov import gradient_descent_nesterov
from env_utils import plotting_style, PATHS, print_bars
from pattern_formation import dtype_real, device, initialize_u0_random
from params import labyrinth_data_params, get_DataParameters, sim_config
from params import pgd_sim_params as ngd_sim_params
import torch

from dataclasses import replace, asdict


if __name__ == "__main__":


    plotting_style()
    FOLDER_PATH = PATHS.PATH_EXAMPLES

    LIVE_PLOT = True
    DATA_LOG = True

    labyrinth_data_params = replace(labyrinth_data_params, N = 256, gamma = 0.0001)

    sim_config = replace(sim_config, ENERGY_STOP_TOL = 1e-14)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)

    u0 = initialize_u0_random(N, REAL = True) 
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params),**asdict(sim_config) )