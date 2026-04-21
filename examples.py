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

    labyrinth_data_params = replace(labyrinth_data_params, N = 256, gamma = 0.0004)

    sim_config = replace(sim_config, ENERGY_STOP_TOL = 1e-14)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)

    u0 = initialize_u0_random(N, REAL = True) 
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    EXAMPLE_1 = False
    EXAMPLE_2 = True


    if EXAMPLE_1:
        gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params),**asdict(sim_config) )

    u_ls = []
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4)

    LIVE_PLOT = False
    DATA_LOG = False

    from lipschitz import evaluate_lipschitz_constant

    if EXAMPLE_2:
        for ii, th in enumerate([0.1, 0.5, 1.0, 10, 100]):
            labyrinth_data_params = replace(labyrinth_data_params, th = th)

            eta = evaluate_lipschitz_constant(labyrinth_data_params.gamma, 
                                              labyrinth_data_params.epsilon,
                                              labyrinth_data_params.N,
                                              labyrinth_data_params.gridsize,
                                              labyrinth_data_params.th)
            
            ngd_sim_params = replace(ngd_sim_params, tau = eta)
            u, e = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params),**asdict(sim_config) )  

            axs[ii].imshow(u, cmap='managua', extent=(0,1,0,1))
            axs[ii].axes.get_xaxis().set_ticks([])
            axs[ii].axes.get_yaxis().set_ticks([])
            axs[ii].set_title(f"$\\delta = {th}$")

        plt.show()