from gd_nesterov import gradient_descent_nesterov
from env_utils import plotting_style, PATHS, print_bars
from pattern_formation import dtype_real, device, initialize_u0_random
from params import labyrinth_data_params, get_DataParameters, sim_config
from params import pgd_sim_params as ngd_sim_params
import torch

from lipschitz import evaluate_lipschitz_constant

from dataclasses import replace, asdict

import matplotlib.pyplot as plt

if __name__ == "__main__":

    EXAMPLE_1 = False
    EXAMPLE_2 = True



    plotting_style()
    FOLDER_PATH = PATHS.PATH_EXAMPLES

    LIVE_PLOT = False
    DATA_LOG = False

    labyrinth_data_params = replace(labyrinth_data_params, N = 100, gamma = 0.0008)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    ngd_sim_params = replace(ngd_sim_params, tau = evaluate_lipschitz_constant(gamma, epsilon, N, gridsize, th))

    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    u0 = initialize_u0_random(N)

    from gradient_descent import gradient_descent


    if EXAMPLE_1:
        u, e = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params),**asdict(sim_config) )
        #u, energies = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, c0 = 9/32, alpha = 0.1, num_iters = 50_000, LAPLACE_SPECTRAL=False, STOP_BY_TOL = True)


        u = u[30:70, :]
        im = plt.imshow(u.cpu(), cmap="managua", origin="lower")
        plt.axis("off")
        plt.colorbar(label = r"$m_z$", shrink = 0.4) #ax=ax, fraction=0.046, pad=0.04
        plt.tight_layout()
        plt.savefig(FOLDER_PATH / "pattern_example_with_colobar.png", dpi = 300)
        plt.show()


    u_ls = []
    
    fig, axs = plt.subplots(1, 5)
    

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