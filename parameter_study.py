from gd_nesterov import gradient_descent_nesterov
import matplotlib.pyplot as plt
from params import labyrinth_data_params, pgd_sim_params, get_DataParameters
from dataclasses import replace, asdict
from env_utils import PATHS

from pattern_formation import initialize_u0_random


if __name__ == "__main__":
    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u0 = initialize_u0_random(N, REAL = True)

    pgd_sim_params = replace(pgd_sim_params, num_iters = 1000   )

    epsilon_ls = [1/2, 1/20, 1/200, 1/2000]
    gamma_ls = [1/20, 1/200, 1/2000]

    LIVE_PLOT = False
    DATA_LOG = False
    FOLDER_PATH = PATHS.PATH_PARAMS_STUDY

    fig, axs = plt.subplots( len(epsilon_ls), len(gamma_ls), figsize = (20,20) )

    for ii, _epsilon in enumerate(epsilon_ls):
        for jj, _gamma in enumerate(gamma_ls):

            labyrinth_data_params = replace(labyrinth_data_params, gamma = _gamma, epsilon = _epsilon)

            u, e = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(pgd_sim_params), STOP_BY_TOL=False)
            
            axs[ii, jj].imshow(u.cpu().numpy(), cmap='gray', extent=(0,1,0,1))
            axs[ii, jj].set_title(f"$\\gamma$ = {_gamma}, $\\epsilon$ = {_epsilon}")

    plt.savefig(FOLDER_PATH / "params_study.png", dpi = 300)
    plt.show()