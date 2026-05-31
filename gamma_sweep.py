import matplotlib.pyplot as plt
from dataclasses import replace, asdict

from utils.env_utils import PATHS, main_colormap, plotting_style
from utils.pattern_formation import initialize_u0_random
from params.opt_params import labyrinth_data_params, pgd_sim_params, get_DataParameters
from optimization.gd_nesterov import gradient_descent_nesterov


if __name__ == "__main__":

    plotting_style()

    labyrinth_data_params = replace(labyrinth_data_params, N = 200)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u0 = initialize_u0_random(N, REAL = True)

    pgd_sim_params = replace(pgd_sim_params, num_iters = 5000)
    gamma_ls = [1/10, 1/50, 1/100, 1/500, 1/800, 1/1000, 1/1500, 1/2000, 1/3000, 1/4000]

    LIVE_PLOT = False
    DATA_LOG = False
    
    FOLDER_PATH = PATHS.PATH_GAMMA_SWEEP

    fig, axs = plt.subplots( 2, int(len(gamma_ls)/2), figsize = (8,6) )

    axs = axs.ravel()

    for ii, _gamma in enumerate(gamma_ls):

        labyrinth_data_params = replace(labyrinth_data_params, gamma = _gamma)
        pgd_sim_params = replace(pgd_sim_params, num_iters = 5000)

        u, e = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(pgd_sim_params), STOP_BY_TOL=True)
        axs[ii].imshow(u.cpu().numpy(), cmap=main_colormap, extent=(0,1,0,1))
        
        axs[ii].set_box_aspect(1)
        axs[ii].axes.get_xaxis().set_ticks([])
        axs[ii].axes.get_yaxis().set_ticks([])

        axs[ii].set_title(f"$\\gamma = {_gamma:.5f}$")

    fig.tight_layout()
    plt.savefig(FOLDER_PATH / f"parameter_study_eps={epsilon}.png", dpi = 300)
    plt.show()