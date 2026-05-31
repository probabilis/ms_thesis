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

    LIVE_PLOT = False
    DATA_LOG = False
    
    FOLDER_PATH = PATHS.PATH_PARAMS_STUDY

    th_ls = [5.0, 1.0, 0.5, 0.2]

    fig, axs = plt.subplots( 1, int(len(th_ls)), figsize = (8,4) )

    axs = axs.ravel()

    for ii, _th in enumerate(th_ls):

        labyrinth_data_params = replace(labyrinth_data_params, th = _th)
        pgd_sim_params = replace(pgd_sim_params, num_iters = 5000)

        u, e = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(pgd_sim_params), STOP_BY_TOL=True)
        axs[ii].imshow(u.cpu().numpy(), cmap=main_colormap, extent=(0,1,0,1))
        
        axs[ii].set_box_aspect(1)
        axs[ii].axes.get_xaxis().set_ticks([])
        axs[ii].axes.get_yaxis().set_ticks([])

        axs[ii].set_title(f"$\\delta = {_th:.5f}$")

    fig.suptitle(f"$\\gamma = {labyrinth_data_params.gamma}$")
    fig.tight_layout()
    plt.savefig(FOLDER_PATH / f"thickness_sweep_eps={epsilon}_gamma={labyrinth_data_params.gamma}.png", dpi = 300)
    plt.show()