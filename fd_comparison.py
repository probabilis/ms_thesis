import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from utils.env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic, log_data, main_colormap
from utils.pattern_formation import initialize_u0_random

from params.opt_params import labyrinth_data_params, sim_config, get_DataParameters, get_SimulationParamters
from params.opt_params import pgd_sim_params as ngd_sim_params

from optimization.gd_nesterov import gradient_descent_nesterov




if __name__ == "__main__":

    plotting_style()

    SINGLE_COMPARISON = True
    GAMMA_SWEEP = False

    plotting_style()
    FOLDER_PATH = PATHS.PATH_COMPARISON

    LIVE_PLOT = False
    DATA_LOG = False
    N = 64
    labyrinth_data_params = replace(labyrinth_data_params, N = N, gamma = 0.002, epsilon = 0.01)

    #gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u0 = initialize_u0_random(N, REAL = True)
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    _types = ["Finite Differences / PBC", "Finite Differences / Von Neumann", "Spectral method"]
    PBC_ls = [True, False, True]

    if SINGLE_COMPARISON:
        fig, axs = plt.subplots(len(_types), 2, figsize = (12,12))
        for ii in range(0, len(_types)):
            if ii == 2:
                sim_config = replace(sim_config, LAPLACE_SPECTRAL = True)
            
            u, energies = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), **asdict(sim_config), PBC = PBC_ls[ii])
            
            axs[ii, 0].imshow(u, cmap = main_colormap)
            if ii == 0:
                axs[ii, 0].set_title(f"La'Place: {_types[ii]}")
            else:
                axs[ii, 0].set_title(f"{_types[ii]}")

            axs[ii, 1].loglog(energies)
            axs[ii, 1].hlines(energies[-1],1,len(energies), label = f"$E_\\mathrm{{last}} = {energies[-1]:.3f}$", color = "black", linestyle = ":")
            if ii == 0:
                axs[ii, 1].set_title("Energy evolution")

            
            axs[ii, 0].set_box_aspect(1)
            axs[ii, 0].axes.get_xaxis().set_ticks([])
            axs[ii, 0].axes.get_yaxis().set_ticks([])

            axs[ii, 1].grid(color = "gray")
            axs[ii, 1].legend()
            print_bars()
            axs[ii, 1].set_ylabel("$E_n(u^{(ij)})$")
            
        axs[-1, 1].set_xlabel("iterator $n$")
        plt.savefig(FOLDER_PATH / "laplace_evaluation_comparison.png", dpi = 300)
        plt.show()


    ngd_sim_params = replace(ngd_sim_params, tau = 0.01)

    if GAMMA_SWEEP:
        gamma_ls = [1/80, 1/100, 1/200, 1/500, 1/1000, 1/2000, 1/4000]

        fig, axs = plt.subplots( len(gamma_ls), len(_types), figsize = (12,6) )

        sim_config = replace(sim_config, LAPLACE_SPECTRAL = False)
        u0 = initialize_u0_random(N, REAL = True)


        for tt, type in enumerate(_types):
            if tt == 2:
                sim_config = replace(sim_config, LAPLACE_SPECTRAL = True)
            
            for ii, _gamma in enumerate(gamma_ls):
                labyrinth_data_params = replace(labyrinth_data_params, gamma = _gamma)
                u, e = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), **asdict(sim_config), PBC=PBC_ls[tt])

                axs[ii, tt].imshow(u.cpu().numpy(), extent=(0,1,0,1))
                
                axs[ii, tt].set_box_aspect(1)
                axs[ii, tt].axes.get_xaxis().set_ticks([])
                axs[ii, tt].axes.get_yaxis().set_ticks([])

                axs[ii, tt].set_title(f"$\\gamma = {_gamma:.5f}$")

        
        fig.canvas.draw()  # ensures positions are compute

        for kk in range(0, len(_types)):

            bbox = axs[0, kk].get_position()
            x_center = 0.5 * (bbox.x0 + bbox.x1)
            y_top = bbox.y1 + 0.02
            title = f"LaPlace: {_types[kk]}"
            fig.text(x_center, y_top, title, ha="center", va = "bottom", fontsize=12)

        plt.savefig(FOLDER_PATH / "laplace_evaluation_gamma_sweep.png", dpi = 300)
        plt.show()