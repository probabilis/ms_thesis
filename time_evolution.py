import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from gradient_descent import gradient_descent

from params import labyrinth_data_params, get_DataParameters, get_SimulationParamters, gd_sim_params

from env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic, log_data
from pattern_formation import initialize_u0_random


# ---------------------------------------------------------------



# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    FOLDER_PATH = PATHS.PATH_EXAMPLES

    args = get_args()
    LIVE_PLOT = False
    DATA_LOG = False
    
    N = 200

    labyrinth_data_params = replace(labyrinth_data_params, N = N, gamma = 0.002)
    gd_sim_params = replace(gd_sim_params, num_iters = 1000)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)

    u0 = initialize_u0_random(N, REAL = True)
    
    print_bars()
    print(labyrinth_data_params)
    print(gd_sim_params)
    print_bars()
    
    u_ls, history = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(gd_sim_params), LAPLACE_SPECTRAL=False, SAVE_U_HISTORY=True)
    
    print(len(u_ls))

    fig, axs = plt.subplots( 5, 4, figsize = (14,14) )

    axs = axs.ravel()

    for ii, u in enumerate( u_ls[0:-1] ):


        axs[2*ii].imshow(u.cpu().numpy(), cmap='berlin', extent=(0,1,0,1))
        axs[2*ii].set_box_aspect(1)
        axs[2*ii].axes.get_xaxis().set_ticks([])
        axs[2*ii].axes.get_yaxis().set_ticks([])
        axs[2*ii].set_title(f"$ii = {ii+1}$")

        counts, bins = np.histogram(u_ls[ii])
        axs[2*ii+1].hist(bins[:-1], bins, weights=counts, color = "gray")
        axs[2*ii+1].axes.get_xaxis().set_ticks([])
        axs[2*ii+1].axes.get_yaxis().set_ticks([])

    fig.tight_layout()
    plt.savefig(FOLDER_PATH / f"domain_evolution.png", dpi = 300)
    plt.show()



    fig, axs = plt.subplots(2,2 , figsize = (12,8) )


    axs[0, 0].imshow(u_ls[-1].cpu().numpy(), cmap='berlin',origin="lower", extent=(0,1,0,1) )    

    fftu = torch.fft.fft2(u_ls[-1])
    real_fftu = torch.fft.fftshift(fftu)
    real_fftu = torch.abs(real_fftu)
    axs[0, 1].imshow(real_fftu, cmap = "berlin", origin = "lower")
    axs[0, 0].set_box_aspect(1)
    axs[0, 1].set_box_aspect(1)

    axs[1, 0].loglog(history["E_total"], label = "$E_{total}$")
    
    axs[1, 1].loglog(history["E_ex"], label="$E_{ex}$")
    axs[1, 1].loglog(history["E_demag"], label="$E_{demag}$")
    axs[1, 1].loglog(history["E_dw"], label="$E_{an}$")
    axs[1, 1].legend(loc = "lower right")

    axs[1, 0].grid(color = "gray")
    axs[1, 1].grid(color = "gray")

    plt.show()