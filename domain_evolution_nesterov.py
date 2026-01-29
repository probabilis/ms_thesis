import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from gd_proximal import prox_h
from params import labyrinth_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params

from env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic, log_data

from pattern_formation import fourier_multiplier, energy_value, dtype_real, device, initialize_u0_random, define_spaces, grad_g
from pattern_formation import energy_value_fd, grad_fd

# ---------------------------------------------------------------


def gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, LAPLACE_SPECTRAL = None, STOP_BY_TOL = False, ENERGY_STOP_TOL = 1e-10):
    # Nesterov proximal gradient with adaptive restart

    x, k, modk, modk2 = define_spaces(gridsize, N)

    # --- spaces ---
    if LAPLACE_SPECTRAL is None:
        LAPLACE_SPECTRAL = True
        print("LaPlace Spectral Calculation: ", LAPLACE_SPECTRAL)
    
    if LAPLACE_SPECTRAL:
        energies = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]
        sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
        
    else:        
        sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
        energies = [energy_value_fd(u0, sigma_k, N, gamma, epsilon, c0)]
    

    # M_k for spectral calculation of LAPLACE
    M_k = sigma_k + gamma * epsilon * modk2 

    # --- initialization ---
    u_prev = u0.clone()
    u_curr = u0.clone()
    t_prev = 1.0

    u_ls = []

    # --- main loop ---
    try:

        for n in tqdm(range(1, num_iters+1), desc="Nesterov GD"):
            # 1) extrapolation
            t_curr = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev)**0.5)
            beta = (t_prev - 1.0) / t_curr
            y = u_curr + beta * (u_curr - u_prev)

            # 2) forward step (gradient of smooth part) / linear + nonlocal part (FM part + laplacian)
            if LAPLACE_SPECTRAL:
                ggrad = grad_g(y, M_k)
            else:
                ggrad = grad_fd(y, sigma_k, N, gridsize, gamma, epsilon, c0)
            v = y - tau * ggrad

            # 3) backward step (proximal operator through double well)
            u_next = prox_h(v, tau, gamma, epsilon, c0, prox_newton_iters, tol_newton)

            # 4) update momentum history
            u_prev = u_curr
            u_curr = u_next
            t_prev = t_curr

            # 5) energy evaluation
            if LAPLACE_SPECTRAL:
                E = energy_value(gamma, epsilon, N, u_curr, th, modk, modk2, c0)
            else:
                E = energy_value_fd(u_curr, sigma_k, N, gamma, epsilon, c0)
            
            energy_diff = energies[-1] - E
            energies.append(E)

            if (n % 100) == 0 and LIVE_PLOT:
                #plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u_curr, energies, N, num_iters, gamma, epsilon, n)
                u_ls.append(u_curr)


            if STOP_BY_TOL and abs(energy_diff) < ENERGY_STOP_TOL:
                print("dE[ii-1,ii]", energy_diff)
                break

    except KeyboardInterrupt:
        print("Exit.")

    return u_ls

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    FOLDER_PATH = PATHS.PATH_EXAMPLES

    args = get_args()
    LIVE_PLOT = True
    DATA_LOG = args.data_log

    labyrinth_data_params = replace(labyrinth_data_params, N = 64, gamma = 0.002)
    ngd_sim_params = replace(ngd_sim_params, num_iters = 2000)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    N = 64
    u0 = initialize_u0_random(N, REAL = True)
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    u_ls = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), LAPLACE_SPECTRAL=False )
    print(u_ls)
    fig, axs = plt.subplots( 4, 5, figsize = (14,14) )

    axs = axs.ravel()

    print(len(u_ls))
    for ii, u in enumerate(u_ls):
        axs[ii].imshow(u.cpu().numpy(), cmap='plasma', extent=(0,1,0,1))
        axs[ii].set_box_aspect(1)
        axs[ii].axes.get_xaxis().set_ticks([])
        axs[ii].axes.get_yaxis().set_ticks([])
        axs[ii].set_title(f"$ii = {ii+1}$")

    fig.tight_layout()
    plt.savefig(FOLDER_PATH / f"domain_evolution.png", dpi = 300)
    plt.show()