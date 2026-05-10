import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.env_utils import plotting_schematic, log_data
from utils.pattern_formation import prox_h, fourier_multiplier, energy_value, energy_value_fd,grad_g, grad_fd, define_spaces, dtype_real, device



def gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, LAPLACE_SPECTRAL = False, STOP_BY_TOL = True, ENERGY_STOP_TOL = 1e-10, PBC = True):
    
    # Nesterov proximal gradient descent algo
    
    x, k, modk, modk2 = define_spaces(gridsize, N)
    
    print("LaPlace Spectral Calculation: ", LAPLACE_SPECTRAL)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
    M_k = sigma_k + gamma * epsilon * modk2 * (2*torch.pi)**2  # M_k for spectral calculation of LAPLACE

    if LAPLACE_SPECTRAL:
        energies = [energy_value(gamma, epsilon, N, u0, c0, sigma_k, modk2)]
    else:        
        print("PBC: ", PBC)
        energies = [energy_value_fd(u0, sigma_k, N, gamma, epsilon, c0, PBC)]

    
    # --- initialization ---
    u_prev = u0.clone()
    u_curr = u0.clone()
    t_prev = 1.0


    # plotting setup
    if LIVE_PLOT or DATA_LOG:
        fig1, ax1 = plt.subplots(figsize = (14,12))
        fig2, ax2 = plt.subplots(figsize = (10,10))
        plt.ion()


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
                ggrad = grad_fd(y, sigma_k, N, gridsize, gamma, epsilon, c0, PBC)
            v = y - tau * ggrad

            # 3) backward step (proximal operator through double well)
            u_next = prox_h(v, tau, gamma, epsilon, c0, prox_newton_iters, tol_newton)

            # 4) update momentum history
            u_prev = u_curr
            u_curr = u_next
            t_prev = t_curr

            # 5) energy evaluation
            if LAPLACE_SPECTRAL:
                E = energy_value(gamma, epsilon, N, u_curr, c0, sigma_k, modk2)
            else:
                E = energy_value_fd(u_curr, sigma_k, N, gamma, epsilon, c0, PBC)
            
            energy_diff = energies[-1] - E
            energies.append(E)

            if (n % 100) == 0 and LIVE_PLOT:
                plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u_curr, energies, N, num_iters, gamma, epsilon, n)
                plt.pause(1)

            if STOP_BY_TOL and abs(energy_diff) < ENERGY_STOP_TOL:
                print("dE[ii-1, ii]", energy_diff)
                break

    except KeyboardInterrupt:
        print("Exit.")
    
    plt.ioff()
    
    if DATA_LOG:
        log_data(FOLDER_PATH, u_curr, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u_curr, energies, N, num_iters, gamma, epsilon, n)

    print("Last energy", np.mean(energies[10:-1]))
    return u_curr, energies
