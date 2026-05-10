import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.pattern_formation import define_spaces, fourier_multiplier, energy_value_fd, grad_fd, prox_h, dtype_real, device
from utils.env_utils import plotting_schematic, log_data



def gradient_descent_proximal(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL = True, ENERGY_STOP_TOL = 1e-12):
    
    x, k, modk, modk2 = define_spaces(gridsize, N)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
    M_k = sigma_k + gamma * epsilon * modk2 * (2 * torch.pi)**2

    PBC = True

    #energies = [energy_value(gamma, epsilon, N, u0, c0, sigma_k, modk2)]
    energies = [energy_value_fd(u0, sigma_k, N, gamma, epsilon, c0, PBC)]

    if LIVE_PLOT or DATA_LOG:
        fig1, ax1 = plt.subplots(figsize = (14,12))
        fig2, ax2 = plt.subplots(figsize = (10,10))
        plt.ion()

    u = u0.clone()

    try:
        for n in tqdm(range(num_iters), desc= "GD Proximal"):

            # forward step (gradient of smooth part (laplacian + FM) )
            #ggrad = grad_g(u, M_k)
            ggrad = grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC)     
            v = u - tau * ggrad

            # backward/prox step: solve pointwise prox
            u = prox_h(v, tau, gamma=gamma, eps=epsilon, c0=c0,maxiter=prox_newton_iters, tol=tol_newton)

            try:
                #E = energy_value(gamma, epsilon, N, u, c0, sigma_k, modk2)#
                E = energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC)

            except Exception as e:
                E = None

            energy_diff = energies[-1] - E
            energies.append(E)

            if LIVE_PLOT and (n % 1000) == 0:
                plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)
                plt.pause(1)

            if STOP_BY_TOL and energy_diff < ENERGY_STOP_TOL:
                print("dE[ii-1,ii]", abs(energy_diff) )
                break

    except KeyboardInterrupt:
        print("Exit.")  
    
    plt.ioff()

    if DATA_LOG:
        log_data(FOLDER_PATH, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)

    return u, energies

    


