import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict

from pattern_formation import fourier_multiplier,energy_value, dtype_real, device, initialize_u0_random, define_spaces, grad_g
from gradient_descent_proximal import prox_h
from params import labyrinth_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params
from env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic, log_data


# ---------------------------------------------------------------

def gradient_descent_nesterov(u, LIVE_PLOT, DATA_LOG, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton):
    # Nesterov proximal gradient with adaptive restart

    # Initialize momentum variables
    u_prev = u.clone()           # u^{k-1}
    t_prev = 1.0                 # t_{k-1}
    u_curr = u.clone()           # u^{k}


    x, k, modk, modk2 = define_spaces(gridsize, N)
    energies = []


    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
    M_k = sigma_k + gamma * epsilon * modk2 # M_k multiplier for the quadratic term

    # parameters for restart and diagnostics
    USE_RESTART = True
    PRINT_EVERY = 1000
    bad_restarts = 0

    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
    fig2, ax2 = plt.subplots(1,1, figsize=(5,5))

    plt.ion()

    try:
        for n in tqdm(range(num_iters)):

            # 1) extrapolation (Nesterov momentum)
            t_curr = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev) ** 0.5)   # t_k
            beta = (t_prev - 1.0) / t_curr                               # momentum factor

            y = u_curr + beta * (u_curr - u_prev)

            # 2) forward step at extrapolated point
            # (gradient of smooth part (laplacian + FM) )
            ggrad = grad_g(y, M_k)                    # gradient of g at y
            v = y - tau * ggrad

            # 3) backward/prox step
            u_next = prox_h(v, tau, gamma, epsilon,c0, prox_newton_iters, tol_newton)

            # compute energy to possibly restart
            try:
                E_next = energy_value(gamma, epsilon, N, u_next, th, modk, modk2, c0)
            except Exception:
                E_next = None

            # compute energy of current iterate to compare u_curr
            if n == 0:
                try:
                    E_curr_val = energy_value(gamma, epsilon, N, u_curr, th, modk, modk2, c0)
                except Exception:
                    E_curr_val = None
            else:
                E_curr_val = energies[-1]

            # 4) adaptive restart: if energy increased -> reset momentum
            restarted = False
            if USE_RESTART and (E_next is not None) and (E_curr_val is not None):
                if E_next > E_curr_val + 1e-14:   # small tolerance to avoid numerical jitter
                    # restart: drop momentum and do a plain proximal gradient step from u_curr
                    restarted = True
                    bad_restarts += 1

                    # reset t and momentum history
                    t_curr = 1.0
                    beta = 0.0
                    # recompute plain prox-step: v_plain = u_curr - tau * grad_g(u_curr)
                    v_plain = u_curr - tau * grad_g(u_curr, M_k)
                    u_next = prox_h(v_plain, tau, gamma, epsilon,c0, prox_newton_iters, tol_newton)
                    try:
                        E_next = energy_value(gamma, epsilon, N, u_next, th, modk, modk2, c0)
                    except Exception:
                        E_next = None

            # 5) update momentum history for next iteration
            u_prev = u_curr
            u_curr = u_next
            t_prev = t_curr

            energies.append(E_next)
            if (n % PRINT_EVERY) == 0:
                msg = f"iter {n:6d}: E={E_next:.6e} alpha={tau:.2e}"
                if restarted:
                    msg += " (RESTART)"
                print(msg)

            if (n % 1000) == 0 and LIVE_PLOT:
                plotting_schematic(folder_path, ax1, fig1, ax2, fig2, u_curr, energies, N, num_iters, gamma, epsilon, n)
                plt.pause(1)
        

    except KeyboardInterrupt:
        print("Exit.")
    
    plt.ioff()
    
    if DATA_LOG:
        log_data(folder_path, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(folder_path, ax1, fig1, ax2, fig2, u_curr, energies, N, num_iters, gamma, epsilon, n)
        plt.pause(1)

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    folder_path = PATHS.PATH_NESTEROV

    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u = initialize_u0_random(N, REAL = True)
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    gradient_descent_nesterov(u, LIVE_PLOT, DATA_LOG,**asdict(labyrinth_data_params),**asdict(ngd_sim_params) )