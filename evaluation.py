import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from pattern_formation import fourier_multiplier,energy_value, dtype_real, device, initialize_u0_random, define_spaces, grad_g
from gd_proximal import prox_h
from params import labyrinth_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params

from env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic_eval, log_data
from read import read_csv


def grad_g_with_data(u, M_k, N, _lambda, u_exp):
    """Gradient of smooth part = spectral linear term + quadratic data term."""
    Fu = torch.fft.fft2(u) / (N**2)
    grad_lin = torch.fft.ifft2(M_k * Fu).real * (N**2)
    grad_data = _lambda * (u - u_exp)
    return grad_lin + grad_data

def energy_value_with_data(gamma, epsilon, N, u, th, modk, modk2, c0,
                           _lambda, u_exp):
    """Total energy = labyrinth functional + L2 data fidelity."""
    E_base = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
    E_data = 0.5 * _lambda * torch.sum((u - u_exp)**2)
    return (E_base + E_data).item()

# -----------------------------
# Nesterov PGD with data fidelity
# -----------------------------

def gradient_descent_nesterov_evaluation(
    u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH, gridsize, N, th, gamma, epsilon, tau, c0,
    num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL = False, ENERGY_DIFF_STOP_TOL = 1e-1):
    """
    Nesterov (FISTA-like) proximal gradient with adaptive restart,
    augmented by a quadratic data term (λ/2)||u - u_exp||^2.
    """

    # --- spaces ---
    x, k, modk, modk2 = define_spaces(gridsize, N)
    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
    M_k = sigma_k + gamma * epsilon * modk2

    # --- initialization ---
    u_prev = u0.clone()
    u_curr = u0.clone()
    t_prev = 1.0

    # match dtype/device for experimental data
    u_exp = u_exp.to(device=device, dtype=dtype_real)

    # energy history starts at u0
    energies = [energy_value_with_data(gamma, epsilon, N, u0, th, modk, modk2,c0, _lambda, u_exp)]
    energies_diff = []
    energies_diff_sum_index = 10

    # plotting
    if LIVE_PLOT or DATA_LOG:
        fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
        fig2, ax2 = plt.subplots(1,1, figsize=(5,5))
        plt.ion()

    try:
        for n in tqdm(range(1, num_iters+1), desc="Nesterov GD for Data"):
            # 1) Nesterov extrapolation
            t_curr = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev)**0.5)
            beta = (t_prev - 1.0) / t_curr
            y = u_curr + beta * (u_curr - u_prev)

            # 2) forward step (gradient of smooth part)
            ggrad = grad_g_with_data(y, M_k, N, _lambda, u_exp)
            v = y - tau * ggrad

            # 3) backward step (proximal operator for double well only)
            u_next = prox_h(v, tau, gamma, epsilon, c0,
                            prox_newton_iters, tol_newton)

            # 4) update
            u_prev = u_curr
            u_curr = u_next
            t_prev = t_curr

            # 5) energy
            E = energy_value_with_data(gamma, epsilon, N, u_curr,th, modk, modk2, c0, _lambda, u_exp)
            energy_diff = energies[-1] - E
            energies.append(E)
            energies_diff.append(abs(energy_diff))

            if (n % 10) == 0 and LIVE_PLOT:
                plotting_schematic_eval(OUTPUT_PATH, ax1, fig1, ax2, fig2, u_curr, energies, N, num_iters, gamma, epsilon, _lambda, n)
                plt.pause(1)

            if n > energies_diff_sum_index:
                energy_diff_sum = sum(energies_diff[n-energies_diff_sum_index:-1]) 
                #print("Sum of energy differnces: ", energy_diff_sum)        
                
            if STOP_BY_TOL and (n > energies_diff_sum_index) and (energy_diff_sum < ENERGY_DIFF_STOP_TOL):
                print(f"Energy convergence : sum(last {energies_diff_sum_index} dE_i) = {energy_diff_sum:.3f} < {ENERGY_DIFF_STOP_TOL}")
                break

    except KeyboardInterrupt:
        print("Exit.")

    if DATA_LOG or LIVE_PLOT:
        plotting_schematic_eval(OUTPUT_PATH, ax1, fig1, ax2, fig2, u_curr, energies, N, num_iters, gamma, epsilon, _lambda, n) 
        plt.ioff()

    return u_curr, energies

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    
    INPUT_PATH = PATHS.BASE_INPUT
    OUTPUT_PATH = PATHS.PATH_EVALUATION

    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log

    # ---------------------------------------------------------------

    u_exp = read_csv("data/data_01/csv/mcd_slice_000.csv", "standardize")

    if u_exp.shape[0] != u_exp.shape[1]:
        raise ValueError("Experimental data should be quadratic (NxN tensor).")
    N_exp = u_exp.shape[0]

    # ---------------------------------------------------------------

    num_iters = 5000

    labyrinth_data_params = replace(labyrinth_data_params, N = N_exp)
    labyrinth_data_params = replace(labyrinth_data_params, gamma = 0.02)
    ngd_sim_params = replace(ngd_sim_params, num_iters = num_iters)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u0 = initialize_u0_random(N, REAL=True)

    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    _lambda = torch.std(u0)
    print(f"Learning Rate Lambda := std(u0) = {_lambda}")
    print_bars()

    # ---------------------------------------------------------------

    u, energies = gradient_descent_nesterov_evaluation(u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH,**asdict(labyrinth_data_params),**asdict(ngd_sim_params), STOP_BY_TOL=True)
    
    fig, axs = plt.subplots(1,2) # , figsize = (8,8)
    axs[0].imshow(u.cpu().numpy(), cmap='gray',origin="lower", extent=(0,1,0,1))
    axs[0].set_box_aspect(1)
    axs[0].set_title(f"$\\gamma = {gamma}, \\lambda := std(u_0) = {_lambda:.2f}$")

    axs[1].plot(np.arange(0,len(energies), 1), energies)
    axs[1].set_box_aspect(1)
    axs[1].set_title(f"$\\Sigma \\Delta E < 0.1$")
    #axs[1].set_yscale('log')
    fig.tight_layout()
    plt.savefig(OUTPUT_PATH / f"evaluation_gamma={gamma}_lambda={_lambda:.2f}_num-iters={num_iters}.png", dpi = 300)
    plt.show()

    
