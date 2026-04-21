import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from pattern_formation import double_well_potential, fourier_multiplier,energy_value, dtype_real, device, initialize_u0_random, define_spaces
from pattern_formation import energy_value_fd, grad_fd
from gd_proximal import prox_h

from params import exp_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params

from env_utils import PATHS, print_bars, get_args, plotting_style, plotting_schematic_eval, log_data, log_data_history
from read import read_csv

from spectrum_analysis import radial_wavelength_spectrum


# -----------------------------------------------------------
# FD evaluation of LaPlace

def energy_value_fd_with_data(gamma, epsilon, N, u, sigma_k, c0, _lambda, u_exp, PBC = True):
    """Total energy = functional + Mean Squared Error (L2) data loss term."""
    E_GRAD, E_DW, E_FM = energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC, RETURN_SEPERATE=True)
    E_data = 0.5 * _lambda * torch.sum((u - u_exp)**2) / N**2
    
    return E_GRAD.item(), E_DW.item(), E_FM.item(), E_data.item()


def grad_fd_with_data(u, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, PBC):

    grad_lin = grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC)
    grad_data = _lambda * (u - u_exp)

    return grad_lin + grad_data

# -----------------------------------------------------------
# FD evaluation + wavevector fitting

def energy_value_fd_with_data_and_kpeak(gamma, epsilon, N, u, sigma_k, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC = True):
    """Total energy = labyrinth functional + L2 data fidelity."""
    E_GRAD, E_DW, E_FM = energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC, RETURN_SEPERATE=True)
    E_data = 0.5 * _lambda * ( torch.sum((u - u_exp)**2) / N**2 + (k_peak_sim - k_peak_exp)**2 )

    return E_GRAD.item(), E_DW.item(), E_FM.item(), E_data.item()



def grad_fd_with_data_and_kpeak(u, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC):

    grad_lin = grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC)
    grad_data = _lambda * ( (u - u_exp) + (k_peak_sim - k_peak_exp) ) 

    return grad_lin + grad_data

# -----------------------------------------------------------


def energy_value_fd_with_data_huber_loss(gamma, epsilon, N, u, sigma_k, c0, _lambda, u_exp, delta=2.0, PBC=True):
    """Total energy = labyrinth functional + Huber data fidelity."""
    E_GRAD, E_DW, E_FM = energy_value_fd(
        u, sigma_k, N, gamma, epsilon, c0, PBC, RETURN_SEPERATE=True
    )

    r = u - u_exp
    abs_r = torch.abs(r)

    huber = torch.where(
        abs_r <= delta,
        0.5 * r**2,
        delta * (abs_r - 0.5 * delta)
    )

    E_data = _lambda * torch.sum(huber) / N**2

    return E_GRAD.item(), E_DW.item(), E_FM.item(), E_data.item()


def grad_fd_with_data_huber_loss(u, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, delta=2.0, PBC=True):
    """Gradient of total energy = physical gradient + Huber data term."""
    grad_lin = grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC)

    r = u - u_exp
    abs_r = torch.abs(r)

    grad_data = torch.where(
        abs_r <= delta,
        r,
        delta * torch.sign(r)
    )

    grad_data = _lambda * grad_data # / N**2

    return grad_lin + grad_data



# -----------------------------------------------------------

# Nesterov PGD with experimental image data

def gradient_descent_nesterov_evaluation(
    u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH, gridsize, N, th, gamma, epsilon, tau, c0,
    num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL = False, ENERGY_STOP_TOL = 1e-6):



    PBC = False
    # --- spaces ---

    x, k, modk, modk2 = define_spaces(gridsize, N)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)

    # --- initialization ---
    u_prev = u0.clone()
    u_curr = u0.clone()
    t_prev = 1.0

    u_exp = u_exp.to(device=device, dtype=dtype_real)

    loss_types = ["MSE", "MSE+k_peak", "HuberLoss"]

    LOSS_TYPE = loss_types[1]

    if LOSS_TYPE == "MSE+k_peak":
        k_peak_exp = radial_wavelength_spectrum(u_exp, gridsize/N)["k_peak"]
        k_peak_sim = radial_wavelength_spectrum(u0, gridsize/N)["k_peak"]


    # energy history starts at u0
    if LOSS_TYPE == "MSE":
        E_ex, E_demag, E_dw, E_data = energy_value_fd_with_data(gamma, epsilon, N, u0, sigma_k, c0, _lambda, u_exp, PBC)
    if LOSS_TYPE == "HuberLoss":
        E_ex, E_demag, E_dw, E_data = energy_value_fd_with_data_huber_loss(gamma, epsilon, N, u0, sigma_k, c0, _lambda, u_exp, PBC = PBC)
    if LOSS_TYPE == "MSE+k_peak":
        E_ex, E_demag, E_dw, E_data = energy_value_fd_with_data_and_kpeak(gamma, epsilon, N, u0, sigma_k, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC)
        

    E0 = E_ex + E_demag + E_dw + E_data
    energies = [E0]
    energies_ex = [E_ex]
    energies_demag = [E_demag]
    energies_dw = [E_dw]
    energies_data = [E_data]


    if LIVE_PLOT:
        plt.ion()
        fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
        fig2, ax2 = plt.subplots(1,1, figsize=(5,5))

    try:
        for n in tqdm(range(1, num_iters+1), desc="Nesterov GD for Data"):

            # 1) Nesterov extrapolation
            t_curr = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev)**0.5)
            beta = (t_prev - 1.0) / t_curr
            y = u_curr + beta * (u_curr - u_prev)

            if LOSS_TYPE == "MSE+k_peak": 
                k_peak_sim = radial_wavelength_spectrum(y, gridsize/N)["k_peak"]

            # 2) forward step (gradient of smooth part)
            if LOSS_TYPE == "MSE":
                ggrad = grad_fd_with_data(y, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, PBC)
            if LOSS_TYPE == "HuberLoss":
                ggrad = grad_fd_with_data_huber_loss(y, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, PBC = PBC)
            if LOSS_TYPE == "MSE+k_peak":    
                ggrad = grad_fd_with_data_and_kpeak(y, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC)

            v = y - tau * ggrad

            # 3) backward step (proximal operator for double well only)
            u_next = prox_h(v, tau, gamma, epsilon, c0, prox_newton_iters, tol_newton)

            # 4) update
            u_prev = u_curr
            u_curr = u_next
            t_prev = t_curr
        
            if LOSS_TYPE == "MSE+k_peak": 
                k_peak_sim = radial_wavelength_spectrum(u_curr, gridsize/N)["k_peak"]

            # 5) energy
            if LOSS_TYPE == "MSE":
                E_ex, E_demag, E_dw, E_data = energy_value_fd_with_data(gamma, epsilon, N, u_curr, sigma_k, c0, _lambda, u_exp, PBC)
            if LOSS_TYPE == "HuberLoss":
                E_ex, E_demag, E_dw, E_data = energy_value_fd_with_data_huber_loss(gamma, epsilon, N, u_curr, sigma_k, c0, _lambda, u_exp, PBC = PBC)
            if LOSS_TYPE == "MSE+k_peak":    
                E_ex, E_demag, E_dw, E_data = energy_value_fd_with_data_and_kpeak(gamma, epsilon, N, u_curr, sigma_k, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC)

            E = E_ex + E_demag + E_dw + E_data
            energy_diff = energies[-1] - E
            
            energies.append(E)
            energies_ex.append(E_ex)
            energies_demag.append(E_demag)
            energies_dw.append(E_dw)
            energies_data.append(E_data)


            if (n % 100) == 0 and LIVE_PLOT:
                plotting_schematic_eval(OUTPUT_PATH, ax1, fig1, ax2, fig2, u_curr, energies, N, num_iters, gamma, epsilon, _lambda, n)
                plt.pause(1)
                
            if STOP_BY_TOL and energy_diff < ENERGY_STOP_TOL:
                print("dE", energy_diff)
                break


    except KeyboardInterrupt:
        print("Exit.")
        plt.close()

    plt.ioff()



    history = {
        "E_total": energies,
        "E_ex": energies_ex,
        "E_demag": energies_demag,
        "E_dw": energies_dw,
        "E_data": energies_data,
    }

    if DATA_LOG:
        log_data_history(OUTPUT_PATH, u_curr, history, N, num_iters, gamma, epsilon, _lambda)

    return u_curr, history 

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    
    INPUT_PATH = PATHS.BASE_EXPDATA
    OUTPUT_PATH = PATHS.PATH_EVALUATION

    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log

    # ---------------------------------------------------------------

    dataset = "data_01"
    recording = "001"

    INPUT_FILE_PATH = PATHS.BASE_EXPDATA / f"{dataset}/csv/mcd_slice_{recording}.csv"
    print(f"Reading {INPUT_FILE_PATH} as experimental image data.")
    u_exp = read_csv(INPUT_FILE_PATH, PLOT = True)

    if u_exp.shape[0] != u_exp.shape[1]:
        raise ValueError("Experimental data should be quadratic (NxN tensor).")
    N_exp = u_exp.shape[0]

    # ---------------------------------------------------------------

    num_iters = 5000
    ENERGY_STOP_TOL = 1e-12

    exp_data_params = replace(exp_data_params, gamma = 0.004) 
    ngd_sim_params = replace(ngd_sim_params, num_iters = num_iters, tau = 0.001) # smaller tau because of image

    gridsize, N, th, epsilon, gamma = get_DataParameters(exp_data_params)
    u0 = initialize_u0_random(N, REAL=True)

    print_bars()
    print(exp_data_params)
    print(ngd_sim_params)
    print_bars()

    _lambda = 0.01
    print(f"Learning Rate Lambda: {_lambda}")
    print_bars()

    # ---------------------------------------------------------------

    u, history = gradient_descent_nesterov_evaluation(u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH,**asdict(exp_data_params),**asdict(ngd_sim_params), STOP_BY_TOL=True, ENERGY_STOP_TOL=ENERGY_STOP_TOL)

    fig, axs = plt.subplots(2,2 , figsize = (12,8) )



    axs[0, 0].imshow(u_exp.cpu().numpy(), cmap = 'gray',origin="lower", extent = (0,1,0,1) )
    axs[0, 1].imshow(u.cpu().numpy(), cmap='gray',origin="lower", extent=(0,1,0,1) )    
    axs[0, 0].set_box_aspect(1)
    axs[0, 1].set_box_aspect(1)
    
    #axs[0].set_title(f"$\\gamma = {gamma}, \\lambda = {_lambda}$")

    axs[1, 0].loglog(history["E_total"], label = "$E_{total}$")

    axs[1, 0].loglog(history["E_ex"], label="$E_{ex}$")
    axs[1, 0].loglog(history["E_demag"], label="$E_{demag}$")
    axs[1, 0].loglog(history["E_dw"], label="$E_{an}$")
    axs[1, 0].loglog(history["E_data"], label="$E_{data}$")    
    axs[1, 0].legend()

    material = "Ta/CoFeB/MgO"

    txt = (
        f"Material: {material}\n"
        f"gamma     = {gamma}\n"
        f"epsilon   = {epsilon}\n"
        f"lambda    = {_lambda}\n"
        f"iters     = {len(history['E_total'])-1}\n"
        f"E_final   = {history['E_total'][-1]:.3e}"
    )
    axs[1, 1].axis("off")
    axs[1, 1].text(0.0, 1.0, txt, va="top", ha="left", fontsize=11)

    #axs[1].set_title(f"$\\Delta E < {ENERGY_STOP_TOL}$")
    #axs[1].set_yscale('log')

    fig.tight_layout()
    #plt.savefig(OUTPUT_PATH / f"evaluation_gamma={gamma}_lambda={_lambda:.2f}_num-iters={num_iters}.png", dpi = 300)
    plt.show()

    
