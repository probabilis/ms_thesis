import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Literal

from utils.env_utils import plotting_schematic_eval, log_data, log_data_history

from utils.pattern_formation import fourier_multiplier, dtype_real, device, define_spaces
from utils.pattern_formation import energy_value_fd, grad_fd, prox_h
from spectrum_analysis import radial_wavelength_spectrum


# -----------------------------------------------------------
# Energy and Gradient evaluation with Data Fitting Term
# -----------------------------------------------------------
# FD evaluation + MSE of data

def energy_value_fd_with_data(gamma, epsilon, N, u, sigma_k, c0, _lambda, u_exp, PBC = True):
    """
    Energy functional with data fitting term through Mean Squared Error Loss Term (L2) weighted by lambda
    """

    E_GRAD, E_DW, E_FM = energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC, RETURN_SEPERATE=True)
    E_DATA = 0.5 * _lambda * torch.sum((u - u_exp)**2) / N**2
    
    return E_GRAD, E_DW, E_FM, E_DATA.item()


def grad_fd_with_data(u, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, PBC):
    """
    Gradient of Energy functional with data fitting term through MSE (L2) weighted by lambda
    """

    grad_lin = grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC)
    grad_data = _lambda * (u - u_exp)

    return grad_lin + grad_data

# -----------------------------------------------------------
# FD evaluation + MSE of data + wavevector fit

def energy_value_fd_with_data_and_kpeak(gamma, epsilon, N, u, sigma_k, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC = True):
    """
    same as above but with maximum value from radial wavevector spectrum k_peak
    
    _lambda * 0.5 * [ MSE(u - u_exp)^2 + (k_peak_sim - k_peak_exp)^2]

    """
    E_GRAD, E_DW, E_FM = energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC, RETURN_SEPERATE=True)
    E_DATA = 0.5 * _lambda * ( torch.sum((u - u_exp)**2) / N**2 + (k_peak_sim - k_peak_exp)**2 )

    return E_GRAD, E_DW, E_FM, E_DATA.item()



def grad_fd_with_data_and_kpeak(u, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC):
    """
    same as above but with maximum value from radial wavevector spectrum k_peak
    """

    grad_lin = grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC)
    grad_data = _lambda * ( (u - u_exp) + (k_peak_sim - k_peak_exp) ) 

    return grad_lin + grad_data

# -----------------------------------------------------------
# FD ev

def energy_value_fd_with_data_huber_loss(gamma, epsilon, N, u, sigma_k, c0, _lambda, u_exp, delta=0.5, PBC=True):
    """
    Huber Loss instead of MSE with threshold parameter delta
    """
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

    E_DATA = _lambda * torch.sum(huber) / N**2

    return E_GRAD, E_DW, E_FM, E_DATA.item()


def grad_fd_with_data_huber_loss(u, sigma_k, N, gridsize, gamma, epsilon, c0, _lambda, u_exp, delta=0.5, PBC=True):
    """Huber Loss instead of MSE with threshold parameter delta"""
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
    num_iters, prox_newton_iters, tol_newton, LOSS_TYPE : Literal["MSE", "MSE+k_peak", "HuberLoss"], 
    STOP_BY_TOL = False, ENERGY_STOP_TOL = 1e-6, PBC = False):

    print("Using loss type: ", LOSS_TYPE)

    # Parameter definitions
    # --- spaces ---
    _, _, modk, _ = define_spaces(gridsize, N)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)

    # --- initialization ---
    u_prev = u0.clone()
    u_curr = u0.clone()
    t_prev = 1.0

    u_exp = u_exp.to(device=device, dtype=dtype_real)


    if LOSS_TYPE == "MSE+k_peak":
        k_peak_exp = radial_wavelength_spectrum(u_exp, gridsize/N)["k_peak"]
        k_peak_sim = radial_wavelength_spectrum(u0, gridsize/N)["k_peak"]


    # energy history starts at u0
    if LOSS_TYPE == "MSE":
        E_GRAD, E_DW, E_FM, E_DATA = energy_value_fd_with_data(gamma, epsilon, N, u0, sigma_k, c0, _lambda, u_exp, PBC)
    if LOSS_TYPE == "HuberLoss":
        E_GRAD, E_DW, E_FM, E_DATA = energy_value_fd_with_data_huber_loss(gamma, epsilon, N, u0, sigma_k, c0, _lambda, u_exp, PBC = PBC)
    if LOSS_TYPE == "MSE+k_peak":
        E_GRAD, E_DW, E_FM, E_DATA = energy_value_fd_with_data_and_kpeak(gamma, epsilon, N, u0, sigma_k, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC)
        

    E0 = E_GRAD + E_DW + E_FM + E_DATA
    energies = [E0]
    energies_grad = [E_GRAD]
    energies_dw = [E_DW]
    energies_fm = [E_FM]
    energies_data = [E_DATA]


    if LIVE_PLOT:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,8))

    try:
        for ii in tqdm(range(1, num_iters+1), desc="Nesterov GD for Data"):

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
                E_GRAD, E_DW, E_FM, E_DATA = energy_value_fd_with_data(gamma, epsilon, N, u_curr, sigma_k, c0, _lambda, u_exp, PBC)
            if LOSS_TYPE == "HuberLoss":
                E_GRAD, E_DW, E_FM, E_DATA = energy_value_fd_with_data_huber_loss(gamma, epsilon, N, u_curr, sigma_k, c0, _lambda, u_exp, PBC = PBC)
            if LOSS_TYPE == "MSE+k_peak":    
                E_GRAD, E_DW, E_FM, E_DATA = energy_value_fd_with_data_and_kpeak(gamma, epsilon, N, u_curr, sigma_k, c0, _lambda, u_exp, k_peak_sim, k_peak_exp, PBC)

            E_TOTAL = E_GRAD + E_DW + E_FM + E_DATA
            energy_diff = energies[-1] - E_TOTAL
            
            energies.append(E_TOTAL)
            energies_grad.append(E_GRAD)
            energies_dw.append(E_DW)
            energies_fm.append(E_FM)
            energies_data.append(E_DATA)

            if (ii % 100) == 0 and LIVE_PLOT:
                plotting_schematic_eval(OUTPUT_PATH, fig, ax1, ax2, u_curr, energies, N, num_iters, gamma, epsilon, _lambda, ii, DATA_LOG)
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
        "E_grad": energies_grad,
        "E_dw": energies_dw,
        "E_fm": energies_fm,
        "E_data": energies_data,
    }

    if DATA_LOG:
        log_data_history(OUTPUT_PATH, u_curr, history, N, num_iters, gamma, epsilon, _lambda)

    return u_curr, history 


    
