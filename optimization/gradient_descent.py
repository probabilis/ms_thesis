import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.pattern_formation import define_spaces, fourier_multiplier, dtype_real, device, energy_value, energy_value_fd, grad_g, double_well_prime, grad_fd
from utils.env_utils import plotting_schematic, log_data


def gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, c0, alpha, num_iters, LAPLACE_SPECTRAL = False, STOP_BY_TOL = True, ENERGY_STOP_TOL = 1e-12, PBC = True, SAVE_U_HISTORY = None):
    """
    Proximal Gradient Descent optimization algorithm of energy functional 
    """
    x, k, modk, modk2 = define_spaces(gridsize, N)

    print("LaPlace Spectral Calculation: ", LAPLACE_SPECTRAL)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)

    M_k = sigma_k + gamma * epsilon * modk2 * (2*torch.pi)**2  # M_k for spectral calculation of LAPLACE

    u = u0.clone()

    if LAPLACE_SPECTRAL:
        E_GRAD, E_DW, E_FM = energy_value(gamma, epsilon, N, u0, c0, sigma_k, modk2, RETURN_SEPERATE=True)
        E0 = E_GRAD + E_DW + E_FM
    else:
        print("PBC: ", PBC)
        E_GRAD, E_DW, E_FM = energy_value_fd(u0, sigma_k, N, gamma, epsilon, c0, PBC, RETURN_SEPERATE=True)
        E0 = E_GRAD + E_DW + E_FM


    energies = [E0]
    energies_grad = [E_GRAD]
    energies_dw = [E_DW]
    energies_fm = [E_FM]


    if LIVE_PLOT or DATA_LOG:
        fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10,8))
        plt.ion()

    if SAVE_U_HISTORY is not None:
        u_ls = [u0] # for time evolution multiple fields are saved (only in this method)

    try:
        for ii in tqdm(range(num_iters), desc="GD"):
            if LAPLACE_SPECTRAL:
                # linear + nonlocal part (FM part + laplacian)
                grad_lin = grad_g(u, M_k)

                # nonlinear / local part (double well term)
                grad_double = (gamma / epsilon) * double_well_prime(u, c0)

                # total gradient
                grad_E = grad_lin + grad_double
            else:
                grad_E = grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC, DW_TERM=True)

            
            u -= alpha * grad_E     # GD update

            if SAVE_U_HISTORY is not None and (ii % SAVE_U_HISTORY) == 0:
                u_ls.append(u.clone())


            if LAPLACE_SPECTRAL:
                E_GRAD, E_DW, E_FM = energy_value(gamma, epsilon, N, u0, c0, sigma_k, modk2, RETURN_SEPERATE=True)
                E_TOTAL = E_GRAD + E_DW + E_FM
            else:
                E_GRAD, E_DW, E_FM = energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC, RETURN_SEPERATE=True)
                E_TOTAL = E_GRAD + E_DW + E_FM


            energy_diff = energies[-1] - E_TOTAL
            energies.append(E_TOTAL)
            energies_grad.append(E_GRAD)
            energies_dw.append(E_DW)
            energies_fm.append(E_FM)


            if LIVE_PLOT and (ii % 100) == 0:
                plotting_schematic(FOLDER_PATH, fig, ax1, ax2, u, energies, N, num_iters, gamma, epsilon, ii, DATA_LOG)
                plt.pause(1)  

            if STOP_BY_TOL and abs(energy_diff) < ENERGY_STOP_TOL:
                print("dE[ii-1,ii]", energy_diff )
                break
                
    except KeyboardInterrupt:
        print("Exit.")  

    print("dE[ii-1,ii]", energy_diff )

    plt.ioff()

    if DATA_LOG:
        log_data(FOLDER_PATH, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, fig, ax1, ax2, u, energies, N, num_iters, gamma, epsilon, ii, DATA_LOG)

    history = {
        "E_total": energies,
        "E_grad": energies_grad,
        "E_dw": energies_dw,
        "E_fm": energies_fm,
    }

    if SAVE_U_HISTORY is not None:
        return u_ls, history
    else:
        return u, energies


    
