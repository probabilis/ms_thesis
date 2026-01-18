import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dataclasses import asdict, replace

from pattern_formation import define_spaces, fourier_multiplier, dtype_real, device, energy_value, energy_value_fd, grad_g, double_well_prime, grad_fd, energy_tensor, initialize_u0_random
from env_utils import PATHS,print_bars, get_args, plotting_style, plotting_schematic, log_data

from params import labyrinth_data_params, gd_sim_params, get_DataParameters, get_SimulationParamters

# ---------------------------------------------------------------  

def gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, c0, alpha, num_iters, LAPLACE_SPECTRAL = None, STOP_BY_TOL = True, ENERGY_STOP_TOL = 1e-12):

    if LAPLACE_SPECTRAL is None:
        LAPLACE_SPECTRAL = False

    x, k, modk, modk2 = define_spaces(gridsize, N)

    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)

    u = u0.clone()

    if LAPLACE_SPECTRAL:
        energies = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]
    else:
        energies = [energy_value_fd(u0, sigma_k, N, gamma, epsilon, c0)]

    if LIVE_PLOT or DATA_LOG:
        fig1, ax1 = plt.subplots(figsize = (14,12))
        fig2, ax2 = plt.subplots(figsize = (10,10))
        plt.ion()

    M_k = sigma_k + gamma * epsilon * modk2


    # ToDo: Calculate Lipshitz constant
    Ls = float(M_k.max().cpu().item())
    
    alpha = 2/Ls * 1e-3
    print("Lipschitz constant",Ls)
    print("alpha: ", alpha)

    for ii in tqdm(range(num_iters), desc="GD"):
        if LAPLACE_SPECTRAL:
            # local term 1 (laplcian term)
            # linear + nonlocal part (FM part + laplacian)
            grad_lin = grad_g(u, M_k)

            # nonlinear / local part (double well term)
            grad_double = (gamma / epsilon) * double_well_prime(u, c0)

            # total gradient
            grad_E = grad_lin + grad_double
        else:
            grad_E = grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0)

        # GD update
        u -= alpha * grad_E
        #print(u)

        if LAPLACE_SPECTRAL:
            curr_energy = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
        else:
            curr_energy = energy_value_fd(u, sigma_k, N, gamma, epsilon, c0)

        energy_diff = energies[-1] - curr_energy
        energies.append(curr_energy)

        if LIVE_PLOT and (ii % 5_000) == 0:
            plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, ii)
            plt.pause(1)  
            
        if STOP_BY_TOL and abs(energy_diff) < ENERGY_STOP_TOL:
            print("dE[ii-1,ii]", abs(energy_diff) )
            break

    plt.ioff()
    if DATA_LOG:
        u = u.real
        log_data(FOLDER_PATH, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, ii)

    return energies

# ---------------------------------------------------------------

def backtracking_autograd(u, energy_fn, alpha_init=1e-2, beta=0.5, c=1e-4, max_back=40, verbose=False):
    """
    Autograd-based backtracking line search that computes 
    
    grad = grad(energy_fn)(u)
    # implemented as: 
    # E_[i+1] <= (E_[i]- c * alpha * g_norm2)
    
    Returns: (u_new, E_new_float, alpha_used, grad_tensor)
    """
    # make a detached clone that requires grad
    u_var = u.clone().detach().requires_grad_(True)

    E_curr = energy_fn(u_var)
    # compute gradient via autograd
    E_curr.backward()
    grad = u_var.grad.detach().clone()

    g_norm2 = float(torch.sum(grad * grad).cpu().item().real)

    alpha = alpha_init
    E_curr_val = float(E_curr.detach().cpu().item())

    for i in range(max_back):
        u_try = (u - alpha * grad).detach()   # we step from original u, not u_var
        E_try = energy_fn(u_try)
        # if not finite -> reduce and continue
        if not torch.isfinite(E_try):
            if verbose: print(f" backtrack {i}: E_try not finite, alpha -> {alpha*beta:.2e}")
            alpha *= beta
            continue
        E_try_val = float(E_try.detach().cpu().item())

        # E_[i+1] <= (E_[i]- c * alpha * g_norm2)
        if E_try_val <= E_curr_val - c * alpha * g_norm2:
            if verbose: print(f" backtrack success at {i} alpha={alpha:.2e} E_curr={E_curr_val:.6e} E_new={E_try_val:.6e}")
            return u_try, E_try_val, alpha, grad
        # reduce alpha
        if verbose and i < 4:
            print(f" backtrack {i}: alpha={alpha:.2e} E_try={E_try_val:.6e} need <= {E_curr_val - c*alpha*g_norm2:.6e}")
        alpha *= beta

    if verbose:
        print(" backtracking failed; returning original u")
    return u.clone().detach(), E_curr_val, alpha, grad

# ---------------------------------------------------------------

def gradient_descent_backtracking(u, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, c0, num_iters):
    # direct calculation of the GD method via auto-grad method via PyTorch

    x, k, modk, modk2 = define_spaces(gridsize, N)
    sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)
    M_k = sigma_k + gamma * epsilon * modk2  # (S + γ ε |k|^2)
    
    Ls = float(M_k.max().cpu().item())
    alpha = 1e-5 / Ls   # conservative
    print("Initial alpha: ", alpha)

    energies = []


    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    plt.ion()

    try:
        # -- Gradient descent looop --
        for n in tqdm(range(num_iters)):
            u_new, E_new, alpha_used, grad = backtracking_autograd(
                u, 
                lambda v: energy_tensor(v, gamma, epsilon, N, th, modk, modk2, c0, sigma_k),
                alpha_init=1e-3,
                beta=0.5, c=1e-4, max_back=40, verbose=(n%1000==0)
            )
            u = u_new
            E = E_new
            energies.append(E)

            if LIVE_PLOT and (n % 1_000) == 0:
                plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)
                plt.pause(1)

    except KeyboardInterrupt:
        print("Exit.")  
    
    plt.ioff()
    print(energies)
    if DATA_LOG:
        log_data(FOLDER_PATH, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)
        plt.pause(1)

# ---------------------------------------------------------------

if __name__ == "__main__":
    
    plotting_style()
    FOLDER_PATH = PATHS.PATH_GD
    
    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log
    labyrinth_data_params = replace(labyrinth_data_params, N = 40)
    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    
    u = initialize_u0_random(N, REAL=True)
    #print(u)
    #print(u.shape)
    import time

    print_bars()
    print(labyrinth_data_params)
    print(gd_sim_params)
    print_bars()
    #time.sleep(100)
    #gradient_descent_backtracking(u, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params), num_iters=500_000, c0 = 9/32)
    gradient_descent(u, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params), **asdict(gd_sim_params))

    
