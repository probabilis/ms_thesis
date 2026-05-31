import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from utils.pattern_formation import define_spaces, fourier_multiplier, dtype_real, device, energy_value, energy_value_fd, grad_g, double_well_prime, grad_fd, energy_tensor, initialize_u0_random
from utils.env_utils import plotting_schematic, log_data
from params.opt_params import labyrinth_data_params


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
        #if not finite -> reduce and continue
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


    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,8))
    plt.ion()

    try:
        # -- Gradient descent looop --
        for ii in tqdm(range(num_iters)):
            u_new, E_new, alpha_used, grad = backtracking_autograd(
                u, 
                lambda v: energy_tensor(v, gamma, epsilon, N, th, modk, modk2, c0, sigma_k),
                alpha_init=1e-3,
                beta=0.5, c=1e-4, max_back=40, verbose=(ii%1000==0)
            )
            u = u_new
            E = E_new
            energies.append(E)

            if LIVE_PLOT and (ii % 100) == 0:
                plotting_schematic(FOLDER_PATH, fig, ax1, ax2, u, energies, N, num_iters, gamma, epsilon, ii, DATA_LOG)
                plt.pause(1)

    except KeyboardInterrupt:
        print("Exit.")  
    
    plt.ioff()
    print(energies)
    if DATA_LOG:
        log_data(FOLDER_PATH, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, fig, ax1, ax2, u, energies, N, num_iters, gamma, epsilon, ii, DATA_LOG)
        plt.pause(1)


