import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dataclasses import asdict

from pattern_formation import *
from env_utils import PATHS,print_bars, get_args, plotting_style, plotting_schematic, log_data

from params import labyrinth_data_params, gd_sim_params, get_DataParameters, get_SimulationParamters


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

def gradient_descent_backtracking(u, LIVE_PLOT, DATA_LOG, gridsize, N, th, gamma, epsilon, c0, num_iters):

    x, k, modk, modk2 = define_spaces(gridsize, N)
    sigma_k = fourier_multiplier(modk)
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

            if LIVE_PLOT and (n % 10_000) == 0:
                plotting_schematic(folder_path, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)
                plt.pause(1)

    except KeyboardInterrupt:
        print("Exit.")  
    
    plt.ioff()
    
    if DATA_LOG:
        log_data(folder_path, u, energies, N, num_iters, gamma, epsilon)
        plotting_schematic(folder_path, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, n)
        plt.pause(1)

# ---------------------------------------------------------------

if __name__ == "__main__":
    
    plotting_style()
    folder_path = PATHS.PATH_GD
    
    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u = initialize_u0_random(N)
    
    print_bars()
    print(labyrinth_data_params)
    print(gd_sim_params)
    print_bars()
    
    gradient_descent_backtracking(u, LIVE_PLOT, DATA_LOG, **asdict(labyrinth_data_params), **asdict(gd_sim_params))


    
