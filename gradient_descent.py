import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pattern_formation import fourier_multiplier, energy_value, dtype_complex, dtype_real, device
#from params import labyrinth_data_params, sim_params, get_DataParameters, get_SimulationParamters, sin_data_params
from main import initialize_u0_random
from env_utils import get_args

folder_path = r"out_gd/"

# -----------------------------------------------------


L = 1.0
N = 32

epsilon = 1/20
gamma = 1/200

c0 = 9/32

#alpha = 1e-8 # learning rate
num_iter = 2_000_000


#k = np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])
k = torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device), torch.arange(-N // 2, 0, dtype=dtype_real, device = device)])
xi, eta = torch.meshgrid(k, k, indexing='ij')
modk2 = (xi ** 2 + eta ** 2).to(dtype_real)
modk = torch.sqrt(modk2).to(dtype_real)
#modk = torch.sqrt(modk2)

# -----------------------------------------------------


def double_well_potential(u, c0):
    u2 = 1 - torch.abs(u) ** 2
    return c0 * (u2 ** 2)

def double_well_prime(u, c0):
    return -4.0 * c0 * u * (1.0 - u*u)


def laplacian(u, dx):
    # with periodic BC
    lap_u = (torch.roll(u, 1, dims=0) + torch.roll(u, -1, dims=0) +
             torch.roll(u, 1, dims=1) + torch.roll(u, -1, dims=1) - 4 * u) / dx**2
    return lap_u


def backtracking_autograd(u, energy_fn, alpha_init=1e-2, beta=0.5, c=1e-4, max_back=40, verbose=False):
    """
    Autograd-based backtracking line search that computes grad = grad(energy_fn)(u)
    Returns: (u_new, E_new_float, alpha_used, grad_tensor)
    """
    # make a detached clone that requires grad
    u_var = u.clone().detach().requires_grad_(True)

    E_curr = energy_fn(u_var)
    print("E_curr", E_curr)
    # compute gradient via autograd
    E_curr.backward()
    print("E_curr", E_curr)
    grad = u_var.grad.detach().clone()

    g_norm2 = float(torch.sum(grad * grad).cpu().item().real)

    alpha = alpha_init
    E_curr_val = float(E_curr.detach().cpu().item())

    for i in range(max_back):
        u_try = (u - alpha * grad).detach()   # note we step from original u, not u_var
        E_try = energy_fn(u_try)
        # if not finite, shrink and continue
        if not torch.isfinite(E_try):
            if verbose: print(f" backtrack {i}: E_try not finite, alpha -> {alpha*beta:.2e}")
            alpha *= beta
            continue
        E_try_val = float(E_try.detach().cpu().item())
        if E_try_val <= E_curr_val - c * alpha * g_norm2:
            if verbose: print(f" backtrack success at {i} alpha={alpha:.2e} E_curr={E_curr_val:.6e} E_new={E_try_val:.6e}")
            return u_try, E_try_val, alpha, grad
        # shrink alpha
        if verbose and i < 4:
            print(f" backtrack {i}: alpha={alpha:.2e} E_try={E_try_val:.6e} need <= {E_curr_val - c*alpha*g_norm2:.6e}")
        alpha *= beta

    # fail-safe: return original u (or last try)
    if verbose:
        print(" backtracking failed; returning original u")
    return u.clone().detach(), E_curr_val, alpha, grad


def energy_tensor(u, gamma, epsilon, N, th, modk, modk2, c0, sigma_k):
    # returns a torch scalar (not .item()), matching your energy_value implementation
    # NB: keep operations in torch (no .item())
    W = double_well_potential(u, c0)                    # tensor
    ftu = torch.fft.fft2(u) / (N ** 2)                   # match your energy_value style
    S = sigma_k
    e1 = (gamma / epsilon) * torch.sum(W) / (N ** 2)     # use same normalization as energy_value
    e2 = 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu) ** 2)
    return e1 + e2

# -----------------------------------------------------

if __name__ == "__main__":

    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log


    sigma_k = fourier_multiplier(modk)
    u = initialize_u0_random(N)

    M_k = sigma_k + gamma * epsilon * modk2  # (S + γ ε |k|^2)

    h = L / N
    th = 1

    energies = []

    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)

    plt.ion()

    LACPLACE_SPECTRAL = True

    Ls = float(M_k.max().cpu().item())
    alpha = 0.5 / Ls   # conservative
    print(alpha)
    alpha = 1e-6


    try:
        # -- Gradient descent looop --
        for n in tqdm(range(num_iter)):
            
            """
            # 1. Local gradient term
            lap_u = torch.zeros_like(u)
            # via spectral method
            if LACPLACE_SPECTRAL:
                lap_u = torch.fft.ifft2(-modk2 * torch.fft.fft2(u)).real
            # via finite difference
            else:
                lap_u = laplacian(u, h)
            
            grad_local = - epsilon * lap_u
            """

            """
            # spectral linear+nonlocal gradient
            Fu = torch.fft.fft2(u)
            #lin = torch.fft.ifft2(M_k * Fu).real
            lin = torch.fft.ifft2((sigma_k + gamma * epsilon * modk2) * (Fu)).real
            nl  = (gamma/epsilon) * double_well_prime(u, c0)

            grad_E = lin + nl
            u -= alpha * grad_E

            E = energy_value(gamma,epsilon,N,u,th,modk,modk2,c0)
            energies.append(E)
            """
            u_new, E_new, alpha_used, grad = backtracking_autograd(
                u, 
                lambda v: energy_tensor(v, gamma, epsilon, N, th, modk, modk2, c0, sigma_k),
                alpha_init=1e-3,
                beta=0.5, c=1e-4, max_back=40, verbose=(n%1000==0)
            )
            u = u_new
            E = E_new
            energies.append(E)


            #if (n % 100) == 0:
            #    print(f"Iter {n}, E={E:.6f}, alpha={alpha:.2e}")
            if LIVE_PLOT and (n % 10_000) == 0:
                ax1.clear()
                ax2.clear()
                #print(f"Iteration {n}")
                ax1.imshow(u.real, cmap='gray',extent=(0, 1, 0, 1))
                ax1.set_title(f"Iteration {n}")
                fig1.savefig(folder_path + f"image_graddescent_N={N}_nmax={num_iter}_alpha={alpha}_gamma={gamma}_eps={epsilon}.png")
                
                ax2.plot(torch.arange(0,len(energies), 1), energies)
                #ax2.set_yscale('log')

                ax2.set_title("energy evolution")
                fig2.savefig(folder_path + f"energy_graddescent_N={N}_nmax={num_iter}_alpha={alpha}_gamma={gamma}_eps={epsilon}.png")
                
                plt.pause(0.1)


    except KeyboardInterrupt:
        print("Exit.")  


    ax1.imshow(u.real, cmap='gray',extent=(0, 1, 0, 1))
    ax1.set_title(f"Iteration {n}")
    fig1.savefig(folder_path + f"image_graddescent_N={N}_nmax={num_iter}_alpha={alpha}_gamma={gamma}_eps={epsilon}.png")

    ax2.plot(torch.arange(0,len(energies), 1), energies)
    #ax2.set_yscale('log')

    ax2.set_title("energy evolution")
    fig2.savefig(folder_path + f"energy_graddescent_N={N}_nmax={num_iter}_alpha={alpha}_gamma={gamma}_eps={epsilon}.png")
