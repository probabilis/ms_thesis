import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pattern_formation import fourier_multiplier,energy_value, dtype_real, device
from params import labyrinth_data_params, sim_params, get_DataParameters, get_SimulationParamters, sin_data_params
from env_utils import get_args, plotting_style

# ---------------------------------------------------------------

plotting_style()
folder_path = r"out/pgd/"

# ---------------------------------------------------------------
# --- parameters ---

L = 1.0
N = 512               
epsilon = 1/20
gamma = 1/200
c0 = 9/32

tau = 5e-3              # proximal gradient step size
num_iters = 100_000     # total iterations
prox_newton_iters = 20  # iterations for prox Newton
tol_newton = 1e-8       # stop tol inside prox


# ---------------------------------------------------------------
k = torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device=device),
               torch.arange(-N // 2, 0, dtype=dtype_real, device=device)])

xi, eta = torch.meshgrid(k, k, indexing='ij')
modk2 = (xi ** 2 + eta ** 2).to(dtype_real)
modk = torch.sqrt(modk2)
th = 1.0

sigma_k = fourier_multiplier(th * modk).to(dtype_real).to(device)

M_k = sigma_k + gamma * epsilon * modk2 # M_k multiplier for the quadratic term

def initialize_u0_random(N):
    amplitude = 0.1
    u0 = amplitude * (2 * torch.rand(N, N, dtype=dtype_real, device=device) - 1) 
    #+ amplitude * 1j * (2*torch.rand(N, N, dtype=dtype_real, device=device) - 1)
    return u0

u = initialize_u0_random(N)

def fft2_real(x):
    return torch.fft.fft2(x)

def ifft2_real(x_hat):
    return torch.fft.ifft2(x_hat).real


def grad_g(u):
    # gradient of g(u) via spectral multiplication
    Fu = fft2_real(u)          
    grad_hat = M_k * Fu
    grad_real = ifft2_real(grad_hat)
    return grad_real


def prox_h(v, tau, gamma=gamma, eps=epsilon, c0=c0,maxiter=prox_newton_iters, tol=tol_newton):
    # --- proximal operator for h(x) = (gamma/epsilon) * c0 * (1 - x^2)^2 ---
    # via vectorized Newton method, returns prox evaluated elementwise
    # Ref.: https://stackoverflow.com/questions/30191851/vectorize-a-newton-method-in-python-numpy
    # minimize 0.5*(x-v)^2 + tau*(gamma/eps)*c0*(1-x^2)^2

    lam = tau * (gamma / eps) * c0
    x = v.clone()

    for i in range(maxiter):

        grad = x - v - 4.0 * lam * x * (1.0 - x * x)
        hess = 1.0 + lam * (4.0 * (x * x - 1) + 8.0 * x * x)
        hess_safe = torch.where(torch.abs(hess) < 1e-12, torch.sign(hess) * 1e-12, hess)
        step = grad / hess_safe # ratio for newtons method

        # damped update (clamp step to avoid runaway)
        # use backtracking-like damping factor to ensure phi decreases (simple safeguard)
        # Ref.: claude.ai + stackoverflow
        alpha = 1.0
        x_new = x - alpha * step

        max_jump = 0.5
        delta = x_new - x
        overshoot = torch.abs(delta) > max_jump
        if overshoot.any():
            # scale down the step where overshooting
            scale = max_jump / (torch.abs(delta) + 1e-16)
            x_new = x + delta * torch.where(overshoot, scale, torch.ones_like(scale))

        # check convergence (max abs difference)
        if torch.max(torch.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    return x

if __name__ == "__main__":

    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log

    # --- main proximal-gradient loop ---
    energies = []
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
    fig2, ax2 = plt.subplots(1,1, figsize=(5,5))
    plt.ion()

    try:
        for n in tqdm(range(num_iters)):

            # forward step (gradient of smooth part)
            ggrad = grad_g(u)     
            v = u - tau * ggrad

            # backward/prox step: solve pointwise prox
            u = prox_h(v, tau)


            # u = torch.clamp(u, -10.0, 10.0) # keep u within reasonable bounds to avoid blowup
            try:
                E = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
            except Exception as e:
                E = None
            energies.append(E)

            if LIVE_PLOT and (n % 1000) == 0:
                ax1.clear()
                ax2.clear()
                
                ax1.imshow(u.cpu().numpy(), cmap='gray', extent=(0,1,0,1))
                ax1.set_title(f"Iteration {n}")

                ax2.plot(torch.arange(0,len(energies), 1), energies)
                ax2.set_title("energy evolution")
                plt.pause(1e-1)

        plt.ioff()

    except KeyboardInterrupt:
        print("Exit.")  


    if DATA_LOG:
        ax1.imshow(u.real, cmap='gray',extent=(0, 1, 0, 1))
        ax1.set_title(f"Iteration {n}")
        fig1.savefig(folder_path + f"image_graddescent_N={N}_nmax={num_iters}_alpha={tol_newton}_gamma={gamma}_eps={epsilon}.png")

        ax2.plot(torch.arange(0,len(energies), 1), energies)
        #ax2.set_yscale('log')

        ax2.set_title("energy evolution")
        fig2.savefig(folder_path + f"energy_graddescent_N={N}_nmax={num_iters}_alpha={tol_newton}_gamma={gamma}_eps={epsilon}.png")
