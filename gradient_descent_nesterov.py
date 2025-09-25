import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pattern_formation import fourier_multiplier,energy_value, dtype_real, device
from params import labyrinth_data_params, sim_params, get_DataParameters, get_SimulationParamters, sin_data_params
from env_utils import get_args, plotting_style

from gradient_descent_proximal import *

# --- FISTA-like (Nesterov) proximal gradient with adaptive restart ---

plotting_style()
folder_path = r"out_nesterov/"

args = get_args()
LIVE_PLOT = args.live_plot
DATA_LOG = args.data_log

# Initialize momentum variables
u_prev = u.clone()           # u^{k-1}
t_prev = 1.0                 # t_{k-1}
u_curr = u.clone()           # u^{k}

energies = []
fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
fig2, ax2 = plt.subplots(1,1, figsize=(5,5))
plt.ion()

# parameters for restart and diagnostics
USE_RESTART = True
PRINT_EVERY = 1000
MAX_BAD_RESTARTS = 0  # optional limit on restarts (0 means unlimited)
bad_restarts = 0

try:
    for n in tqdm(range(num_iters)):

        # --- 1) extrapolation (Nesterov momentum) ---
        t_curr = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev) ** 0.5)   # t_k
        beta = (t_prev - 1.0) / t_curr                               # momentum factor

        # y = u_curr + beta * (u_curr - u_prev)
        y = u_curr + beta * (u_curr - u_prev)

        # --- 2) forward step at extrapolated point ---
        ggrad = grad_g(y)                    # gradient of g at y
        v = y - tau * ggrad

        # --- 3) prox (backward) step ---
        u_next = prox_h(v, tau)

        # compute energy to possibly restart
        try:
            E_next = energy_value(gamma, epsilon, N, u_next, th, modk, modk2, c0)
        except Exception:
            E_next = None

        # optional: compute energy of current iterate to compare (u_curr)
        if n == 0:
            try:
                E_curr_val = energy_value(gamma, epsilon, N, u_curr, th, modk, modk2, c0)
            except Exception:
                E_curr_val = None
        else:
            E_curr_val = energies[-1]

        # --- 4) adaptive restart: if energy increased -> reset momentum ---
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
                v_plain = u_curr - tau * grad_g(u_curr)
                u_next = prox_h(v_plain, tau)
                try:
                    E_next = energy_value(gamma, epsilon, N, u_next, th, modk, modk2, c0)
                except Exception:
                    E_next = None

        # --- 5) update momentum history for next iteration ---
        u_prev = u_curr
        u_curr = u_next
        t_prev = t_curr

        # logging
        energies.append(E_next)
        if (n % PRINT_EVERY) == 0:
            msg = f"iter {n:6d}: E={E_next:.6e} alpha={tau:.2e}"
            if restarted:
                msg += " (RESTART)"
            print(msg)

        # live plotting (occasional)
        if (n % 1000) == 0 and LIVE_PLOT:
            ax1.clear()
            ax2.clear()
            ax1.imshow(u_curr.cpu().numpy(), cmap='gray', extent=(0,1,0,1))
            ax1.set_title(f"Iteration {n}")
            fig1.savefig(folder_path + f"image_graddescent_nesterov_N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}.png")
            ax2.plot(torch.arange(0,len(energies)), energies)

            ax2.set_title("energy evolution")
            fig2.savefig(folder_path + f"energy_graddescent_nesterov_N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}.png")
            plt.pause(1)

    plt.ioff()

except KeyboardInterrupt:
    print("Exit.")