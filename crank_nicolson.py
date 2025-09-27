from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pandas as pd
from dataclasses import asdict
from scipy.ndimage import gaussian_filter

from pattern_formation import *
from params import labyrinth_data_params, sim_params, get_DataParameters, get_SimulationParamters, sin_data_params
from env_utils import PATHS, get_args, plotting_style


# ---------------------------------------------------------------

def adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0):
        # Adapted Crank-Nicolson Schematic (Reference Condette Paper)
    x, k, modk, modk2 = define_spaces(gridsize, N)
    
    if LIVE_PLOT:
        plt.imshow(torch.real(u0).real, cmap='gray', extent=(0, 1, 0, 1), interpolation='none')
        plt.show()
        time.sleep(1)
        plt.close()
    
    L = gamma * epsilon * modk2 + fourier_multiplier(th * modk) # (2 * np.pi)**2 
    L[0, 0] = fourier_multiplier(torch.tensor(0.0))


    time_vector = [0.0]
    energy_list = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]

    u_n = u0

    energy_diff = 1000
    ii = 0

    plt.ion()
    fig1, ax1 = plt.subplots(figsize = (14,12))
    fig2, ax2 = plt.subplots(figsize = (10,10))

    fp_iterations = []

    pbar = tqdm(total=max_it)

    try:
        while ii < max_it and energy_diff > stop_limit:
            ii_fp, u_np1, err, conv = fixpoint(u_n, L, dt, N, epsilon, gamma, max_it_fixpoint, tol, c0)
            fp_iterations.append(ii_fp)
            #print("Energy diff: ", energy_diff)
            if conv:
                curr_energy = energy_value(gamma, epsilon, N, u_np1, th, modk, modk2, c0)
                u_diff = torch.max(torch.abs(u_np1 - u_n)).item()
                energy_diff = energy_list[-1] - curr_energy
                energy_list.append(curr_energy)
                _time = time_vector[-1] + dt
                time_vector.append(_time)
                #print("time: ", _time)

                

                u_n = u_np1
                ii += 1
                
                if LIVE_PLOT and ii % 100 == 0:
                    ax1.clear()
                    ax2.clear()
                    #ax.imshow(torch.abs(u).cpu().numpy(), cmap='gray', extent=(0, 1, 0, 1))
                    ax1.imshow(torch.real(u_n).real, cmap='gray', extent=(0, 1, 0, 1), interpolation='none')

                    ax1.set_title(f"Time = {_time:.8f}")
                    ax2.plot(time_vector, energy_list)


                    plt.pause(0.5)
                
            else:
                dt = dt / 4
                if dt < 1e-12:
                    print("exit.")
                    raise RuntimeError("Time step too small. Exiting.")
                

            pbar.update(1)

    except KeyboardInterrupt:
        print("Exit.")


    pbar.close()

    # -----------------------------

    print("Final energy difference: ", energy_diff)
    time.sleep(1)

    ax1.clear()
    ax2.clear()

    ax1.set_title(f"Pattern evolution after time = {_time:.2f}")
    ax1.imshow(u_n.cpu().numpy(), cmap='gray', extent=(0, 1, 0, 1), interpolation='none')
    
    ax2.plot(time_vector, energy_list)
    ax2.grid(color="gray")
    ax2.set_title("energy evolution")
    #ax2.set_yscale('log')
    ax2.set_xlabel("time / 1")
    ax2.set_ylabel("energy / 1")

    plt.ioff()

    if DATA_LOG:
        file_params = f"N={N}_gamma={gamma}_eps={epsilon}_dt={dt}_th={th}"
        df_energies = pd.DataFrame(energy_list)
        u_n_np = u_n.numpy()
        df_u_n = pd.DataFrame(u_n_np)
        
        df_energies.to_csv(folder_path + f"energy_{file_params}", index = False, header = False)
        df_u_n.to_csv(folder_path + f"image_{file_params}", index = False, header = False)

        fig1.savefig(folder_path + f"image_{file_params}" + '.png')
        fig2.savefig(folder_path + f"energy_{file_params}" + '.png')
    
    
    plt.show()

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    folder_path = PATHS.PATH_CN

    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log


    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u0 = initialize_u0_random(N)
    #u0 = initialize_u0_sin(N, x)
    #u0 = initialize_u0_image('input_test.png')

    adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, **asdict(labyrinth_data_params), **asdict(sim_params))

    