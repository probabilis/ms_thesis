import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from pattern_formation import initialize_u0_random

from params import labyrinth_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params

from env_utils import PATHS, print_bars, plotting_style,log_data

from gd_nesterov import gradient_descent_nesterov
from params import sim_config



if __name__ == "__main__":
    
    plotting_style()
    FOLDER_PATH = PATHS.PATH_PARAMS_STUDY

    LIVE_PLOT = False
    DATA_LOG = False

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    N = 64

    ngd_sim_params = replace(ngd_sim_params, num_iters = 5_000)
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print(sim_config)
    print_bars()

    PLOT = False
    gamma_ls = np.linspace(0.001, 0.00002, 20)
    N_est = 20
    frequencies = torch.zeros(N_est, len(gamma_ls))

    if PLOT:
        fig, axs = plt.subplots(2,2)

    for ii in range(N_est):
        values = []
        for gamma in gamma_ls:

            u0 = initialize_u0_random(N, REAL = True)

            labyrinth_data_params = replace(labyrinth_data_params, N = 64, gamma = gamma)
            
            u, energies = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), **asdict(sim_config))
            
            u_hat = torch.fft.fft2(u, norm='ortho')
            xf = torch.fft.fftfreq(N, d = gridsize / N)
            yf = torch.diagonal(u_hat)
            yf = torch.abs(yf)
            
            index = torch.argmax(yf[0:N//2])
            
            if PLOT:
                axs[0,0].clear()

                axs[0, 0].set_title("$u_{opt}$")
                axs[0, 0].imshow(u)
                axs[0, 1].set_title("$FFT(u_{opt})$")
                axs[0, 1].imshow(torch.abs(u_hat)[0:N//2, 0:N//2])
                axs[1, 0].set_title("Energy")
                axs[1, 0].plot(np.arange(0, len(energies), 1), energies)
                
                axs[1, 1].set_title("$FFT(u_{opt})$ spectrum")
                axs[1, 1].plot(xf[0:N//2], 2.0/N * torch.abs(yf[0:N//2]))
                axs[1, 1].vlines(float(xf[index]), 0, 2.0/N * torch.max(yf[0:N//2]), label = f"$\\omega = ${float(xf[index]):.3f}", color = "red")
                axs[1, 1].legend()
                
                #ax2.plot(np.arange(0,len(values),1),values)

                #yf_mean = (yf_prev + yf) / 2
                #axs[1, 1].plot(xf[0:N//2], 2.0/N * torch.abs(yf_mean[0:N//2]),linestyle = "--", color = "gray")
                
                plt.pause(1)
            print(f"Pattern frequency: {xf[index]}")
            values.append(float(xf[index]))
        
        frequencies[ii] = torch.tensor(values)

    print(frequencies)
    mean_frequencies = torch.mean(frequencies, dim = 0)
    print(mean_frequencies)
    SPECTRUM = True
    if SPECTRUM:
        plt.figure()
        plt.title("Characteristic Fourier Frequency of spectrum")
        plt.xlabel("Gamma $\\gamma$")
        plt.ylabel("Fourier frequency $\\omega$ / 1")
        plt.plot(gamma_ls, values)
        plt.grid(color = "gray")
        plt.savefig(FOLDER_PATH / "fourier_frequencies.png", dpi = 300)     
        plt.show()