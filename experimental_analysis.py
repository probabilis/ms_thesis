from env_utils import read_sim_dat_from_csv, plotting_style
from params import labyrinth_data_params, exp_data_params, get_DataParameters
from dataclasses import replace
from env_utils import PATHS
import numpy as np
from read import read_csv
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd

from spectrum_analysis import radial_wavelength_spectrum


if __name__ == "__main__":


    num_iters = 5_000

    dataset = "data_01"
    


    file_indices = [1, 2, 3, 4, 5, 6, 7]
    arrangement = [3, 2, 1, 5, 4, None, 6]
    FILE_AMOUNT = 6
    


    INPUT_PATH = PATHS.BASE_EXPDATA

    # ---------------------------------------------------------------


    fig, axs = plt.subplots( 3,6, figsize = (16,10)) 


    for ii, OPT_FILE in enumerate( file_indices ):
        
        recording = f"00{OPT_FILE}"

        INPUT_FILE_PATH = INPUT_PATH / f"{dataset}/csv/mcd_slice_{recording}.csv"
        OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording

        df = pd.read_csv(OUTPUT_PATH / f"ranking.csv", index_col = 0)

        cut = df["perimeter"].quantile(0.6)
        df_filt = df[df["perimeter"] <= cut].copy()
        #print("Best parameter constellation: ")
        best = df_filt.sort_values(["fisherJ", "perimeter"], ascending = [False, True])
        

        _gamma = best.iloc[0]["gamma"]
        _lambda = best.iloc[0]["lambda"]

        if ii == 6:
            _gamma = df.iloc[1]["gamma"]
            _lambda = df.iloc[1]["lambda"]
    
        u_exp = read_csv(INPUT_FILE_PATH, PLOT = False)
        df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, exp_data_params.N, num_iters, _gamma, exp_data_params.epsilon, _lambda)    

        if arrangement[ii] is not None:
            new_index = arrangement[ii] - 1

            axs[0, new_index].set_title(f"$\\gamma$ = {_gamma} \n $\\lambda$ = {_lambda}")
            
            axs[0, new_index].imshow(u_exp)
            axs[1, new_index].imshow(u_sim)


            results = radial_wavelength_spectrum(torch.tensor(u_sim.values), exp_data_params.gridsize/exp_data_params.N, plot = False)


            axs[2, new_index].loglog(results["k"].cpu(), results["profile"].cpu(), lw=2, color = "blue", label = "opt.")
            axs[2, new_index].axvline(results["k_peak"], linestyle="--", label=f"$k^*$ = {results["k_peak"]:.4g}", color = "blue")
            

            results_raw = radial_wavelength_spectrum(u_exp, exp_data_params.gridsize/exp_data_params.N, plot = False)
            
            axs[2, new_index].loglog(results_raw["k"].cpu(), results_raw["profile"].cpu(), lw=2, color = "red", label = "raw")
            axs[2, new_index].axvline(results_raw["k_peak"], linestyle="--", label=f"$k^*$ = {results_raw["k_peak"]:.4g}", color = "red")

            #y = results["profile"].cpu() - results_raw["profile"].cpu()
            #axs[2, new_index].loglog(results_raw["k"], torch.abs(y), color = "black")
            axs[2, new_index].set_title(f"$D_p = $ {results["wavelength_peak"]:.4g}")


            axs[2, new_index].legend()

            for jj in range(2):
                axs[jj, new_index].set_box_aspect(1)
                axs[jj, new_index].axes.get_xaxis().set_ticks([])
                axs[jj, new_index].axes.get_yaxis().set_ticks([])



    fig.tight_layout()
    plt.show()
        
    