import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from utils.env_utils import PATHS, read_sim_dat_from_csv, plotting_style, main_colormap
from exp_data_processing.read import read_csv
from params.opt_params import labyrinth_data_params, exp_data_params, get_DataParameters
from exp_data_processing.postprocessing import quality_scores

from spectrum_analysis import radial_wavelength_spectrum
from exchange_length import reduced_exp_image_width




if __name__ == "__main__":


    num_iters = 5_000
    dataset = "data_01"
    
    file_indices = [1, 2, 3, 4, 5, 6, 7]
    arrangement = [3, 2, 1, 5, 4, None, 6]
    FILE_AMOUNT = 6
    
    domain_pattern_lengths = np.zeros(FILE_AMOUNT)

    INPUT_PATH = PATHS.BASE_EXPDATA

    # ---------------------------------------------------------------

    self_picked_indices = [14, 11, 11, 7, 4, 1]

    fig, axs = plt.subplots( 5, 6, figsize = (16,10)) 



    for ii, OPT_FILE in enumerate( file_indices ):
        
        recording = f"00{OPT_FILE}"

        INPUT_FILE_PATH = INPUT_PATH / f"{dataset}/csv/mcd_slice_{recording}.csv"
        OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording

        df = pd.read_csv(OUTPUT_PATH / f"ranking.csv", index_col = 0)

        cut = df["perimeter"].quantile(0.6)
        df_filt = df[df["perimeter"] <= cut].copy()
        
        best = df_filt.sort_values(["fisherJ", "perimeter"], ascending = [False, True])


        _gamma = best.iloc[0]["gamma"]
        _lambda = best.iloc[0]["lambda"]

        if ii == 6:
            _gamma = df.iloc[1]["gamma"]
            _lambda = df.iloc[1]["lambda"]     
    
        u_exp = read_csv(INPUT_FILE_PATH, PLOT = False)
        df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, exp_data_params.N, num_iters, _gamma, exp_data_params.epsilon, _lambda)    

        if arrangement[ii] is not None:

            kk = arrangement[ii] - 1
            self_picked_new_index = self_picked_indices[kk]
            self_picked_gamma = df.iloc[self_picked_new_index]["gamma"]
            self_picked_lambda = df.iloc[self_picked_new_index]["lambda"]
            self_df_energies, self_u_sim = read_sim_dat_from_csv(OUTPUT_PATH, exp_data_params.N, num_iters, self_picked_gamma, exp_data_params.epsilon, self_picked_lambda)    

            new_index = arrangement[ii] - 1

            axs[0, new_index].imshow(u_exp, cmap = main_colormap, origin="lower", extent=(0,1,0,1))


            u_sim = torch.tensor(u_sim.values)
            axs[1, new_index].set_title(f"$\\gamma$ = {_gamma} \n $\\lambda$ = {_lambda}")
            axs[1, new_index].imshow(u_sim, cmap = main_colormap, origin="lower", extent=(0,1,0,1))
            
            
            self_u_sim = torch.tensor(self_u_sim.values)
            axs[2, new_index].imshow(self_u_sim, cmap = main_colormap, origin="lower", extent=(0,1,0,1))
            axs[2, new_index].set_title(f"$\\gamma$ = {self_picked_gamma} \n $\\lambda$ = {self_picked_lambda}")


            fisherJ, perimeter, W = quality_scores(u_exp, self_u_sim)
            W = torch.where(torch.abs(self_u_sim) > 0.15, 0, self_u_sim)
            axs[3, new_index].imshow(W.float(), cmap = main_colormap, origin="lower", extent=(0,1,0,1) )
            axs[3, new_index].set_title(f"$S(u)$ = {fisherJ:.4g} \n $P(u)$ = {perimeter:.5g}")


            results = radial_wavelength_spectrum(self_u_sim, exp_data_params.gridsize/exp_data_params.N, plot = False)
            axs[4, new_index].loglog(results["k"].cpu(), results["profile"].cpu(), lw=2, color = "blue", label = "opt.")
            axs[4, new_index].axvline(results["k_peak"], linestyle="--", label=f"$k^*$ = {results["k_peak"]:.4g}", color = "blue")
            

            results_raw = radial_wavelength_spectrum(u_exp, exp_data_params.gridsize/exp_data_params.N, plot = False)
            axs[4, new_index].loglog(results_raw["k"].cpu(), results_raw["profile"].cpu(), lw=2, color = "red", label = "raw")
            axs[4, new_index].axvline(results_raw["k_peak"], linestyle="--", label=f"$k^*$ = {results_raw["k_peak"]:.4g}", color = "red")
            domain_pattern_length = results["wavelength_peak"]/2 * reduced_exp_image_width * 1e6
            axs[4, new_index].set_title(f"$D_p = $ {domain_pattern_length:.4g}")
            axs[4, new_index].legend()
        


            for jj in range(3):
                axs[jj, new_index].set_box_aspect(1)
                axs[jj, new_index].axes.get_xaxis().set_ticks([])
                axs[jj, new_index].axes.get_yaxis().set_ticks([])


            domain_pattern_lengths[new_index] = domain_pattern_length
            kk += 1

    print(domain_pattern_lengths)


    fig.tight_layout()
    plt.show()


    
    