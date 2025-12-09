from env_utils import read_sim_dat_from_csv, plotting_style
from params import labyrinth_data_params, exp_data_params, get_DataParameters
from dataclasses import replace
from env_utils import PATHS
import numpy as np
from read import read_csv
import torch
import json
import matplotlib.pyplot as plt

from read import standardize_shift


if __name__ == "__main__":

    plotting_style()

    PLOT_HIST = False
    PLOT_DIFF_COMPARISON = True
    PLOT_ENERGY_CONVERGENCE_COMPARISON = False

    dataset = "data_00"
    recording = "003"
    ENERGY_DIFF_STOP_TOL = "0.1"

    read_types = ["raw", "standardize", "shift", "clipped"]

    INPUT_PATH = PATHS.BASE_EXPDATA

    colors = ['gray', 'gray', 'viridis']
    colors_hist = ['black', 'black', 'cornflowerblue']
    titles = ["raw recording $u_{exp}$", "optimized $u_{opt}$", "if $u_{opt}^{ij} < tol \\rightarrow \\alpha + u_{exp}$"]        # "RMSE $|u_{exp} - u_{opt}|^2$"

    # ---------------------------------------------------------------

    with open(INPUT_PATH / f"{dataset}" / "params_file.json", "r") as _file:
        params_file = json.load(_file)

    gamma_ls = params_file[recording]
    _lambda_ls = [0.06, 0.1, 0.2]
    print("Gamma's: ", gamma_ls)
    print("Lambda's: ", _lambda_ls)
    num_iters = 5000

    gridsize, N, th, epsilon, gamma = get_DataParameters(exp_data_params)


    def blue_ratio_blocks(u_exp, u_sim, tol=0.25, block_size=16):
        """
        Computes blue ratios inside domain walls for local blocks.
        Returns an array of ratios (one per block).
        """
        h, w = u_exp.shape
        ratios = []

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                ue = u_exp[i:i+block_size, j:j+block_size]
                us = u_sim[i:i+block_size, j:j+block_size]

                domain_mask = np.abs(us) <= tol
                if domain_mask.sum() == 0:
                    continue

                blue = (ue > 0) & domain_mask
                ratios.append(blue.sum() / domain_mask.sum())

        return np.array(ratios)

    def u_overlap(u_sim, u_exp, tol = 0.25):                    
        #diff = torch.abs(u_exp - u_sim) ** 2 * 0.5
        #diff = standardize_shift(diff)
        #new = np.where(np.abs(u_sim) > tol, 0, np.nan)
        new = torch.where(torch.abs(u_sim) < tol, 10, u_sim)
        #new = torch.from_numpy(new.astype(np.float32))
        return u_exp + new

    for read_type in read_types:
        print("Reading type: ", read_type)
        OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording / ENERGY_DIFF_STOP_TOL / read_type
        u_exp = read_csv(INPUT_PATH / f"{dataset}/csv/mcd_slice_{recording}.csv", read_type)

        if PLOT_HIST:
            for gamma in gamma_ls:
                for _lambda in _lambda_ls:
                    
                    df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, N, num_iters, gamma, epsilon, _lambda)
                    u_sim = torch.tensor(u_sim.values)
                    print(type(u_sim))

                    u_plot = [u_exp, u_sim, u_overlap(u_sim, u_exp) ]

                    ratios = blue_ratio_blocks(u_exp, u_sim, tol=0.25, block_size=16)

                    fig, axs = plt.subplots( 3, 2, figsize = (8,8))

                    for ii, u in enumerate(u_plot):
                        axs[ii, 0].imshow(u, cmap=colors[ii],origin="lower", extent=(0,1,0,1))
                        axs[ii, 0].set_box_aspect(1)
                        axs[ii, 0].axes.get_xaxis().set_ticks([])
                        axs[ii, 0].axes.get_yaxis().set_ticks([])
                        axs[ii, 0].set_title(titles[ii])

                        if ii != 2:
                            axs[ii, 1].hist(u.ravel(), bins=256, density = True, color=colors_hist[ii])
                        else:
                            axs[ii, 1].hist(ratios, bins = 40, density = True)
                        
                        axs[ii, 1].set_xlabel("$u_{ij}$")
                        #axs[ii, 1].set_ylabel("norm. sample distribution $p(u_{ij})$")
                        axs[ii, 1].set_ylabel("$p(u_{ij})$")
                        axs[ii, 1].grid(color = "gray")


                    fig.tight_layout()
                    plt.savefig(OUTPUT_PATH / f"recording={recording}_postprocessing_lambda={_lambda:.2f}_gamma={gamma}_num-iters={num_iters}.png")
                    #plt.show()

        if PLOT_DIFF_COMPARISON:
            fig, axs = plt.subplots( len(gamma_ls), len(_lambda_ls), figsize = (8,8))

            for ii, gamma in enumerate(gamma_ls):
                for jj, _lambda in enumerate(_lambda_ls):

                    df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, N, num_iters, gamma, epsilon, _lambda)

                    u_sim = torch.tensor(u_sim.values)
                    
                    im = axs[ii, jj].imshow(u_overlap(u_sim, u_exp), cmap= "viridis", origin="lower", extent=(0,1,0,1))
                    axs[ii, jj].set_box_aspect(1)
                    axs[ii, jj].axes.get_xaxis().set_ticks([])
                    axs[ii, jj].axes.get_yaxis().set_ticks([])
                    #axs[ii, jj].set_title(f"$\\gamma = {gamma:.3f}$")

            #fig.tight_layout()
            fig.canvas.draw()  # ensures positions are compute
            
            for kk, _lambda in enumerate(_lambda_ls):

                bbox = axs[0, kk].get_position()
                x_center = 0.5 * (bbox.x0 + bbox.x1)
                y_top = bbox.y1 + 0.02
                if kk == 0:
                    title = f"$\\lambda := std(u_0) = {_lambda:.2f}$"
                else:
                    title = f"$\\lambda = {_lambda:.2f}$"
                fig.text(x_center, y_top, title, ha="center", va = "bottom", fontsize=12)

            
            for jj, _gamma in enumerate(gamma_ls):
                _ax  = axs[jj, 0]
                boox = _ax.get_position()
                x_left = boox.x0 - 0.02
                y_center = 0.5 * (boox.y0 + boox.y1)
                title = f"$\\gamma = {_gamma:.3f}$"
                fig.text(x_left, y_center, title, ha="right", va="center", rotation = "vertical", fontsize=12)
            
            plt.colorbar(im)
            plt.savefig(OUTPUT_PATH / f"recording={recording}_postprocessing_overview_{read_type}.png")
            #plt.show()

        if PLOT_ENERGY_CONVERGENCE_COMPARISON:
            
            fig = plt.figure(figsize=(8, 6))

            dict = {}
        
            for jj, _lambda in enumerate(_lambda_ls):
                e_ls = []
                for ii, gamma in enumerate(gamma_ls):
                    df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, N, num_iters, gamma, epsilon, _lambda)
                    energies = df_energies.values
                    e_ls.append(energies[-1])


                dict[_lambda] = e_ls
                plt.plot(gamma_ls, e_ls, label = f"$\\lambda$ = {_lambda}")

            plt.title(f"Energy convergence for dataset: {dataset} \n recording: {recording} / reading type: {read_type}")
            plt.legend()
            plt.xlabel("$\\gamma$")
            plt.ylabel("Converged energy value")
            plt.grid(color = "gray")
            plt.savefig(OUTPUT_PATH / f"recording={recording}_postprocessing_energy_convergence_{read_type}.png", dpi = 300)
            plt.show()
            
