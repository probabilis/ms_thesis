import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt 

from utils.env_utils import PATHS, read_sim_dat_from_csv, plotting_style,parse_args_exp_data, main_colormap
from params.opt_params import labyrinth_data_params, exp_data_params, get_DataParameters

from exp_data_processing.read import read_csv
from exp_data_processing.postprocessing import quality_scores, wall_mask_from_labels


if __name__ == "__main__":

    plotting_style()

    PLOT_ENERGY_CONVERGENCE_COMPARISON = False

    args = parse_args_exp_data()   
    dataset = args.dataset
    recording = args.recording

    INPUT_PATH = PATHS.BASE_EXPDATA

    colors = ['gray', 'gray', 'viridis']
    colors_hist = ['black', 'black', 'cornflowerblue']
    titles = ["raw recording $u_{exp}$", "optimized $u_{opt}$", "if $u_{opt}^{ij} < tol \\rightarrow \\alpha + u_{exp}$"] # "RMSE $|u_{exp} - u_{opt}|^2$"

    # ---------------------------------------------------------------

    with open(INPUT_PATH / f"{dataset}" / "params_file.json", "r") as _file:
        params_file = json.load(_file)

    gamma_ls = params_file[recording]
    _lambda_ls = [0.001, 0.01, 0.1]
    print("Gamma's: ", gamma_ls)
    print("Lambda's: ", _lambda_ls)
    num_iters = 5000

    gridsize, N, th, epsilon, gamma = get_DataParameters(exp_data_params)

    OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording
    u_exp = read_csv(INPUT_PATH / f"{dataset}" / "csv", recording, PLOT = False) #OUTPUT_PATH / "raw.png"

    records = []

    bars = ["Fisher discriminant $S(u)$", "Perimeter($u$)"]

    THRESHOLD = 0.25

    WITH_FISHER_PERIMETER_TITLE = True

    # ---------------------------------------------------------------

    fig, axs = plt.subplots( len(gamma_ls), 2 * len(_lambda_ls), figsize = (len(gamma_ls) * 4.5, len(_lambda_ls) * 6 ) ) # 18 x 15

    for ii, gamma in enumerate(gamma_ls):
        for jj, _lambda in enumerate(_lambda_ls):

            df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, N, num_iters, gamma, epsilon, _lambda)
            u_sim = torch.tensor(u_sim.values)

            fisherJ, perimeter, W = quality_scores(u_exp, u_sim)

            perimeter /= u_exp.shape[0]**2
            perimeter *= 100

            rec = {
                "gamma": float(gamma),
                "lambda": float(_lambda),
                "fisherJ": float(fisherJ),
                "perimeter": float(perimeter)
            }
            records.append(rec)
            
            # plot W instead of u_sim? 
            im = axs[ii, 2*jj].imshow(torch.where(torch.abs(u_sim) < THRESHOLD, 2.0, u_sim), cmap = main_colormap, origin="lower", extent=(0,1,0,1))
            axs[ii, 2*jj].set_box_aspect(1)
            axs[ii, 2*jj].axes.get_xaxis().set_ticks([])
            axs[ii, 2*jj].axes.get_yaxis().set_ticks([])
            if WITH_FISHER_PERIMETER_TITLE:
                axs[ii, 2*jj].set_title(f"$P_\%(u) = {perimeter:.2f} \%$ \n $S(u) = {fisherJ:.3f}$", fontsize = 12)

            axs[ii, 2*jj+1].hist(u_sim.ravel(), bins=32, density = True, alpha = 0.6, color = 'b')
            if WITH_FISHER_PERIMETER_TITLE:
                axs[ii, 2*jj+1].set_xlabel("$u^{(ij)}$")
                axs[ii, 2*jj+1].set_ylabel("$p[u^{(ij)}]$")
            axs[ii, 2*jj+1].grid(color = "gray")

            axs[ii, 2*jj+1].set_xlim(-1.5, +1.5)
            
            axs[ii, 2*jj+1].set_xticks([-1.5, 0.0, +1.5])
            ymin, ymax = axs[ii, 2*jj+1].get_ylim()
            axs[ii, 2*jj+1].set_yticks(np.round(np.linspace(ymin, ymax, 4), 0))

    fig.canvas.draw()  # ensures positions are compute

    for kk, _lambda in enumerate(_lambda_ls):
        bbox_left = axs[0, 2*kk].get_position()
        bbox_right = axs[0, 2*kk+1].get_position()
        x_center = 0.5 * (bbox_left.x0 + bbox_right.x1)
        y_top = bbox_right.y1 + 0.02
        title = f"$\\lambda = {_lambda}$"
        fig.text(x_center, y_top, title, ha="center", va = "bottom", fontsize=20)


    for jj, _gamma in enumerate(gamma_ls):
        _ax  = axs[jj, 0]
        boox = _ax.get_position()
        x_left = boox.x0 - 0.02
        y_center = 0.5 * (boox.y0 + boox.y1)
        title = f"$\\gamma = {_gamma}$"
        fig.text(x_left, y_center, title, ha="right", va="center", rotation = "vertical", fontsize=20)


    df = pd.DataFrame.from_records(records)

    cut = df["perimeter"].quantile(0.6)
    df_filt = df[df["perimeter"] <= cut].copy()
    print(df)
    print("Best parameter constellation: ")
    best = df_filt.sort_values(["fisherJ", "perimeter"], ascending = [False, True])
    print(best)
    print(best.index)

    df.to_csv(OUTPUT_PATH / f"ranking.csv")

    

    #fig.suptitle("Fisher discriminant $S(u)$ and Perimeter $P(u)$")
    if WITH_FISHER_PERIMETER_TITLE:
        plt.savefig(OUTPUT_PATH / f"recording={recording}_postprocessing_overview.png", dpi = 300)
    else:
        plt.savefig(OUTPUT_PATH / f"recording={recording}_postprocessing_overview_without_label.png", dpi = 300)
    plt.show()


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

        plt.title(f"Energy convergence for dataset: {dataset} \n recording: {recording}")
        plt.legend()
        plt.xlabel("$\\gamma$")
        plt.ylabel("Converged energy value")
        plt.grid(color = "gray")
        plt.savefig(OUTPUT_PATH / f"recording={recording}_postprocessing_energy_convergence.png", dpi = 300)
        plt.show()