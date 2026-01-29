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



def wall_mask_from_labels(L):
    # L: bool (H,W)
    W = torch.zeros_like(L, dtype=torch.bool)
    W[:, 1:] |= (L[:, 1:] != L[:, :-1])
    W[1:, :] |= (L[1:, :] != L[:-1, :])
    return W



@torch.no_grad()
def quality_scores(u_exp, u_opt, blur_sigma=1.0):
    """
    Fisher discriminant & Perimeter functional
    """
    # labels from optimized solution
    L = (u_opt > 0)

    """
    https://www.wikiwand.com/en/articles/Linear_discriminant_analysis
    """
    I1 = u_exp[L]
    I0 = u_exp[~L]
    mu1, mu0 = I1.mean(), I0.mean()

    v1 = I1.var(unbiased=True) if I1.numel() > 1 else torch.tensor(0., device=u_exp.device)
    v0 = I0.var(unbiased=True) if I0.numel() > 1 else torch.tensor(0., device=u_exp.device)

    fisher_J = (mu1 - mu0).pow(2) / (v1 + v0 + 1e-12)
    
    # boundary calculation
    W = wall_mask_from_labels(L)
    perimeter = W.float().sum()

    return float(fisher_J), float(perimeter), W.float()



if __name__ == "__main__":

    plotting_style()


    PLOT_ENERGY_CONVERGENCE_COMPARISON = False

    dataset = "data_00"
    recording = "003"

    INPUT_PATH = PATHS.BASE_EXPDATA

    colors = ['gray', 'gray', 'viridis']
    colors_hist = ['black', 'black', 'cornflowerblue']
    titles = ["raw recording $u_{exp}$", "optimized $u_{opt}$", "if $u_{opt}^{ij} < tol \\rightarrow \\alpha + u_{exp}$"] # "RMSE $|u_{exp} - u_{opt}|^2$"

    # ---------------------------------------------------------------

    with open(INPUT_PATH / f"{dataset}" / "params_file.json", "r") as _file:
        params_file = json.load(_file)

    gamma_ls = params_file[recording]
    _lambda_ls = [0.001, 0.01]
    print("Gamma's: ", gamma_ls)
    print("Lambda's: ", _lambda_ls)
    num_iters = 5000

    gridsize, N, th, epsilon, gamma = get_DataParameters(exp_data_params)

    OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording
    u_exp = read_csv(INPUT_PATH / f"{dataset}/csv/mcd_slice_{recording}.csv", "standardize")

    records = []

    bars = ["Fisher discriminant $S(u)$", "Perimeter($u$)"]

    THRESHOLD = 0.2

    # ---------------------------------------------------------------

    fig, axs = plt.subplots( len(gamma_ls), 2 * len(_lambda_ls), figsize = (18,16) )

    for ii, gamma in enumerate(gamma_ls):
        for jj, _lambda in enumerate(_lambda_ls):

            df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, N, num_iters, gamma, epsilon, _lambda)
            u_sim = torch.tensor(u_sim.values)
            new = torch.where(torch.abs(u_sim) > 0.2, 0, u_sim)

            fisherJ, perimeter, W = quality_scores(u_exp, u_sim)

            rec = {
                "gamma": float(gamma),
                "lambda": float(_lambda),
                "fisherJ": float(fisherJ),
                "perimeter": float(perimeter)
            }
            records.append(rec)

            if jj == 1:
                jj = 1 + jj
            
            axs[ii, jj+0].imshow(torch.where(torch.abs(u_sim) < THRESHOLD, 10, u_sim), origin="lower", extent=(0,1,0,1))
            axs[ii, jj+0].set_box_aspect(1)
            axs[ii, jj+0].axes.get_xaxis().set_ticks([])
            axs[ii, jj+0].axes.get_yaxis().set_ticks([])

            axs[ii, jj].set_title(f"$P$ = {perimeter:.2f} | $S$ = {fisherJ:.3f}", fontsize = 8)

            axs[ii, jj+1].hist(u_sim.ravel(), bins=256, density = True)
            axs[ii, jj+1].set_xlabel("$u_{ij}$")
            #axs[ii, 1].set_ylabel("norm. sample distribution $p(u_{ij})$")
            axs[ii, jj+1].set_ylabel("$p(u_{ij})$")
            axs[ii, jj+1].grid(color = "gray")

            axs[ii, jj+1].set_xlim(-1.5, +1.5)
            
    
    fig.canvas.draw()  # ensures positions are compute

    for kk, _lambda in enumerate(_lambda_ls):
        if kk == 1:
            kk = 2 + kk
        bbox = axs[0, kk].get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        y_top = bbox.y1 + 0.02
        title = f"$\\lambda = {_lambda}$"
        fig.text(x_center, y_top, title, ha="center", va = "bottom", fontsize=12)


    for jj, _gamma in enumerate(gamma_ls):
        _ax  = axs[jj, 0]
        boox = _ax.get_position()
        x_left = boox.x0 - 0.02
        y_center = 0.5 * (boox.y0 + boox.y1)
        title = f"$\\gamma = {_gamma}$"
        fig.text(x_left, y_center, title, ha="right", va="center", rotation = "vertical", fontsize=12)


    df = pd.DataFrame.from_records(records)

    cut = df["perimeter"].quantile(0.6)
    df_filt = df[df["perimeter"] <= cut].copy()

    print("Best parameter constellation: ")
    best = df_filt.sort_values(["fisherJ", "perimeter"], ascending = [False, True])
    print(best)
    print(best.index)

    fig.suptitle("Fisher discriminant $S(u)$ and Perimeter $P(u)$")

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

