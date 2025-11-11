import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict, replace
from pathlib import Path

from pattern_formation import initialize_u0_random
from params import labyrinth_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params
from env_utils import PATHS, print_bars, plotting_style

from evaluation import gradient_descent_nesterov_evaluation
from read import read_csv

# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluating multiple RCP / LCP datastacks from experimental Magnetic Imaging.")
    parser.add_argument("--dataset", required=True, type=str, help="Folder containing LCP *.TIF and *.DAT files.")
    parser.add_argument("--recording", required=True, type=str, help="Recorded slices.")
    
    return parser.parse_args()

# ---------------------------------------------------------------

if __name__ == "__main__":
        
    LIVE_PLOT = DATA_LOG = False
    args = parse_args()

    # ---------------------------------------------------------------

    dataset = args.dataset
    recording = args.recording

    # INPUT PATHs
    INPUT_PATH = PATHS.BASE_EXPDATA

    INPUT_FILE_PATH = PATHS.BASE_EXPDATA / f"{dataset}/csv/mcd_slice_{recording}.csv"

    # OUTPUT PATHs
    OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt/"
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------

    with open(INPUT_PATH / f"{dataset}" / "params_file.json", "r") as _file:
        params_file = json.load(_file)

    gamma_ls = params_file[recording]

    u_exp = read_csv(INPUT_FILE_PATH, "standardize", PLOT = False)

    if u_exp.shape[0] != u_exp.shape[1]:
        raise ValueError("Experimental data should be quadratic (NxN tensor).")
    N_exp = u_exp.shape[0]

    # ---------------------------------------------------------------

    num_iters = 5000

    labyrinth_data_params = replace(labyrinth_data_params, N = N_exp)
    ngd_sim_params = replace(ngd_sim_params, num_iters = num_iters)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    u0 = initialize_u0_random(N, REAL=True)

    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print_bars()

    _lambda = torch.std(u0)

    print(f"Learning Rate Lambda := std(u0) = {_lambda}")
    print_bars()

    plotting_style()
    fig, axs = plt.subplots( len(gamma_ls), 2, figsize = (6,3 * len(gamma_ls) ))

    N_ticks = 4

    for ii, _gamma in enumerate(gamma_ls):
        print("Gamma: ", _gamma)
        labyrinth_data_params = replace(labyrinth_data_params, gamma = _gamma)
        u, energies = gradient_descent_nesterov_evaluation(u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), STOP_BY_TOL=True)
        
        axs[ii, 0].imshow(u.cpu().numpy(), cmap='gray',origin="lower", extent=(0,1,0,1))
        axs[ii, 0].set_box_aspect(1)
        axs[ii, 0].set_title(f"$\\gamma = {_gamma}, \\lambda := std(u_0) = {_lambda:.2f}$")
        axs[ii, 0].axes.get_xaxis().set_ticks([])
        axs[ii, 0].axes.get_yaxis().set_ticks([])


        axs[ii, 1].plot(np.arange(0,len(energies), 1), energies)
        axs[ii, 1].set_box_aspect(1)
        axs[ii, 1].set_title(f"$\\Sigma(N-10:-1) \\Delta E < 0.1$")
        
        ymin, ymax = axs[ii, 1].get_ylim()
        xmin, xmax = axs[ii, 1].get_xlim()
        axs[ii, 1].set_yticks(np.round(np.linspace(ymin, ymax, N_ticks), 2))
        axs[ii, 1].set_xticks(np.round(np.linspace(xmin, xmax, N_ticks), 2))
        
        print_bars()
    
    fig.tight_layout()
    plt.savefig(OUTPUT_PATH / f"recording={recording}_lambda={_lambda:.2f}_num-iters={num_iters}.png", dpi = 300)
    #plt.show()
    #plt.close()
