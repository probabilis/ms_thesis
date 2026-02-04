import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict, replace
from pathlib import Path

from pattern_formation import initialize_u0_random
from params import exp_data_params, get_DataParameters, get_SimulationParamters
from params import pgd_sim_params as ngd_sim_params
from env_utils import PATHS, print_bars, plotting_style, read_sim_dat_from_csv

from evaluation import gradient_descent_nesterov_evaluation
from read import read_csv

# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluating multiple RCP / LCP datastacks from experimental Magnetic Imaging.")
    parser.add_argument("--dataset", required=True, type=str, help="Folder containing LCP *.TIF and *.DAT files.")
    parser.add_argument("--recording", required=True, type=str, help="Recorded slices.")
    
    return parser.parse_args()

# ---------------------------------------------------------------

def different_image_preprocessings(exp_data_params, ngd_sim_params, SIMULATE_OR_READ):

    LIVE_PLOT = False
    DATA_LOG = True
    args = parse_args()

    ENERGY_STOP_TOL = 1e-6

    num_iters = 5000

    # ---------------------------------------------------------------

    dataset = args.dataset
    recording = args.recording

    # INPUT PATHs
    INPUT_PATH = PATHS.BASE_EXPDATA

    INPUT_FILE_PATH = PATHS.BASE_EXPDATA / f"{dataset}/csv/mcd_slice_{recording}.csv"

    # ---------------------------------------------------------------

    with open(INPUT_PATH / f"{dataset}" / "params_file.json", "r") as _file:
        params_file = json.load(_file)

    gamma_ls = params_file[recording]

    types = ["raw", "standardize", "shift", "clipped"]

    for _type in types:
        print("Reading type:", _type)

        OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording / _type
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

        u_exp = read_csv(INPUT_FILE_PATH, _type, PLOT = False)

        if u_exp.shape[0] != u_exp.shape[1]:
            raise ValueError("Experimental data should be quadratic (NxN tensor).")
        N_exp = u_exp.shape[0]

        # ---------------------------------------------------------------

        ngd_sim_params = replace(ngd_sim_params, num_iters = num_iters)

        gridsize, N, th, epsilon, gamma = get_DataParameters(exp_data_params)
        u0 = initialize_u0_random(N, REAL=True)

        print_bars()
        print(exp_data_params)
        print(ngd_sim_params)
        print_bars()

        _lambda_ls = [0.01,0.05, 0.1, 0.2]

        print(f"Learning Rate Lambda: {_lambda_ls}")
        print_bars()

        plotting_style()
        N_ticks = 4
        
        fig, axs = plt.subplots( len(gamma_ls), 2 * len(_lambda_ls), figsize = (6 * len(_lambda_ls), 3 * len(gamma_ls) ))

        for kk, _lambda in enumerate(_lambda_ls):
            print("Lambda: ", _lambda)

            for ii, _gamma in enumerate(gamma_ls): 
                print("Gamma: ", _gamma)
                exp_data_params = replace(exp_data_params, gamma = _gamma)
                
                if SIMULATE_OR_READ == "simulate":
                    print("Simulating")
                    u, energies = gradient_descent_nesterov_evaluation(u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH, **asdict(exp_data_params),**asdict(ngd_sim_params), STOP_BY_TOL=True, ENERGY_STOP_TOL = ENERGY_STOP_TOL)
                elif SIMULATE_OR_READ == "read":
                    print("Reading")
                    df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, N, num_iters, _gamma, epsilon, _lambda)
                    energies = df_energies.values
                    u = torch.tensor(u_sim.values)


                axs[ii, 2*kk].imshow(u.cpu().numpy(), cmap='gray',origin="lower", extent=(0,1,0,1))
                axs[ii, 2*kk].set_box_aspect(1)
                
                axs[ii, 2*kk].set_title(f"$\\gamma = {_gamma:.3f}$")

                axs[ii, 2*kk].axes.get_xaxis().set_ticks([])
                axs[ii, 2*kk].axes.get_yaxis().set_ticks([])

                axs[ii, 2*kk + 1].plot(np.arange(0,len(energies), 1), energies)
                axs[ii, 2*kk + 1].set_box_aspect(1)
                
                if ii == 0:
                    axs[ii, 2*kk + 1].set_title(f"$\\Delta E < {ENERGY_STOP_TOL}$")
                
                ymin, ymax = axs[ii, 2*kk + 1].get_ylim()
                xmin, xmax = axs[ii, 2*kk + 1].get_xlim()
                axs[ii, 2*kk + 1].set_yticks(np.round(np.linspace(ymin, ymax, N_ticks), 2))
                axs[ii, 2*kk + 1].set_xticks(np.round(np.linspace(xmin, xmax, N_ticks), 2))
                axs[ii, 2*kk + 1].grid(color = "gray")

                print_bars()
            

        fig.tight_layout()
        fig.canvas.draw()  # ensures positions are compute
        for kk, _lambda in enumerate(_lambda_ls):
            # Get positions of the two columns
            ax_left  = axs[0, 2*kk]
            ax_right = axs[0, 2*kk + 1]

            bbox_l = ax_left.get_position()
            bbox_r = ax_right.get_position()

            x_center = 0.5 * (bbox_l.x0 + bbox_r.x1)
            y_top = bbox_l.y1 + 0.02  # small offset above top row
            title = f"$\\lambda = {_lambda:.2f}$"

            fig.text(x_center, y_top, title, ha="center", va="bottom", fontsize=14)
            
        plt.savefig(OUTPUT_PATH / f"recording={recording}_num-iters={num_iters}_{_type}.png", dpi = 300)


def different_image_lambdas_and_gammas(exp_data_params, ngd_sim_params, SIMULATE_OR_READ):

    LIVE_PLOT = False
    DATA_LOG = True
    args = parse_args()

    ENERGY_STOP_TOL = 1e-8

    num_iters = 5_000

    # ---------------------------------------------------------------

    dataset = args.dataset
    recording = args.recording

    # INPUT PATHs
    INPUT_PATH = PATHS.BASE_EXPDATA

    INPUT_FILE_PATH = PATHS.BASE_EXPDATA / f"{dataset}/csv/mcd_slice_{recording}.csv"

    # ---------------------------------------------------------------

    with open(INPUT_PATH / f"{dataset}" / "params_file.json", "r") as _file:
        params_file = json.load(_file)

    gamma_ls = params_file[recording]

    OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording 
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    u_exp = read_csv(INPUT_FILE_PATH, "standardize", PLOT = False)

    if u_exp.shape[0] != u_exp.shape[1]:
        raise ValueError("Experimental data should be quadratic (NxN tensor).")

    # ---------------------------------------------------------------

    ngd_sim_params = replace(ngd_sim_params, num_iters = num_iters)

    gridsize, N, th, epsilon, gamma = get_DataParameters(exp_data_params)
    u0 = initialize_u0_random(N, REAL=True)

    print_bars()
    print(exp_data_params)
    print(ngd_sim_params)
    print(f"STOP_BY_TOL = {STOP_BY_TOL}")
    print_bars()

    _lambda_ls = [0.001, 0.01]

    print(f"Learning Rate Lambda := {_lambda_ls}")
    print_bars()

    plotting_style()
    N_ticks = 4
    
    fig, axs = plt.subplots( len(gamma_ls), 2 * len(_lambda_ls), figsize = (6 * len(_lambda_ls), 3 * len(gamma_ls) ))

    for kk, _lambda in enumerate(_lambda_ls):
        print("Lambda: ", _lambda)

        for ii, _gamma in enumerate(gamma_ls): 
            print("Gamma: ", _gamma)
            exp_data_params = replace(exp_data_params, gamma = _gamma)
            
            if SIMULATE_OR_READ == "simulate":
                print("Simulating")
                u, energies = gradient_descent_nesterov_evaluation(u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH, **asdict(exp_data_params),**asdict(ngd_sim_params), STOP_BY_TOL=True, ENERGY_STOP_TOL = ENERGY_STOP_TOL)
            elif SIMULATE_OR_READ == "read":
                print("Reading")
                df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, N, num_iters, _gamma, epsilon, _lambda)
                energies = df_energies.values
                u = torch.tensor(u_sim.values)


            axs[ii, 2*kk].imshow(u.cpu().numpy(), cmap='gray',origin="lower", extent=(0,1,0,1))
            axs[ii, 2*kk].set_box_aspect(1)
            
            axs[ii, 2*kk].set_title(f"$\\gamma = {_gamma:.5f}$")

            axs[ii, 2*kk].axes.get_xaxis().set_ticks([])
            axs[ii, 2*kk].axes.get_yaxis().set_ticks([])

            axs[ii, 2*kk + 1].plot(np.arange(0,len(energies), 1), energies)
            axs[ii, 2*kk + 1].set_box_aspect(1)
            
            if ii == 0:
                axs[ii, 2*kk + 1].set_title(f"$\\Delta E < {ENERGY_STOP_TOL}$")
            
            ymin, ymax = axs[ii, 2*kk + 1].get_ylim()
            xmin, xmax = axs[ii, 2*kk + 1].get_xlim()
            axs[ii, 2*kk + 1].set_yticks(np.round(np.linspace(ymin, ymax, N_ticks), 2))
            axs[ii, 2*kk + 1].set_xticks(np.round(np.linspace(xmin, xmax, N_ticks), 2))
            axs[ii, 2*kk + 1].grid(color = "gray")

            print_bars()
        

    #fig.tight_layout()
    fig.canvas.draw()  # ensures positions are compute
    for kk, _lambda in enumerate(_lambda_ls):
        ax_left  = axs[0, 2*kk]
        ax_right = axs[0, 2*kk + 1]

        bbox_l = ax_left.get_position()
        bbox_r = ax_right.get_position()

        x_center = 0.5 * (bbox_l.x0 + bbox_r.x1)
        y_top = bbox_l.y1 + 0.02

        title = f"$\\lambda = {_lambda}$"

        fig.text(x_center, y_top, title, ha="center", va="bottom", fontsize=14)
        
    plt.savefig(OUTPUT_PATH / f"recording={recording}_num-iters={num_iters}.png", dpi = 300)



if __name__ == "__main__":
    #different_image_preprocessings(exp_data_params, ngd_sim_params, "simulate")
    different_image_lambdas_and_gammas(exp_data_params, ngd_sim_params, "simulate")
