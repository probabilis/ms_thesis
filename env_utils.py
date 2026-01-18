import os
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path 
import pandas as pd


# ---------------------------------------------------------------

class PATHS:

    _BASE = Path() / 'data'

    BASE_EXPDATA = _BASE / 'expdata'
    BASE_OUTPUT = _BASE / 'tests'

    PATH_CN = BASE_OUTPUT / 'cn'
    PATH_GD = BASE_OUTPUT / 'gd'
    PATH_PGD = BASE_OUTPUT / 'pgd'
    PATH_NESTEROV = BASE_OUTPUT / 'nesterov'

    PATH_COMPARISON = BASE_OUTPUT / 'comparison'
    PATH_EVALUATION = BASE_OUTPUT / 'evaluation'
    PATH_PARAMS_STUDY = BASE_OUTPUT / 'params_study'

    PATH_EXAMPLES = _BASE / 'examples'


    

# ---------------------------------------------------------------


def plotting_style():
    plt.style.use('classic')


# ---------------------------------------------------------------

term_size = os.get_terminal_size()


def print_bars(term_size = term_size):
    print(term_size.columns * "-")


class bcolors:
    """
    class for colored terminal output
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_filename(N, num_iters, gamma, epsilon, _lambda = None):
    if _lambda is not None:
        return f"N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}_lambda={_lambda:.2f}"
    else:
        return f"N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}"


def log_data(folder_path, u, energies, N, num_iters, gamma, epsilon, _lambda = None):

    file_name = get_filename(N, num_iters, gamma, epsilon, _lambda)

    df_energies = pd.DataFrame(energies)
    u_np = u.numpy()
    df_u = pd.DataFrame(u_np)

    _path_1 = folder_path / f"{file_name}_energy_data.csv"
    _path_2 = folder_path / f"{file_name}_pattern_data.csv"
    
    df_energies.to_csv(_path_1, index = False, header = False)   
    df_u.to_csv(_path_2, index = False, header = False)

    print(f"Sucessfully saved data: \n {_path_1} \n {_path_2}.")



def read_sim_dat_from_csv(folder_path, N, num_iters, gamma, epsilon, _lambda = None):

    file_name = get_filename(N, num_iters, gamma, epsilon, _lambda)

    _path_1 = folder_path / f"{file_name}_energy_data.csv"
    _path_2 = folder_path / f"{file_name}_pattern_data.csv"

    df_energies = pd.read_csv(_path_1, header = None)
    df_u = pd.read_csv(_path_2, header = None)

    return df_energies, df_u



def plotting_schematic(folder_path, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, ii):
    ax1.clear()
    ax2.clear()
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    file_name = get_filename(N, num_iters, gamma, epsilon)

    ax1.imshow(u.real.cpu().numpy(), cmap='gray', extent=(0,1,0,1))
    ax1.set_title(f"Iteration {ii}")
    fig1.savefig(folder_path / f"{file_name}_pattern.png")
    ax2.plot(torch.arange(0,len(energies)), energies)

    ax2.set_title("energy evolution")
    fig2.savefig(folder_path / f"{file_name}_energy.png")

    return None


def plotting_schematic_eval(folder_path, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, _lambda, ii):
    
    plotting_style()
    
    ax1.clear()
    ax2.clear()
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    file_name = get_filename(N, num_iters, gamma, epsilon)
    file_name = file_name + f"_lambda={_lambda}"

    ax1.imshow(u.cpu().numpy(), cmap='gray', extent=(0,1,0,1))
    ax1.set_title(f"$\\gamma$ = {gamma}, $\\epsilon$ = {epsilon}, $\\lambda$ = {_lambda:.3f} ($ii$ = {ii})")
    fig1.savefig(folder_path / f"{file_name}_pattern.png")
    ax2.plot(torch.arange(0,len(energies)), energies)

    ax2.set_title("energy evolution")
    fig2.savefig(folder_path / f"{file_name}_energy.png")
    #plt.close()

    return None


def tensor_type(x):
    print("Is float : ",isinstance(x,torch.FloatTensor) )
    print("Is double : ", isinstance(x,torch.DoubleTensor) )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--live_plot", action="store_true",
        help="Enable live plotting (default: False)"
    )
    parser.add_argument(
        "--data_log", action="store_true",
        help="Enable data logging (default: False)"
    )

    return parser.parse_args()


# --------------------------------------------------------------------


