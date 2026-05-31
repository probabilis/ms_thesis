import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path 
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------

term_size = os.get_terminal_size() # get current terminal size for screen wide printing
main_colormap = "managua" # main color map used
N_ticks = 4 

def adapt_managua_low_white(n=256, transition=0.18):
    """
    modified managua colormap for FFT plot (for low values white)
    """
    base = plt.colormaps["managua"]

    colors = base(np.linspace(0, 1, n))
    k = int(n * transition)
    orange = colors[k].copy()
    white = np.array([1.0, 1.0, 1.0, 1.0])

    for i in range(k):
        t = i / max(k - 1, 1)
        colors[i] = (1 - t) * white + t * orange

    return LinearSegmentedColormap.from_list(
        "managua_white_low",
        colors,
        N=n
    )


sub_colormap = adapt_managua_low_white(transition=0.2)


# ---------------------------------------------------------------

class PATHS:
    """
    class for saving all Path's of the project
    """
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
    PATH_GAMMA_SWEEP = BASE_OUTPUT / 'gamma_sweep'

    PATH_EXAMPLES = _BASE / 'examples'
    PATH_THESIS = _BASE / 'thesis_helper'

    PATH_EVOLUTION = BASE_OUTPUT / 'evolution'    

# ---------------------------------------------------------------

def plotting_style(CHANGE_FONT_SIZES = True, USE_TEX = True):
    """
    function for unit plotting style of project
    """
    plt.style.use('classic')

    if USE_TEX:
        
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif'
            })

    if CHANGE_FONT_SIZES:
        
        plt.rc('font', size=16) # default text font size
        plt.rc('axes', titlesize=16) # axes title text font size 
        plt.rc('axes', labelsize=16) # axes label text font size
        plt.rc('xtick', labelsize=16) # font size for x tick labels
        plt.rc('ytick', labelsize=16) # font size for y tick labels
        plt.rc('legend', fontsize=18) # font size for legend
        plt.rc('figure', titlesize=20) # font size of figure title
# ---------------------------------------------------------------

def print_bars(term_size = term_size):
    print(term_size.columns * "-")

# ---------------------------------------------------------------

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

# ---------------------------------------------------------------

def get_filename(N, num_iters, gamma, epsilon, _lambda = None):
    """
    function for getting filename of consistent file namings
    """
    if _lambda is not None:
        return f"N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}_lambda={_lambda}"
    else:
        return f"N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}"

# ---------------------------------------------------------------

def log_data(folder_path, u, energies, N, num_iters, gamma, epsilon, _lambda = None):
    """
    function for logging files consistently
    """
    file_name = get_filename(N, num_iters, gamma, epsilon, _lambda)

    df_energies = pd.DataFrame(energies)
    u_np = u.numpy()
    df_u = pd.DataFrame(u_np)

    _path_1 = folder_path / f"{file_name}_energy_data.csv"
    _path_2 = folder_path / f"{file_name}_pattern_data.csv"
    
    df_energies.to_csv(_path_1, index = False, header = False)   
    df_u.to_csv(_path_2, index = False, header = False)

    print(f"Sucessfully saved data: \n {_path_1} \n {_path_2}.")

# ---------------------------------------------------------------

def log_data_history(folder_path, u, history, N, num_iters, gamma, epsilon, _lambda = None):
    """
    function for logging files consistently -> not used
    """
    file_name = get_filename(N, num_iters, gamma, epsilon, _lambda)

    df_energies = pd.DataFrame(history)
    u_np = u.numpy()
    df_u = pd.DataFrame(u_np)

    _path_1 = folder_path / f"{file_name}_energy_data.csv"
    _path_2 = folder_path / f"{file_name}_pattern_data.csv"
    
    df_energies.to_csv(_path_1, index = False)   
    df_u.to_csv(_path_2, index = False, header = False)

    print(f"Sucessfully saved data: \n {_path_1} \n {_path_2}.")

# ---------------------------------------------------------------

def read_sim_dat_from_csv(folder_path, N, num_iters, gamma, epsilon, _lambda = None):
    """
    function for reading files
    """
    file_name = get_filename(N, num_iters, gamma, epsilon, _lambda)

    _path_1 = folder_path / f"{file_name}_energy_data.csv"
    _path_2 = folder_path / f"{file_name}_pattern_data.csv"

    df_energies = pd.read_csv(_path_1) # header = None
    df_u = pd.read_csv(_path_2, header = None)

    return df_energies, df_u

# ---------------------------------------------------------------

def plotting_schematic(folder_path, fig, ax1, ax2, u, energies, N, num_iters, gamma, epsilon, ii, DATA_LOG):
    """
    function of standard plotting schematic for optimization algos
    """
    plotting_style()

    ax1.clear()
    ax2.clear()
    file_name = get_filename(N, num_iters, gamma, epsilon)

    ax1.imshow(u.real.cpu().numpy(), cmap=main_colormap, extent=(0,1,0,1))
    ax2.loglog(torch.arange(1,len(energies)+1), energies)

    fig.suptitle(f"$\\gamma = {gamma}, \\epsilon = {epsilon} / ii = {ii}$") 
    if DATA_LOG:
        fig.savefig(folder_path / f"{file_name}_energy.png")

# ---------------------------------------------------------------

def plotting_schematic_eval(folder_path, fig, ax1, ax2, u, energies, N, num_iters, gamma, epsilon, _lambda, ii, DATA_LOG):
    """
    function of standard plotting schematic for optimization algos for evaluation
    """
    plotting_style()
    
    ax1.clear()
    ax2.clear()
    file_name = get_filename(N, num_iters, gamma, epsilon) + f"_lambda={_lambda}"

    ax1.imshow(u.cpu().numpy(), cmap=main_colormap, origin="lower", extent=(0,1,0,1))
    ax2.loglog(torch.arange(1,len(energies)+1), energies)
    ax2.grid(color="gray")

    fig.suptitle(f"$\\gamma = {gamma}, \\epsilon = {epsilon}, \\lambda = {_lambda:.3f} / ii = {ii}$")   

    if DATA_LOG:
        fig.savefig(folder_path / f"{file_name}_data_log.png")

# ---------------------------------------------------------------

def tensor_type(x):
    """
    check tensor type
    """
    print("Is float : ",isinstance(x,torch.FloatTensor) )
    print("Is double : ", isinstance(x,torch.DoubleTensor) )

# ---------------------------------------------------------------

def get_args():
    """
    function for getting arguments for live_plot and data_log
    """
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


def parse_args_exp_data() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluating multiple RCP / LCP datastacks from experimental Magnetic Imaging.")
    parser.add_argument("--dataset", required=True, type=str, help="Folder containing LCP *.TIF and *.DAT files.")
    parser.add_argument("--recording", required=True, type=str, help="Recorded slices.")
    
    return parser.parse_args()