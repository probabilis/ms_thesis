import os
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path 
import pandas as pd

# ---------------------------------------------------------------

class PATHS:

    _BASE = Path() / 'out'
    
    PATH_CN = _BASE / 'cn'
    PATH_GD = _BASE / 'gd'
    PATH_PGD = _BASE / 'pgd'
    PATH_NESTEROV = _BASE / 'nesterov'

    COMPARISON = _BASE / 'comparison'

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


def get_filename(N, num_iters, gamma, epsilon):
    return f"N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}"


def log_data(folder_path, u, energies, N, num_iters, gamma, epsilon):

    file_name = get_filename(N, num_iters, gamma, epsilon)

    df_energies = pd.DataFrame(energies)
    u_np = u.numpy()
    df_u = pd.DataFrame(u_np)
    
    df_energies.to_csv(folder_path / f"{file_name}_energy_data", index = False, header = False)
    df_u.to_csv(folder_path / f"{file_name}_pattern_data", index = False, header = False)



def plotting_schematic(folder_path, ax1, fig1, ax2, fig2, u, energies, N, num_iters, gamma, epsilon, ii):
    ax1.clear()
    ax2.clear()
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    file_name = get_filename(N, num_iters, gamma, epsilon)

    ax1.imshow(u.cpu().numpy(), cmap='gray', extent=(0,1,0,1))
    ax1.set_title(f"Iteration {ii}")
    fig1.savefig(folder_path / f"{file_name}_pattern.png")
    ax2.plot(torch.arange(0,len(energies)), energies)

    ax2.set_title("energy evolution")
    fig2.savefig(folder_path / f"{file_name}_energy.png")

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


def fourier_multiplier_simple(k_mag, th):
    """ 
    1 / k fourier multiplier
    """
    k_safe = torch.where(k_mag < 1e-12, torch.tensor(1e-12), k_mag)
    return th / k_safe

def fourier_multiplier_exp(k_mag, th):
    """
    Exponential cutoff
    """
    return th * torch.exp(-k_mag * th)


def fourier_multiplier_dipolar(k_mag, th):
    """
    Dipolar interaction kernel for magnetic thin films
    For thin films, the demagnetization factor depends on thickness th
    
    Physical form: σ(k) = 1 - exp(-|k|*th) for thin films
    This creates the characteristic labyrinth patterns
    """
    # Avoid numerical issues at k=0
    k_safe = torch.where(k_mag < 1e-12, torch.tensor(1e-12, dtype=dtype_real, device=device), k_mag)
    
    # Dipolar kernel for thin films
    sigma = 1.0 - torch.exp(-k_safe * th)
    
    # Handle k=0 case properly
    sigma = torch.where(k_mag < 1e-12, torch.tensor(0.0, dtype=dtype_real, device=device), sigma)
    
    return sigma