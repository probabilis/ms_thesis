import os
import argparse
import torch
import matplotlib.pyplot as plt

term_size = os.get_terminal_size()


def plotting_style():
    plt.style.use('classic')


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


def plotting_u_and_energy(u, energies):
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