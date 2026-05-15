import torch
import matplotlib.pyplot as plt
import math

from utils.pattern_formation import double_well_potential, fourier_multiplier
from utils.env_utils import plotting_style, PATHS


def plot_double_well(OUT_PATH):
    """
    plotting normalized double well potential
    """
    plt.figure(figsize = (6,4) )

    x = torch.arange(-2, +2, 0.01)
    y = double_well_potential(x, 9/32)
    

    plt.vlines(0, y.min()-0.1,y.max(), color = "gray")
    plt.hlines(0, x.min(), x.max(), color = "gray")

    plt.plot(x, y, label = "$W(u) = c_0 (1 - u^2)^2$", linewidth = 2)
    
    plt.ylim(-0.1, y.max())
    #plt.title("Double Well function $W(u)$")
    plt.xlabel("$u$")
    plt.ylabel("$W(u)$")
    plt.legend()
    plt.grid(color = "gray")
    plt.tight_layout()
    plt.savefig(OUT_PATH / "double_well.png", dpi = 300)
    plt.show()



def plot_nesterov_momentum_paramter(OUT_PATH):    
    """
    plotting nesterov momentum paramter and approximation used for nesterov acceleration
    """
    plt.figure(figsize = (6,4) )

    lambda_km1 = 1
    beta_k = 0

    lambda_ls = []
    beta_ls = []
    beta_pock_ls = []
    k_ls = []

    for k in range(1, 200):
        lambda_k = (1 + math.sqrt(1 + lambda_km1**2 * 4)) / 2
        beta_k = (lambda_km1 - 1) / lambda_k
        lambda_km1 = lambda_k

        beta_k_pock = (k - 1) / (k + 2)

        lambda_ls.append(lambda_k)
        beta_ls.append(beta_k)
        beta_pock_ls.append(beta_k_pock)
        k_ls.append(k)

    print(beta_ls)

    # plt.plot(k_ls, lambda_ls)
    #plt.title("Nesterov Momentum Paramter $\\beta_n$")
    plt.plot(k_ls, beta_ls, label = "$\\beta_k(\\lambda_k)$", linewidth = 2)
    plt.plot(k_ls, beta_pock_ls, label = "$\\beta_k = \\frac{k - 1}{k + 2}$", linewidth = 2)
    plt.hlines(1, k_ls[0], k_ls[-1], color = "gray")
    #plt.ylim(0, 1.1)
    plt.xscale("log")
    plt.grid(color = "gray")
    plt.xlabel("iterator $k$")
    plt.ylabel("$\\beta_k$")
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.savefig(OUT_PATH / "nesterov_momentum_parameter.png", dpi = 300)
    plt.show()



def plot_fourier_multiplier(OUT_PATH):
    """
    plotting used fourier multiplier for dipolar interaction energy 
    """
    th = 0.1

    plt.figure(figsize = (6,4) )

    x = torch.arange(-10,+10, 1/100)
    y = fourier_multiplier(x)

    plt.vlines(0, y.min()-0.1,y.max(), color = "gray")
    plt.hlines(0, x.min(), x.max(), color = "gray")


    plt.plot(x, y, linewidth = 2, label = "$\\sigma(k)$")
    plt.xlabel("wavevector $k$")
    plt.ylabel("$\\sigma(k)$")
    plt.ylim(-0.1, y.max())
    #plt.title("Fourier Multiplier $\\sigma(k)$")
    plt.legend()
    plt.grid(color = "gray")
    plt.tight_layout()
    plt.savefig(OUT_PATH / "fourier_multiplier.png", dpi = 300)
    plt.show()


def plot_fourier_multiplier_thickness_loop(OUT_PATH):
    """
    plotting used fourier multiplier for dipolar interaction energy over thickness regime (delta)
    """
    plt.figure(figsize = (6,4) )

    th_ls = [0.1, 1.0, 10.0]

    for th in th_ls:

        x = torch.arange(-10,+10, 1/100)
        y = fourier_multiplier(th * x)

        plt.plot(x, y, linewidth = 2, label = f"$\\sigma(\\delta \\cdot k)$ with $\\delta = {th}$")


    plt.xlabel("wavevector $k$")
    plt.ylabel("$\\sigma(k)$")
    plt.ylim(-0.1, y.max())
    plt.title("Fourier Multiplier $\\sigma(k)$")
    plt.legend()
    plt.grid(color = "gray")    

    plt.vlines(0, y.min()-0.1,y.max(), color = "gray")
    plt.hlines(0, x.min(), x.max(), color = "gray")

    plt.tight_layout()
    plt.savefig(OUT_PATH / "fourier_multiplier_with_thickness.png", dpi = 300)
    plt.show()




def plot_bloch_wall_transition_eps(OUT_PATH):
    """
    plotting bloch wall transition via
    function used from condette thesis / p.27
    """
    
    def profile(x, eps):
        return torch.tanh(x / eps)

    plt.figure(figsize = (6,4) )

    x = torch.arange(-1,+1, 1/100)
    
    eps_ls = [0.01, 0.1, 0.5, 1.0]

    for eps in eps_ls:
        
        y = profile(x, eps)
        #plt.vlines(0, y.min()-0.1,y.max(), color = "gray")
        #plt.hlines(0, x.min(), x.max(), color = "gray")
        plt.plot(x, y, linewidth = 2, label = f"$\\varepsilon = {eps}$")


    plt.xlabel("$x$")
    plt.ylabel("$m_z$")
    plt.ylim(-1.1, 1.1)
    
    plt.legend()
    plt.grid(color = "gray")
    plt.tight_layout()
    plt.savefig(OUT_PATH / "bloch_wall_transition.png", dpi = 300)
    plt.show()



if __name__ == "__main__":

    OUT_PATH = PATHS.PATH_THESIS

    plotting_style()
    
    #plot_double_well(OUT_PATH)
    plot_nesterov_momentum_paramter(OUT_PATH)
    #plot_fourier_multiplier(OUT_PATH)
    #plot_fourier_multiplier_thickness_loop(OUT_PATH)
    #plot_bloch_wall_transition_eps(OUT_PATH)