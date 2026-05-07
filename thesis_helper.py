import torch
import matplotlib.pyplot as plt
import math

from pattern_formation import double_well_potential, fourier_multiplier
from env_utils import plotting_style, PATHS


def plot_double_well(OUT_PATH):

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


    # plt.plot(k_ls, lambda_ls)
    #plt.title("Nesterov Momentum Paramter $\\beta_n$")
    plt.plot(k_ls, beta_ls, label = "$\\beta_n(\\lambda_n)$", linewidth = 2)
    plt.plot(k_ls, beta_pock_ls, label = "$\\beta_n = \\frac{n - 1}{n + 2}$", linewidth = 2)
    plt.hlines(1, k_ls[0], k_ls[-1], color = "gray")
    plt.ylim(0, 1.1)
    plt.grid(color = "gray")
    plt.xlabel("iterator $n$")
    plt.ylabel("$\\beta_n$")
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.savefig(OUT_PATH / "nesterov_momentum_parameter.png", dpi = 300)
    plt.show()



def plot_fourier_multiplier(OUT_PATH):

    th = 0.1

    plt.figure(figsize = (6,4) )

    x = torch.arange(-10,+10, 1/100)
    y = fourier_multiplier(x)

    plt.vlines(0, y.min()-0.1,y.max(), color = "gray")
    plt.hlines(0, x.min(), x.max(), color = "gray")


    plt.plot(x, y, linewidth = 2, label = "$\\sigma(k)$")
    plt.grid(color = "gray")
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


def gradient():

    N = 128

    u = torch.rand((N,N))

    ones = torch.ones_like(u)

    u = u + torch.sin(ones)

    fig, axs = plt.subplots(1,4)

    ux = uy = u

    plt.ion()

    for ii in range(100):
        print("ii", ii)
        ux = u - torch.roll(u, 1, 0)
        uy = u - torch.roll(u, 1, 1)

        u_grad = torch.sqrt(ux*ux + uy*uy)

        axs[0].imshow(u)
        axs[1].imshow(ux)
        axs[2].imshow(uy)
        axs[3].imshow(u_grad)

        u = u_grad

        plt.pause(1)

    plt.ioff()
    #plt.show()




if __name__ == "__main__":

    OUT_PATH = PATHS.PATH_THESIS

    plotting_style()
    
    plot_double_well(OUT_PATH)
    plot_nesterov_momentum_paramter(OUT_PATH)
    plot_fourier_multiplier(OUT_PATH)
    #plot_fourier_multiplier_thickness_loop(OUT_PATH)