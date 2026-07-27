import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from utils.env_utils import PATHS, print_bars, get_args, plotting_style, main_colormap, sub_colormap
from utils.pattern_formation import initialize_u0_random
from params.opt_params import labyrinth_data_params, get_DataParameters, get_SimulationParamters, gd_sim_params, sim_config

from optimization.gradient_descent import gradient_descent
from spectrum_analysis import radial_wavelength_spectrum

# ---------------------------------------------------------------
"""
script for time-evolution of plain gradient descent method with some analysis plots
"""


def plot_pdf_heatmap(u_ls, energies, xlim=(-1.1, 1.1), n_x=200):

    fig, axs = plt.subplots(2,1, figsize=(10, 8), sharex=True)

    x = np.linspace(xlim[0], xlim[1], n_x)
    pdfs = []

    for u in u_ls:
        u = np.asarray(u).ravel()
        kde = gaussian_kde(u)
        pdfs.append(kde(x))

    pdfs = np.array(pdfs)   # shape: (time, x)

    pdfs = pdfs.T

    im = axs[0].imshow(
        pdfs,
        aspect="auto",
        origin="lower",
        extent=[0, len(u_ls)-1, xlim[0], xlim[1]],
        cmap = main_colormap,
    )

    #fig.colorbar(im,ax=axs[0], label=r"$\\mathcal{N}(u)$")
    axs[0].set_ylabel("$\\mathcal{N}[u^{(ij)}_n]$")
    axs[1].set_ylabel("$E[u_n(x,y)]$")
    axs[1].plot(history["E_grad"], label="$E_{\\nabla}$")
    axs[1].plot(history["E_dw"], label="$E_{W}$")
    axs[1].plot(history["E_fm"], label="$E_{\\mathcal{F}}$")
    axs[1].set_xlabel("iterator $n$")

    axs[1].set_yscale('log')
    axs[1].grid(color = "gray")
    axs[1].legend()
    fig.tight_layout()


def plot_pdf_waterfall(u_ls, NR_OF_ACCUMULATED_ITERATIONS, step=20, offset=0.25, xlim=(-1.1, 1.1)):
    x = np.linspace(xlim[0], xlim[1], 300)

    colors = plt.cm.berlin(np.linspace(0,1,len(u_ls))) #this gets the colormap as an array of colours

    plt.figure(figsize=(10, 8) )

    for k, u in enumerate(u_ls[::step]):
        u = np.asarray(u).ravel()
        kde = gaussian_kde(u)
        p = kde(x)

        y0 = k * offset
        plt.plot(x, p + y0, lw=2, label=f"{k*NR_OF_ACCUMULATED_ITERATIONS}", color = colors[k])
        plt.fill_between(x, y0, p + y0, alpha=0.2, color = "gray")


    plt.grid(color = "gray")
    plt.xlabel(r"$u^{(ij)}_n$")
    plt.ylabel(f"normalized PDF $p(u_n)$ shifted by $\\Delta = {int(offset)}$ after each $n = {NR_OF_ACCUMULATED_ITERATIONS}$ iterations")
    plt.title("probability density function PDF time evolution")
    plt.legend(title="iterator $n$", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xlim(*xlim)
    plt.tight_layout()

# ---------------------------------------------------------------

if __name__ == "__main__":

    plotting_style()
    FOLDER_PATH = PATHS.PATH_EVOLUTION

    # parameter definition 
    # ---------------------------------------------------------------

    LIVE_PLOT = False
    DATA_LOG = False
    
    N = 200
    num_iters_max = 900
    gamma = 0.002

    labyrinth_data_params = replace(labyrinth_data_params, N = N, gamma = gamma)
    gd_sim_params = replace(gd_sim_params, num_iters = num_iters_max)
    sim_config = replace(sim_config, STOP_BY_TOL = False)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)

    u0 = initialize_u0_random(N, REAL = True)
    
    print_bars()
    print(labyrinth_data_params)
    print(gd_sim_params)
    print(sim_config)
    print_bars()    

    SHOW_PLOT = True

    PDF_LINE_PLOT = False # not used in thesis
    PDF_HEATMAP_PLOT = False
    SUMMARY = True

    if PDF_LINE_PLOT:
        NR_OF_ACCUMULATED_ITERATIONS = 100
        u_ls, history = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, 
                                        **asdict(labyrinth_data_params), 
                                        **asdict(gd_sim_params), 
                                        **asdict(sim_config), 
                                        SAVE_U_HISTORY = NR_OF_ACCUMULATED_ITERATIONS)

        plot_pdf_waterfall(u_ls, NR_OF_ACCUMULATED_ITERATIONS, step=1, offset = 1.0)
        
        plt.savefig(FOLDER_PATH / f"pdf_evolution_N={N}_num-iters={num_iters_max}_gamma={gamma}.png", dpi = 300)
        
        if SHOW_PLOT:
            plt.show()
            plt.close()

    if PDF_HEATMAP_PLOT:
        u_ls, history = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, 
                                        **asdict(labyrinth_data_params), 
                                        **asdict(gd_sim_params), 
                                        **asdict(sim_config), 
                                        SAVE_U_HISTORY = 1)
        
        plot_pdf_heatmap(u_ls, history)
        plt.savefig(FOLDER_PATH / f"pdf_evolution_heatmap_N={N}_num-iters={num_iters_max}_gamma={gamma}.png", dpi = 300)
        
        if SHOW_PLOT:
            plt.show()
            plt.close()

    if SUMMARY:
        u_ls, history = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(gd_sim_params), **asdict(sim_config), SAVE_U_HISTORY=100)    
        
        fig, axs = plt.subplots( 5, 4, figsize = (10,10))

        axs = axs.ravel()

        for ii, u in enumerate( u_ls ):

            axs[2*ii].imshow(u.cpu().numpy(), cmap=main_colormap, extent=(0,1,0,1))
            #axs[2*ii].set_box_aspect(1)
            axs[2*ii].axes.get_xaxis().set_ticks([])
            axs[2*ii].axes.get_yaxis().set_ticks([])
            axs[2*ii].set_title(rf"$u_{{n={(ii) * 100 }}}(x,y)$")

            counts, bins = np.histogram(u_ls[ii])
            axs[2*ii+1].hist(bins[:-1], bins, weights=counts, color = "gray", density = True)
        
            axs[2*ii+1].set_ylabel("$p(u_n)$")
            axs[2*ii+1].grid(color = "gray")

            ymin, ymax = axs[2*ii+1].get_ylim()
            xmin, xmax = axs[2*ii+1].get_xlim()
            axs[2*ii+1].set_yticks(np.round(np.linspace(ymin, ymax, 4), 2))
            axs[2*ii+1].set_xticks(np.round(np.linspace(xmin, xmax, 4), 2))


        axs[17].set_xlabel("$u^{(ij)}_n$")
        axs[19].set_xlabel("$u^{(ij)}_n$")

        fig.tight_layout()
        plt.savefig(FOLDER_PATH / f"domain_evolution_N={N}_num-iters={num_iters_max}_gamma={gamma}.png", dpi = 300)
        
        if SHOW_PLOT:
            plt.show()
            plt.close()

        fig, axs = plt.subplots(2, 2, figsize = (10,10) )

        im0 = axs[0, 0].imshow(u_ls[-1].cpu().numpy(), cmap=main_colormap, origin="lower", extent=(0,1,0,1) )    

        axs[0, 0].set_box_aspect(1)
        axs[0, 0].set_xlabel("$x$")
        axs[0, 0].set_ylabel("$y$")
        axs[0, 0].set_title(rf"$u_{{n={num_iters_max}}}(x,y)$")

        results = radial_wavelength_spectrum(u, gridsize/N)
        im1 = axs[0, 1].imshow(torch.log1p(results["S"]).cpu(), cmap=sub_colormap, origin="lower", extent=(-N//2,N//2,-N//2,N//2))
        
        axs[0, 1].set_box_aspect(1)
        axs[0, 1].set_xlabel("$k_x$")
        axs[0, 1].set_ylabel("$k_y$")

        axs[0, 1].set_title(rf"$\mathcal{{F}}[u_{{n={num_iters_max}}}(x,y)]$")

        axs[1, 0].loglog(history["E_total"], label = "$E_{total}$")
        axs[1, 0].legend(loc = "lower right")

        axs[1, 1].loglog(history["E_grad"], label="$E_{\\nabla}$")
        axs[1, 1].loglog(history["E_dw"], label="$E_{W}$")
        axs[1, 1].loglog(history["E_fm"], label="$E_{\\mathcal{F}}$")
        axs[1, 1].legend(loc = "lower right")

        for ii in range(2):
            axs[1, ii].set_xlabel("iterator $n$")
            axs[1, ii].set_ylabel("$E[u_n(x,y)]$")
            axs[1, ii].grid(color = "gray")
            axs[1, ii].grid(color = "gray")

        fig.tight_layout()
        plt.savefig(FOLDER_PATH / f"domain_evolution_summary_N={N}_num-iters={num_iters_max}_gamma={gamma}.png", dpi = 300)
        
        if SHOW_PLOT:
            plt.show()