import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from utils.env_utils import PATHS, print_bars, plotting_style, log_data, term_size, main_colormap, sub_colormap
from utils.pattern_formation import initialize_u0_random

from params.opt_params import labyrinth_data_params, get_DataParameters, get_SimulationParamters, sim_config
from params.opt_params import pgd_sim_params as ngd_sim_params
from params.lipschitz import evaluate_lipschitz_constant
from optimization.gd_nesterov import gradient_descent_nesterov





def radial_wavelength_spectrum(
    u: torch.Tensor,
    dx: float = 1.0,
    use_power: bool = True,
    nbins: int | None = None,
    remove_mean: bool = True,
    eps: float = 1e-12,
    plot: bool = False,
):
    """
    Compute radial average of a 2D FFT spectrum and convert frequency to wavelength
    """

    if u.ndim != 2:
        raise ValueError("u must be a 2D tensor")

    u = u.detach().float()
    Nx, Ny = u.shape
    device = u.device

    if remove_mean:
        u = u - u.mean()

    ftu = torch.fft.fft2(u, norm="ortho")
    Fshift = torch.fft.fftshift(ftu)

    # Spectrum
    if use_power:
        S = torch.abs(Fshift) ** 2
    else:
        S = torch.abs(Fshift)

    # Frequency coordinates (cycles per unit length)
    fx = torch.fft.fftshift(torch.fft.fftfreq(Nx, d=dx)).to(device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(Ny, d=dx)).to(device)

    FX, FY = torch.meshgrid(fx, fy, indexing="ij")
    KR = torch.sqrt(FX**2 + FY**2)   # radial frequency

    # Radial bins
    k_max = KR.max().item()
    if nbins is None:
        nbins = min(Nx, Ny) // 2

    bin_edges = torch.linspace(0.0, k_max, nbins + 1, device=device)
    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    profile = torch.zeros(nbins, device=device)
    counts = torch.zeros(nbins, device=device)

    # Bin by radius
    for i in range(nbins):
        mask = (KR >= bin_edges[i]) & (KR < bin_edges[i + 1])
        c = mask.sum()
        if c > 0:
            profile[i] = S[mask].mean()
            counts[i] = c

    # Ignore zero-frequency / DC bin when searching for characteristic scale
    valid = (k_centers > eps) & (counts > 0)
    if valid.sum() == 0:
        raise ValueError("No valid nonzero radial frequency bins found")

    k_valid = k_centers[valid]
    p_valid = profile[valid]

    peak_idx = torch.argmax(p_valid)
    k_peak = k_valid[peak_idx].item()
    wavelength_peak = 1.0 / k_peak

    wavelength = torch.full_like(k_centers, float("inf"))
    nonzero = k_centers > eps
    wavelength[nonzero] = 1.0 / k_centers[nonzero]

    results = {
        "k": k_centers.cpu(),
        "wavelength": wavelength.cpu(),
        "profile": profile.cpu(),
        "k_peak": k_peak,
        "wavelength_peak": wavelength_peak,
        "Fshift": Fshift.cpu(),
        "S": S.cpu(),
    }


    if plot:
        pass

    return results





if __name__ == "__main__":
    
    plotting_style()
    FOLDER_PATH = PATHS.PATH_PARAMS_STUDY

    LIVE_PLOT = False
    DATA_LOG = False

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)
    N = 100

    ngd_sim_params = replace(ngd_sim_params, num_iters = 10_000, tau = evaluate_lipschitz_constant(gamma, epsilon, N, gridsize) )
    labyrinth_data_params = replace(labyrinth_data_params, N = N, gamma = 0.002)

    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print(sim_config)
    print_bars()

    u0 = initialize_u0_random(N, REAL = True)

    u, energies = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params), **asdict(ngd_sim_params), **asdict(sim_config))
    results = radial_wavelength_spectrum(u, gridsize/N)
    num_iters_max = len(energies)

    SINGLE_RUN = True
    FREQUENCY_SWEEP = False

    if SINGLE_RUN:
        fig, axs = plt.subplots(1, 3, figsize = (10,6))
        im0 = axs[0].imshow(u.cpu(), cmap=main_colormap, origin="lower", extent=(0,1,0,1) )
        axs[0].set_title(rf"$u_{{n={num_iters_max}}}(x,y)$")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        im1 = axs[1].imshow(torch.log1p(results["S"]).cpu(), cmap=sub_colormap, origin="lower", extent=(-N//2,N//2,-N//2,N//2)) # 
        axs[1].set_title("$\\mathrm{log}(1 + \\hat{u}^{shift}_n)$")
        #axs[1].set_title(rf"$\mathcal{{F}}[u_{{n={num_iters_max}}}(x,y)]$")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        axs[2].loglog(results["k"], results["profile"], lw=2)
        k_peak = results["k_peak"]
        wavelength_peak = results["wavelength_peak"]

        axs[2].axvline(k_peak, linestyle="--", label=f"$k^* \\approx {k_peak:.3g}$")
        axs[2].set_xlabel("$k$")
        axs[2].set_ylabel("Radial mean intensity")
        
        #axs[2].set_title(f"$\\lambda^* \\approx {wavelength_peak:.3g}$")
        #axs[2].set_xscale("log")
        axs[2].legend(loc = "lower left")
        axs[2].grid("gray")
        plt.tight_layout()
        plt.savefig(FOLDER_PATH / "spectrum_analysis.png", dpi = 300)
        plt.show()


    if FREQUENCY_SWEEP:
        gamma_ls = np.linspace(0.02, 0.0003, 20)

        N_est = 1
        frequencies = torch.zeros(N_est, len(gamma_ls))

        for ii in range(N_est):
            values = []
            for gamma in gamma_ls:
                print(term_size.columns * "-")
                print("gamma:", gamma)
                
                eta = evaluate_lipschitz_constant(gamma, epsilon, N, gridsize)  
                u0 = initialize_u0_random(N, REAL = True)

                labyrinth_data_params = replace(labyrinth_data_params, N = N, gamma = gamma)
                ngd_sim_params = replace(ngd_sim_params, tau = eta)
                
                u, energies = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params), **asdict(ngd_sim_params), **asdict(sim_config))
                results = radial_wavelength_spectrum(u, gridsize/N)
                num_iters_max = len(energies)

                print(f"Pattern frequency [cycles per unit length]: {results["k_peak"]}, wavelength [unit length]: {results["wavelength_peak"]}")
                values.append(results["k_peak"])

            frequencies[ii] = torch.tensor(values)

        mean_frequencies = torch.mean(frequencies, dim = 0)

        if len(gamma_ls) > 1:
            plt.figure() 
            plt.title("Characteristic Radial spatial frequency $k$ [cycles / unit length] of spectrum")
            plt.xlabel(r"Gamma $\\gamma$")
            plt.ylabel("Radial spatial frequency $k$ [cycles / unit length]")
            plt.plot(gamma_ls, values)
            plt.grid(color = "gray")
            plt.savefig(FOLDER_PATH / "fourier_frequencies.png", dpi = 300)     
            plt.show()