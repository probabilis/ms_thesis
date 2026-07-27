import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from utils.env_utils import PATHS, read_sim_dat_from_csv, plotting_style, main_colormap, sub_colormap
from exp_data_processing.read import read_csv
from params.opt_params import labyrinth_data_params, exp_data_params, get_DataParameters
from exp_data_processing.postprocessing import quality_scores

from spectrum_analysis import radial_wavelength_spectrum
from exchange_length import reduced_exp_image_width



import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def plot_thickness_wedge(ax):
    """
    Draw schematic CoFeB thickness wedge:
    left  = 1.3 nm
    right = 1.4 nm
    """
    
    # Geometry in arbitrary plot units
    x0, x1 = 0.0, 10.0
    y_bottom = 0.0
    y_left_top = 0.8
    y_right_top = 1.35

    # Wedge polygon
    polygon = Polygon(
        [
            (x0, y_bottom),
            (x1, y_bottom),
            (x1, y_right_top),
            (x0, y_left_top),
        ],
        closed=True,
        facecolor="0.75",     # gray tone
        edgecolor="0.25",
        linewidth=2,
    )

    ax.add_patch(polygon)

    # Orange endpoint markers
    ax.scatter(
        [x0, x1],
        [y_left_top, y_right_top],
        s=180,
        color="orange",
        edgecolor="0.35",
        linewidth=1.5,
        zorder=5,
    )

    # Labels
    ax.text(
        x0 - 0.05,
        y_left_top + 0.15,
        r"CoFeB thickness $\delta$" "\n" r"$1.3~\mathrm{nm}$",
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )

    ax.text(
        x1 - 0.1,
        y_right_top - 0.25,
        r"$1.4~\mathrm{nm}$",
        ha="right",
        va="top",
        fontsize=18,
        fontweight="bold",
    )

    # Optional numbering like in your image
    ax.text(x0 - 0.15, y_left_top, "9", ha="right", va="center", fontsize=16)
    ax.text(x1, y_right_top + 0.12, "5", ha="center", va="bottom", fontsize=16)

    # Clean schematic style
    ax.set_xlim(-0.6, 10.4)
    ax.set_ylim(-0.2, 1.8)
    ax.set_aspect("auto")
    ax.axis("off")


def TaCoFeBMgO_stack_data01_image_series():
    plotting_style(CHANGE_FONT_SIZES=False)

    num_iters = 5_000
    dataset = "data_01"
    
    file_indices = [1, 2, 3, 4, 5, 6, 7]
    arrangement = [3, 2, 1, 5, 4, None, 6]
    FILE_AMOUNT = 6
    
    domain_pattern_lengths = np.zeros(FILE_AMOUNT)

    INPUT_PATH = PATHS.BASE_EXPDATA

    # ---------------------------------------------------------------

    self_picked_indices = [14, 11, 11, 7, 4, 1]
    self_picked_indices = [14, 11, 11, 11, 4, 1]

    fig, axs = plt.subplots( 5, FILE_AMOUNT, figsize = (16,10) )

    FIRST_RUN = False  
    SECOND_RUN = True

    THRESHOLD = 0.2


    for ii, OPT_FILE in enumerate( file_indices ):
        
        recording = f"00{OPT_FILE}"

        INPUT_FILE_PATH = INPUT_PATH / f"{dataset}/csv/mcd_slice_{recording}.csv"
        if FIRST_RUN:
            OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / "_old" / recording
        if SECOND_RUN:
            OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording

        df = pd.read_csv(OUTPUT_PATH / f"ranking.csv", index_col = 0)

        cut = df["perimeter"].quantile(0.6)
        df_filt = df[df["perimeter"] <= cut].copy()
        
        best = df_filt.sort_values(["fisherJ", "perimeter"], ascending = [False, True])


        _gamma = best.iloc[0]["gamma"]
        _lambda = best.iloc[0]["lambda"]

        if ii == 6:
            _gamma = df.iloc[1]["gamma"]
            _lambda = df.iloc[1]["lambda"]     
    
        u_exp = read_csv(INPUT_FILE_PATH, PLOT = False)
        df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, exp_data_params.N, num_iters, _gamma, exp_data_params.epsilon, _lambda)    

        if arrangement[ii] is not None:

            kk = arrangement[ii] - 1
            self_picked_new_index = self_picked_indices[kk]
            self_picked_gamma = df.iloc[self_picked_new_index]["gamma"]
            self_picked_lambda = df.iloc[self_picked_new_index]["lambda"]
            self_df_energies, self_u_sim = read_sim_dat_from_csv(OUTPUT_PATH, exp_data_params.N, num_iters, self_picked_gamma, exp_data_params.epsilon, self_picked_lambda)    

            new_index = arrangement[ii] - 1

            axs[0, new_index].imshow(u_exp, cmap = "gray", origin="lower", extent=(0,reduced_exp_image_width*1e6,0,reduced_exp_image_width*1e6))
            axs[0, new_index].set_xlabel("$\\mu\\mathrm{m}$")
            axs[0, new_index].set_ylabel("$\\mu\\mathrm{m}$")

            u_sim = torch.tensor(u_sim.values)
            axs[1, new_index].set_title(f"$\\gamma = {_gamma}$ \n $\lambda = {_lambda}$", fontsize = 10)
            axs[1, new_index].imshow(u_sim, cmap = main_colormap, origin="lower", extent=(0,1,0,1))
            
            
            self_u_sim = torch.tensor(self_u_sim.values)
            axs[2, new_index].imshow(self_u_sim, cmap = main_colormap, origin="lower", extent=(0,1,0,1))
            axs[2, new_index].set_title(f"$\\gamma = {self_picked_gamma}$ \n $\\lambda = {self_picked_lambda}$", fontsize = 10)


            fisherJ, perimeter, W = quality_scores(u_exp, self_u_sim)
            W_thick = torch.where(torch.abs(self_u_sim) > THRESHOLD, -1, self_u_sim)
            axs[3, new_index].imshow(W_thick.float(), cmap = "binary", origin="lower", extent=(0,1,0,1) )
            perimeter /= (u_exp.shape[0]**2) * 100
            axs[3, new_index].set_title(f"$S(u) = {fisherJ:.4g}$ \n $P(u)= {perimeter:.3g} \\%$", fontsize = 10)


            results = radial_wavelength_spectrum(self_u_sim, exp_data_params.gridsize/exp_data_params.N, plot = False)
            axs[4, new_index].loglog(results["k"].cpu(), results["profile"].cpu(), lw=2, color = "blue")
            axs[4, new_index].axvline(results["k_peak"], linestyle="--", label= "$k^*_\\mathrm{opt}$ =" + f"{results["k_peak"]:.4g}", color = "blue")
            

            results_raw = radial_wavelength_spectrum(u_exp, exp_data_params.gridsize/exp_data_params.N, plot = False)
            axs[4, new_index].loglog(results_raw["k"].cpu(), results_raw["profile"].cpu(), lw=2, color = "red")
            axs[4, new_index].axvline(results_raw["k_peak"], linestyle="--", label="$k^*_\\mathrm{raw}$ =" + f"{results_raw["k_peak"]:.4g}", color = "red")
            domain_pattern_length = results["wavelength_peak"]/2 * reduced_exp_image_width * 1e6
            axs[4, new_index].set_title(f"$D_p = {domain_pattern_length:.4g}$" + "$\\mu\\mathrm{m}$")
            axs[4, new_index].legend(fontsize = 10, loc = "lower right")
            axs[4, new_index].axes.get_yaxis().set_ticks([])
            axs[4, new_index].grid(color = "gray")
        
            for jj in range(4):
                axs[jj, new_index].set_box_aspect(1)
                if jj >= 1:
                    axs[jj, new_index].axes.get_xaxis().set_ticks([])
                    axs[jj, new_index].axes.get_yaxis().set_ticks([])


            domain_pattern_lengths[new_index] = domain_pattern_length
            kk += 1

    print(domain_pattern_lengths)


    fig.tight_layout()
    fig.savefig(PATHS.BASE_EXPDATA / dataset / "opt" / f"experimental_analysis.png" , dpi = 300)
    plt.show()




def TaCoFeBMgO_stack_reduced_image_series():


    plotting_style(CHANGE_FONT_SIZES=False)

    num_iters = 5_000
    dataset = "data_01"
    
    file_indices = [2, 5 , 4, 7]
    self_picked_indices = [11, 11, 4, 1]
    FILE_AMOUNT = len(file_indices)


    domain_pattern_lengths = np.zeros(FILE_AMOUNT)
    domain_pattern_lengths_unc = np.zeros(FILE_AMOUNT)
    INPUT_PATH = PATHS.BASE_EXPDATA
    
    THRESHOLD = 0.2
    # ---------------------------------------------------------------

    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=5,
        ncols=4,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 2.0],
    )


    ax_top = fig.add_subplot(gs[0, :])

    axs_mid = []
    for i in range(3):
        row_axes = []
        for j in range(4):
            ax = fig.add_subplot(gs[i + 1, j])
            row_axes.append(ax)
        axs_mid.append(row_axes)


    plot_thickness_wedge(ax_top)

    ax_bottom = fig.add_subplot(gs[4, :])
    axs = np.array(axs_mid)

    #fig, axs = plt.subplots( 3, FILE_AMOUNT, figsize = (12,8) )

    self_picked_gammas = []

    for ii, OPT_FILE in enumerate( file_indices ):
        
        recording = f"00{OPT_FILE}"
        INPUT_FILE_PATH = INPUT_PATH / f"{dataset}" / "csv"
        OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording
        df = pd.read_csv(OUTPUT_PATH / f"ranking.csv", index_col = 0)

        u_exp = read_csv(INPUT_FILE_PATH, recording, PLOT = False)

        self_picked_new_index = self_picked_indices[ii]
        self_picked_gamma = df.iloc[self_picked_new_index]["gamma"]
        self_picked_lambda = df.iloc[self_picked_new_index]["lambda"]
        _ , self_u_sim = read_sim_dat_from_csv(OUTPUT_PATH, exp_data_params.N, num_iters, self_picked_gamma, exp_data_params.epsilon, self_picked_lambda)    

        self_picked_gammas.append(self_picked_gamma)

        axs[0, ii].imshow(u_exp, cmap = "gray", origin="lower", extent=(0,reduced_exp_image_width*1e6,0,reduced_exp_image_width*1e6))
        axs[0, ii].set_xlabel("$\\mu\\mathrm{m}$")
        axs[0, ii].set_ylabel("$\\mu\\mathrm{m}$")

        
        self_u_sim = torch.tensor(self_u_sim.values)
        axs[1, ii].imshow(self_u_sim, cmap = main_colormap, origin="lower", extent=(0,1,0,1))
        axs[1, ii].set_title(f"$\\gamma = {self_picked_gamma}$ \n $\\lambda = {self_picked_lambda}$")

        results = radial_wavelength_spectrum(self_u_sim, exp_data_params.gridsize/exp_data_params.N, plot = False)

        domain_pattern_length = results["wavelength_peak"]/2 * reduced_exp_image_width * 1e6
        domain_pattern_length_unc = 1/(results["k_delta"]*2) * reduced_exp_image_width * 1e6 # uses standard deviation from radial intensity profile, sloppy but IDONT WAnT anymore xd


        domain_pattern_length_unc = np.round(domain_pattern_length_unc, 1) # round for 1 decimal
        if domain_pattern_length_unc < 1e-12:
            domain_pattern_length_unc = 0.1 # save 

        print("Dp", domain_pattern_length)
        print("Delta Dp", domain_pattern_length_unc)

        axs[2, ii].set_title(f"$D_p = ({domain_pattern_length:.2g} \pm {domain_pattern_length_unc:.1f})$" + "$\\mu\\mathrm{m}$")

        W_thick = torch.where(torch.abs(self_u_sim) > THRESHOLD, -1, self_u_sim)
        axs[2, ii].imshow(W_thick.float(), cmap = "binary", origin="lower", extent=(0,1,0,1) )

        for jj in range(3):
            axs[jj, ii].set_box_aspect(1)
            if jj >= 1:
                axs[jj, ii].axes.get_xaxis().set_ticks([])
                axs[jj, ii].axes.get_yaxis().set_ticks([])

        domain_pattern_lengths[ii] = domain_pattern_length
        domain_pattern_lengths_unc[ii] = domain_pattern_length_unc


    print(domain_pattern_lengths_unc)

    t_nm = np.array([1.30, 1.33, 1.36, 1.40])  # nm

    t = t_nm * 1e-9      # m
    Dp = domain_pattern_lengths * 1e-6    # m

    # initial guesses
    C0 = 1e-9       # m, prefactor scale
    B0 = 1e-12       # m, exponential length scale

    from scipy.optimize import curve_fit
    from domain_width_theory import domain_period_model # test script

    popt, pcov = curve_fit(
        domain_period_model,
        t,
        Dp,
        p0=[C0, B0],
        #maxfev=10000
    )

    C_fit, B_fit = popt
    C_err, B_err = np.sqrt(np.diag(pcov))

    print("Fit results:")
    print(f"C = {C_fit:.6e} ± {C_err:.6e} m")
    print(f"B = {B_fit:.6e} ± {B_err:.6e} m")

    print()
    print("Equivalent:")
    print(f"C = {C_fit * 1e9:.6f} nm")
    print(f"B = {B_fit * 1e9:.6f} nm")


    t_fit_nm = np.linspace(t_nm.min(), t_nm.max(), 300)
    t_fit = t_fit_nm * 1e-9
    Dp_fit = domain_period_model(t_fit, C_fit, B_fit)

    print(domain_pattern_lengths)
    print(domain_pattern_lengths_unc) 

    #ax_bottom.set_yscale("log")
    
    ax_bottom.plot(t_fit_nm, Dp_fit * 1e6, label="fit", linewidth = 2)
    ax_bottom.errorbar(t_nm, domain_pattern_lengths, yerr=domain_pattern_lengths_unc, label='data', color = "red", fmt=' ', elinewidth=2)
    
    #ax_bottom.scatter(t_nm, Dp_um, label="data", zorder=5, color = "black")
    
    ax_bottom.set_xlim(1.29, 1.41)
    ax_bottom.set_xlabel(r"$\delta~[\mathrm{nm}]$")
    ax_bottom.set_ylabel(r"$D_\mathrm{P}~[\mu\mathrm{m}]$")
    ax_bottom.set_title(r"Fit of $D_\mathrm{P}(\delta) = C \exp(B/\delta)$")
    ax_bottom.legend()
    ax_bottom.grid(color = "gray")

    fig.tight_layout()
    fig.savefig(PATHS.BASE_EXPDATA / dataset / "opt" / f"experimental_analysis_reduced_analysis.png" , dpi = 300)
    plt.show()


if __name__ == "__main__":
    #TaCoFeBMgO_stack_data01_image_series()
    TaCoFeBMgO_stack_reduced_image_series()
    
    