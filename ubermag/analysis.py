import pandas as pd 
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch

import sys
sys.path.insert(0,'')
from spectrum_analysis import radial_wavelength_spectrum
from utils.env_utils import plotting_style, main_colormap, sub_colormap



def load_field_csv(csv_path):
    """
    Reads the CSV written by save_field_csv().

    Returns:
        x (2D array) [m]
        y (2D array) [m]
        mx, my, mz (2D arrays)
    """

    data = []

    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue  # skip metadata
            if line.strip() == "":
                continue
            data.append(line.strip().split(","))

    # first row is header
    header = data[0]
    rows = data[1:]

    arr = np.array(rows, dtype=float)

    i = arr[:, 0].astype(int)
    j = arr[:, 1].astype(int)
    x = arr[:, 2]
    y = arr[:, 3]
    mx = arr[:, 4]
    my = arr[:, 5]
    mz = arr[:, 6]

    # infer grid size
    nx = i.max() + 1
    ny = j.max() + 1

    # reshape
    x2d = np.zeros((nx, ny))
    y2d = np.zeros((nx, ny))
    mx2d = np.zeros((nx, ny))
    my2d = np.zeros((nx, ny))
    mz2d = np.zeros((nx, ny))

    for k in range(len(i)):
        x2d[i[k], j[k]] = x[k]
        y2d[i[k], j[k]] = y[k]
        mx2d[i[k], j[k]] = mx[k]
        my2d[i[k], j[k]] = my[k]
        mz2d[i[k], j[k]] = mz[k]

    return x2d, y2d, mx2d, my2d, mz2d





def plot_mz_from_csv(csv_path, title=None):
    x, y, _, _, mz = load_field_csv(csv_path)

    extent = [
        x.min() * 1e6,
        x.max() * 1e6,
        y.min() * 1e6,
        y.max() * 1e6,
    ]

    plt.figure(figsize=(7, 6))
    plt.imshow(
        mz.T,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    plt.colorbar(label=r"$m_z$")
    plt.xlabel(r"$x\;[\mu m]$")
    plt.ylabel(r"$y\;[\mu m]$")

    if title is None:
        title = "Loaded m_z from CSV"

    plt.title(title)
    plt.tight_layout()
    plt.show()





from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def plot_oommf_grid_3x9(
    summary_csv,
    base_dir=None,
    save_path=None,
    compute_wavelength=True,
):
    """
    Plot OOMMF grid runs as 3 x 9 figure.

    Rows:
        increasing thickness t_nm

    Columns:
        clustered by A_pJ_per_m, then increasing Keff_kJ_per_m3

    Requires:
        load_field_csv(csv_path)
        radial_wavelength_spectrum(u, dx)
    """

    summary_csv = Path(summary_csv)
    df = pd.read_csv(summary_csv)

    # Keep only successful runs
    if "status" in df.columns:
        df = df[df["status"].eq("ok")].copy()

    # Resolve paths
    if base_dir is None:
        base_dir = summary_csv.parent
    else:
        base_dir = Path(base_dir)

    def resolve_path(p):
        p = Path(p)
        if p.is_absolute():
            return p

        if p.exists():
            return p

        p2 = summary_csv.parent / p
        if p2.exists():
            return p2

        p3 = base_dir / p.name
        return p3

    df["csv_path_resolved"] = df["csv_path"].apply(resolve_path)

    # Sort parameter axes
    thickness_values = sorted(df["t_nm"].unique())
    A_values = sorted(df["A_pJ_per_m"].unique())
    Keff_values = sorted(df["Keff_kJ_per_m3"].unique())

    n_rows = len(thickness_values)
    n_cols = len(A_values) * len(Keff_values)

    if n_rows != 3 or n_cols != 9:
        print(f"Warning: expected 3x9, got {n_rows}x{n_cols}")

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.25 * n_cols, 2.55 * n_rows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    if n_rows == 1:
        axs = axs[None, :]
    if n_cols == 1:
        axs = axs[:, None]

    wavelength_records = []

    for row, t_nm in enumerate(thickness_values):
        for a_idx, A_pJ in enumerate(A_values):
            for k_idx, Keff_kJ in enumerate(Keff_values):

                col = a_idx * len(Keff_values) + k_idx
                ax = axs[row, col]

                sub = df[
                    np.isclose(df["t_nm"], t_nm)
                    & np.isclose(df["A_pJ_per_m"], A_pJ)
                    & np.isclose(df["Keff_kJ_per_m3"], Keff_kJ)
                ]

                if len(sub) == 0:
                    ax.axis("off")
                    ax.set_title("missing")
                    continue

                if len(sub) > 1:
                    print(
                        f"Warning: multiple runs for "
                        f"t={t_nm}, A={A_pJ}, Keff={Keff_kJ}; using first."
                    )

                run = sub.iloc[0]
                csv_path = run["csv_path_resolved"]

                x, y, _, _, mz = load_field_csv(csv_path)

                extent = [
                    x.min() * 1e6,
                    x.max() * 1e6,
                    y.min() * 1e6,
                    y.max() * 1e6,
                ]

                ax.imshow(
                    mz.T,
                    origin="lower",
                    extent=extent,
                    cmap="gray",
                    vmin=-1,
                    vmax=1,
                    interpolation="nearest",
                )

                ax.set_xticks([])
                ax.set_yticks([])

                # Column titles only on top row
                if row == 0:
                    ax.set_title(
                        rf"$A={A_pJ:.0f}$ pJ/m" "\n"
                        rf"$K_{{eff}}={Keff_kJ:.0f}$ kJ/m$^3$",
                        fontsize=9,
                    )

                # Row labels on left side
                if col == 0:
                    ax.set_ylabel(
                        rf"$\delta={t_nm:.1f}$ nm",
                        fontsize=11,
                        rotation=90,
                    )

                # Optional wavelength analysis
                wavelength = np.nan

                if compute_wavelength:
                    u = torch.from_numpy(mz.T.astype(np.float64))

                    # Your old code used dx = 1 / N.
                    # This gives wavelength in normalized domain units.
                    dx_norm = 1.0 / u.shape[0]

                    results = radial_wavelength_spectrum(u, dx_norm)

                    # Keeping your previous division by 2.
                    wavelength = results["wavelength_peak"] / 2.0

                    ax.text(
                        0.03,
                        0.04,
                        rf"$\lambda={wavelength:.3f}$",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="white",
                        bbox=dict(
                            facecolor="black",
                            alpha=0.55,
                            edgecolor="none",
                            pad=2,
                        ),
                    )

                wavelength_records.append(
                    {
                        "t_nm": t_nm,
                        "A_pJ_per_m": A_pJ,
                        "Keff_kJ_per_m3": Keff_kJ,
                        "wavelength": wavelength,
                        "csv_path": str(csv_path),
                    }
                )

    fig.suptitle(
        r"OOMMF grid sweep: thickness rows, clustered $A$ and $K_{eff}$ columns",
        fontsize=15,
    )

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()

    wavelength_df = pd.DataFrame(wavelength_records)

    return fig, axs, wavelength_df

def test():
    DIR_PATH = Path("oommf") / "oommf_grid_runs3"
    df = pd.read_csv(DIR_PATH / "grid_summary.csv")

    col_thickness = "t_nm"
    col1 = "A_J_per_m"

    _df = df.sort_values(by=col_thickness, ascending=True)

    #data_paths = _df["csv_path"].iloc[4:8].sort_values(ascending=True)
    data_paths = _df["csv_path"].iloc[:].sort_values(ascending=True)

    print(data_paths)

    #titles = _df["t_nm"].iloc[0:4].to_list()
    titles = _df["t_nm"].iloc[:].sort_values(ascending=True).to_list()

    print(_df)
    exit(0)

    fig, axs = plt.subplots( 3, 9, figsize = (20,16))
    wavelengths = []

    axs = axs.ravel()

    for ii, csv_path in enumerate(data_paths):

        x, y, _, _, mz = load_field_csv(csv_path)

        extent = [
            x.min() * 1e6,
            x.max() * 1e6,
            y.min() * 1e6,
            y.max() * 1e6,
        ]
        
        axs[ii].imshow(
            mz.T,
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        #plt.colorbar(label=r"$m_z$")
        #plt.xlabel(r"$x\;[\mu m]$")
        #plt.ylabel(r"$y\;[\mu m]$")
        axs[ii].set_title(f"$\\delta = {titles[ii]}$nm")

        u = torch.from_numpy(mz.T)
        #F = torch.fft.fft2(u, norm="ortho")
        #Fshift = torch.fft.fftshift(F)

        results = radial_wavelength_spectrum(u, 1/u.shape[0])
        wavelength = results['wavelength_peak'] / 2
        wavelengths.append(wavelength)

        #axs[1, ii].imshow(torch.abs(Fshift), origin="lower")
        #axs[1, ii].set_title(f"{wavelength}")

    
    plt.tight_layout()
    plt.show()

    thickness = [float(x) for x in titles]
    plt.figure()
    plt.scatter(thickness, wavelengths)
    plt.xlabel("thickness $\\delta$")
    plt.ylabel("pattern length $\\lambda$")
    plt.show()



if __name__ == "__main__":
    SUMMARY_PATH = Path("oommf") / "oommf_grid_runs3" / "grid_summary.csv"
    
    plot_oommf_grid_3x9(SUMMARY_PATH)