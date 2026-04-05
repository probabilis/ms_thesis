import pandas as pd 
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch

import sys
sys.path.insert(0,'')
from spectrum_analysis import radial_wavelength_spectrum

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




if __name__ == "__main__":
    DIR_PATH = Path("oommf") / "oommf_grid_runs1"
    df = pd.read_csv(DIR_PATH / "grid_summary.csv")

    col_thickness = "t_nm"
    col1 = "A_J_per_m"

    _df = df.sort_values(by=col1, ascending=True)

    data_paths = _df["csv_path"].iloc[4:8].sort_values(ascending=True)
    print(data_paths)
    #print(_df)
    #exit(0)

    titles = _df["t_nm"].iloc[0:4].to_list()

    fig, axs = plt.subplots(2,4)

    wavelengths = []

    for ii, csv_path in enumerate(data_paths):

        x, y, _, _, mz = load_field_csv(csv_path)

        extent = [
            x.min() * 1e6,
            x.max() * 1e6,
            y.min() * 1e6,
            y.max() * 1e6,
        ]

        
        axs[0, ii].imshow(
            mz.T,
            origin="lower",
            extent=extent,
            #cmap="gray",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        #plt.colorbar(label=r"$m_z$")
        #plt.xlabel(r"$x\;[\mu m]$")
        #plt.ylabel(r"$y\;[\mu m]$")
        axs[0, ii].set_title(f"{titles[ii]}")

        u = torch.from_numpy(mz.T)
        F = torch.fft.fft2(u, norm="ortho")
        Fshift = torch.fft.fftshift(F)

        results = radial_wavelength_spectrum(u, 1/u.shape[0])
        wavelength = results['wavelength_peak'] / 2 * 10
        wavelengths.append(wavelength)

        axs[1, ii].imshow(torch.abs(Fshift), origin="lower")

        axs[1, ii].set_title(f"{wavelength}")

    thickness = [float(x) for x in titles]
    

    plt.tight_layout()

    plt.figure()

    plt.scatter(thickness, wavelengths)
    plt.show()
