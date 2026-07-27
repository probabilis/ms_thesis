import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import torch
import pandas as pd

from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from utils.env_utils import plotting_style, PATHS


def read_csv(FOLDER_PATH, FILE_NUMBER, OUTPUT_PATH = None, PLOT = False, deltaN = 180, return_gaussian_filter = False):

    """
    Read function for raw saved MCD data (*.csv format -> already converted from *.tif to *.csv by preprocessing.py script)
    
    uses: 
    FOLDER_PATH : [Path object]
    FILENUMBER : [str]
    """

    FILE_PATH = FOLDER_PATH / f"mcd_slice_{FILE_NUMBER}.csv"
    print(f"Reading file path: {FILE_PATH}")

    img_raw = pd.read_csv(FILE_PATH, header=None).values.astype(np.float32)
    
    with open(FOLDER_PATH / "metadata.csv", "r") as f:
        metadata = json.load(f)

    if int(FILE_NUMBER) >= 1:
        exp_image_width = metadata["FOV"][int(FILE_NUMBER)-1]
    else:
        raise ValueError("FileNumber can not be smaller than 1.")
    
    print("Exp. image width [mum]:", exp_image_width)

    # 2) crop ROI
    N = img_raw.shape[0]
    print("Raw size: ", N)
    #print("Initial image size:", N)
    roi = img_raw[deltaN:N-deltaN, deltaN:N-deltaN]
    roi_size = roi.shape[0]
    #print("Reduce image size:", roi_size)
    reduced_exp_image_width = exp_image_width * roi_size/N


    image_widths = [exp_image_width, reduced_exp_image_width, reduced_exp_image_width]

    # 3) standardization 
    #m, s = np.median(roi), np.median(np.abs(roi - np.median(roi))) + 1e-6
    m, s = np.mean(roi), np.std(roi)
    img_standardize = (roi - m) / s


    img_gaussian_filter = gaussian_filter(img_standardize, sigma = 20)

    img_plot = [img_raw, img_standardize, img_gaussian_filter]
    img_titles = [f"Raw", "Standardized $|\\sigma| = 1$", "Standardized and Gaussian filter"]

    if PLOT:
        plotting_style()

        fig, axs = plt.subplots(3, 2, figsize = (14,10) )

        for ii, img in enumerate(img_plot):

            half_width = image_widths[ii] / 2

            # image data plot
            im = axs[ii, 0].imshow(img, cmap='gray',origin="lower", extent=(-half_width, half_width,-half_width,half_width) )
            axs[ii, 0].set_xlabel("$\\mu\\mathrm{m}$")
            axs[ii, 0].set_ylabel("$\\mu\\mathrm{m}$")
            
            # histogram data plot
            axs[ii, 1].hist(img.ravel(), bins=64, density = True, alpha=0.6, color='b')
            
            xmin, xmax = axs[ii, 1].get_xlim()
            ymin, ymax = axs[ii, 1].get_ylim()
            if ii == 0:
                axs[ii, 1].set_xticks(np.round(np.linspace(xmin, xmax, 4), 2))

            if ii == 1:
                x = np.linspace(xmin, xmax, 100)
                (mu, std) = norm.fit(img.ravel())

                p = norm.pdf(x, mu, std)
                ymin, ymax = p.min(), p.max() 
                axs[ii, 1].plot(x, p, 'k', linewidth=2, label = "$\\mathcal{N}(\\mu, \\sigma)$")
                axs[ii, 1].vlines(mu, ymin, ymax, color = "cornflowerblue", linewidth = 3, linestyle = "--", label = f"$\\mu = {abs(mu):.1f}$")
                axs[ii, 1].vlines(mu + std, ymin, ymax/2, color = "salmon", linewidth = 3, linestyle = "--", label = f"$\\sigma = {std:.1f}$")
                axs[ii, 1].vlines(mu - std, ymin, ymax/2, color = "salmon", linewidth = 3, linestyle = "--")
                axs[ii, 1].legend(loc = "lower right")
                axs[ii, 1].set_xlim(mu-4*std, mu+4*std)
                axs[ii, 1].set_xticks(np.round(np.linspace(mu-4*std, mu+4*std, 4), 2))
    
            axs[ii, 1].set_title(img_titles[ii])
            axs[ii, 1].grid(color = "gray")

            axs[ii, 1].set_yticks(np.round(np.linspace(ymin, ymax, 4), 2))

            plt.colorbar(im, label = r"$A_{MCD}$", ax=axs[ii, 0], location = "right")


        axs[0,0].set_title(f"Raw MCD image \n ${N}\\times{N} \\mathrm{{ px}}$")
        axs[1,0].set_title(f"Cropped MCD image with \n ${roi_size}\\times{roi_size} \\mathrm{{ px}}$")
        axs[2,0].set_title(f"Cropped MCD image \n and Gaussian filter")

        for kk in range(0,3):
            axs[kk, 1].set_xlabel("$u^{(ij)}$")
            axs[kk, 1].set_ylabel("$p[u^{(ij)}]$")

        fig.tight_layout()
        if OUTPUT_PATH is None:
            fig.savefig(FILE_PATH.with_suffix(".png"), dpi = 300)
        else:
            fig.savefig(OUTPUT_PATH, dpi = 300)
        plt.show()
    

    if return_gaussian_filter:
        return torch.from_numpy(img_gaussian_filter.astype(np.float32))
    else:
        return torch.from_numpy(img_standardize.astype(np.float32))


if __name__ == "__main__":
    FOLDER_PATH = PATHS.BASE_EXPDATA / "data_01" / "csv"
    for file_nr in range(1, 8):
        read_csv(FOLDER_PATH, f"00{file_nr}", PLOT = True)

