import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from scipy.stats import norm

from utils.env_utils import plotting_style, PATHS


def read_csv(FILE_PATH, OUTPUT_PATH = None, PLOT = False, deltaN = 180):

    """
    Read function for raw saved MCD data (*.csv format -> already converted from *.tif to *.csv by preprocessing.py script)
    """

    img = pd.read_csv(FILE_PATH, header=None).values.astype(np.float32)
    img[np.isnan(img)] = 0

    # 2) crop ROI
    N = img.shape[0]
    #print("Initial image size:", N)
    roi = img[deltaN:N-deltaN, deltaN:N-deltaN]
    roi_size = roi.shape[0]
    #print("Reduce image size:", roi_size)

    # 3) standardization 
    #m, s = np.median(roi), np.median(np.abs(roi - np.median(roi))) + 1e-6
    m, s = np.mean(roi), np.std(roi)
    img_standardize = (roi - m) / s

    img_plot = [roi, img_standardize]
    img_titles = [f"$ROI = {roi_size}\\times{roi_size}$", "Standardized $|\\sigma| = 1$"]

    if PLOT:
        plotting_style()

        fig, axs = plt.subplots(2, 2, figsize = (10,8))

        for ii, img in enumerate(img_plot):

            # image data plot
            axs[ii, 0].imshow(img, cmap='gray',origin="lower", extent=(0,1,0,1))
            axs[ii, 0].axes.get_xaxis().set_ticks([])
            axs[ii, 0].axes.get_yaxis().set_ticks([])
            
            # histogram data plot
            axs[ii, 1].hist(img.ravel(), bins=32, density = True, alpha=0.6, color='b')
            
            ymin, ymax = axs[ii, 1].get_ylim()
            xmin, xmax = axs[ii, 1].get_xlim()

            if ii == 1:
                x = np.linspace(xmin, xmax, 100)
                (mu, std) = norm.fit(img.ravel())

                p = norm.pdf(x, mu, std)
                axs[ii, 1].plot(x, p, 'k', linewidth=2)
                axs[ii, 1].vlines(mu, ymin, ymax, color = "cornflowerblue", linewidth = 3, linestyle = "--", label = f"mean $\\mu = {abs(mu):.1f}$")
                axs[ii, 1].vlines(mu + std, ymin, ymax/2, color = "salmon", linewidth = 3, linestyle = "--", label = f"std $\\sigma = {std:.1f}$")
                axs[ii, 1].vlines(mu - std, ymin, ymax/2, color = "salmon", linewidth = 3, linestyle = "--")
                axs[ii, 1].legend(loc = "lower right")
                
                axs[ii, 1].set_xlim(mu-4*std, mu+4*std)

    
            axs[ii, 1].set_title(img_titles[ii])
            axs[ii, 1].grid(color = "gray")
            #axs[ii, 1].set_xticks(np.round(np.linspace(xmin, xmax, 4), 2))
            #axs[ii, 1].set_yticks(np.round(np.linspace(ymin, ymax, 4), 2))


        axs[0,0].set_title("Raw MCD image")
        fig.tight_layout()
        if OUTPUT_PATH is None:
            fig.savefig(FILE_PATH.with_suffix(".png"), dpi = 300)
        else:
            fig.savefig(OUTPUT_PATH, dpi = 300)
        plt.show()
        
    return torch.from_numpy(img_standardize.astype(np.float32))


if __name__ == "__main__":
    exp_data_test = PATHS.BASE_EXPDATA / "data_01" / "csv" / "mcd_slice_004.csv"
    print(exp_data_test)
    read_csv(exp_data_test, PLOT = True)

