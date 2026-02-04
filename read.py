import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import norm
from skimage import exposure, filters, morphology, measure
from sklearn.mixture import GaussianMixture
from skimage.morphology import disk
from typing import Literal

from env_utils import plotting_style

def standardize_shift(x):

    x = (x - x.mean()) / x.std()          # standardized image
    median = np.median(x)
    x_new = np.empty_like(x)
    x_new[x <= median] = x[x <= median] - 1
    x_new[x >  median] = x[x >  median] + 1
    return x_new

def clean_domains(y, min_obj=100, min_hole=100, se_radius=2):
    """
    y: float array in [-1,1] from GMM step
    min_obj: min area to keep (pixels)
    min_hole: min hole area to fill (pixels)
    se_radius: structuring element radius
    """
    # optional light despeckle before threshold
    y_s = ndi.median_filter(y, size=3)

    # binary masks
    bright = y_s > 0
    dark   = ~bright

    # remove tiny islands and fill tiny holes on BOTH classes
    bright = morphology.remove_small_objects(bright, min_size=min_obj)
    bright = morphology.remove_small_holes(bright, area_threshold=min_hole)

    dark = morphology.remove_small_objects(dark, min_size=min_obj)
    dark = morphology.remove_small_holes(dark, area_threshold=min_hole)

    # mild opening/closing to smooth edges
    se = disk(se_radius)
    bright = morphology.opening(bright, se)
    bright = morphology.closing(bright, se)

    dark = morphology.opening(dark, se)
    dark = morphology.closing(dark, se)

    # resolve overlaps/gaps by majority vote
    undecided = ~(bright ^ dark)
    if np.any(undecided):
        # assign undecided to nearest large region label
        lbl_b = measure.label(bright)
        lbl_d = measure.label(dark)
        dist_b = ndi.distance_transform_edt(~bright)
        dist_d = ndi.distance_transform_edt(~dark)
        assign_bright = dist_b < dist_d
        bright[undecided] = assign_bright[undecided]
        dark[undecided] = ~bright[undecided]

    # map back to [-1,1]
    y_clean = np.where(bright, 1.0, -1.0).astype(np.float32)
    return y_clean

def read_gmm(CLEAN_DOMAINS = False):

    # 4) fit 2-component GMM on intensities
    x = z.reshape(-1,1)
    gm = GaussianMixture(n_components=2, covariance_type="full", n_init=5, random_state=0)
    gm.fit(x)

    # ensure component_1 is the BRIGHT class
    means = gm.means_.ravel()
    bright_idx = np.argmax(means)
    post = gm.predict_proba(x)[:, bright_idx].reshape(z.shape)   # P(bright | x)

    # 5) sharpen posteriors and map to [-1, +1]
    # alpha>1 pushes values to 0/1. Try 2–6.
    alpha = 4.0
    p = post**alpha / (post**alpha + (1.0-post)**alpha)
    y = 2.0*p - 1.0

    if CLEAN_DOMAINS:
        # 6) diagnostics
        y = clean_domains(y)


    fig, axs = plt.figure()

    axs.imshow(y, cmap="gray",origin="lower", extent=(0,1,0,1))
    axs[4,1].set_title("Cropped + GF + GMM posterior → $[-1,1]$")
    axs[4,1].hist(y.ravel(), bins=256, density = True, color='gray')
    axs[4,1].axvline(filters.threshold_otsu(y), ls='--') 
    #axs[4,1].set_xlim(-5, +5)

    return y

def read_csv_old(_FILE_PATH, method : Literal["raw","standardize","shift", "clipped"], PLOT = False):

    """
    Plotting 4 different configurations of the experimental magnetic structure.

    raw ... cropped image
    standardize ... standardized image
    clipped ... clipped image
    shift ... shifted image
    """

    img = pd.read_csv(_FILE_PATH, header=None).values.astype(np.float32)
    img[np.isnan(img)] = 0

    # 2) crop ROI (your deltaN)
    deltaN = 180
    N = img.shape[0]
    print("Initial image size:", N)
    roi = img[deltaN:N-deltaN, deltaN:N-deltaN]
    print("Reduce image size:", roi.shape[0])
    roi -= ndi.gaussian_filter(roi, sigma=200)

    # 3) standardization 
    m, s = np.median(roi), np.median(np.abs(roi - np.median(roi))) + 1e-6
    img_standardize = (roi - m) / s

    img_clipped = np.where(roi > 0, 1, -1)


    
    img_standardize_shift = standardize_shift(roi)

    img_plot = [roi, img_standardize, img_standardize_shift, img_clipped]
    img_titles = ["Raw MCD image (cropped to square)", "Standardized + Gaussian Filter (GF)", "Standardized + GF + Shifted", "GF + Clipped to $\\pm 1$"]

    if PLOT:
        plotting_style()

        fig, axs = plt.subplots(len(img_plot), 2, figsize = (8,8))

        for ii, img in enumerate(img_plot):
            axs[ii, 0].imshow(img, cmap='gray',origin="lower", extent=(0,1,0,1))
            axs[ii, 1].hist(img.ravel(), bins=256, density = True, color='gray')
            axs[ii, 1].set_title(img_titles[ii])

            axs[ii, 1].grid(color = "gray")
            axs[ii, 0].axes.get_xaxis().set_ticks([])
            axs[ii, 0].axes.get_yaxis().set_ticks([])

            if ii == 3:
                axs[ii, 1].set_xlim(-1.2, +1.2)

        fig.tight_layout()
        #fig.savefig(_FILE_PATH.with_suffix(".png"), dpi = 300)
        plt.show()

    if method == "raw":
        return torch.from_numpy(roi.astype(np.float32))

    if method == "standardize":
        return torch.from_numpy(img_standardize.astype(np.float32))

    if method == "clipped":
        return torch.from_numpy(img_clipped.astype(np.float32))

    if method == "shift":
        return torch.from_numpy(img_standardize_shift.astype(np.float32))


# above functions can be deleted
# ----------------------------


def read_csv(FILE_PATH, PLOT = False, deltaN = 180):

    """
    Read function for raw saved MCD data (*.csv format -> already converted from *.tif to *.csv by preprocessing.py script)
    """

    img = pd.read_csv(FILE_PATH, header=None).values.astype(np.float32)
    img[np.isnan(img)] = 0

    # 2) crop ROI
    N = img.shape[0]
    print("Initial image size:", N)
    roi = img[deltaN:N-deltaN, deltaN:N-deltaN]
    roi_size = roi.shape[0]
    print("Reduce image size:", roi_size)

    # 3) standardization 
    #m, s = np.median(roi), np.median(np.abs(roi - np.median(roi))) + 1e-6
    m, s = np.mean(roi), np.std(roi)
    img_standardize = (roi - m) / s

    img_plot = [roi, img_standardize]
    img_titles = [f"Raw MCD image / $ROI$ = {roi_size}$\\times${roi_size}", "Standardized $|\\sigma| = 1$"]

    if PLOT:
        plotting_style()

        fig, axs = plt.subplots(2, 2, figsize = (8,8))

        for ii, img in enumerate(img_plot):
            axs[ii, 0].imshow(img, cmap='gray',origin="lower", extent=(0,1,0,1))
            axs[ii, 0].axes.get_xaxis().set_ticks([])
            axs[ii, 0].axes.get_yaxis().set_ticks([])


            
            axs[ii, 1].hist(img.ravel(), bins=32, density = True,alpha=0.6, color='b')
            
            if ii == 1:
                xmin, xmax = axs[ii, 1].get_xlim()
                ymin, ymax = axs[ii, 1].get_ylim()
                x = np.linspace(xmin, xmax, 100)
                (mu, std) = norm.fit(img.ravel())

                p = norm.pdf(x, mu, std)
                axs[ii, 1].plot(x, p, 'k', linewidth=2)
                axs[ii, 1].vlines(mu, ymin, ymax, color = "cornflowerblue", linewidth = 3, linestyle = "--", label = f"$\\mu = ${mu:.3f}")
                axs[ii, 1].vlines(mu + std, ymin, ymax/2, color = "salmon", linewidth = 3, linestyle = "--", label = f"$\\sigma = ${std:.3f}")
                axs[ii, 1].vlines(mu - std, ymin, ymax/2, color = "salmon", linewidth = 3, linestyle = "--")
                axs[ii, 1].legend(loc = "lower right")

            axs[ii, 1].set_title(img_titles[ii])
            axs[ii, 1].grid(color = "gray")
            ymin, ymax = axs[ii, 1].get_ylim()
            xmin, xmax = axs[ii, 1].get_xlim()
            axs[ii, 1].set_xticks(np.round(np.linspace(xmin, xmax, 4), 2))
            axs[ii, 1].set_yticks(np.round(np.linspace(ymin, ymax, 4), 2))

        fig.tight_layout()
        #fig.savefig(_FILE_PATH.with_suffix(".png"), dpi = 300)
        plt.show()
        
    return torch.from_numpy(img_standardize.astype(np.float32))


if __name__ == "__main__":
    #read_csv_old(Path("data/expdata/data_00/csv/mcd_slice_004.csv"), "standardize", PLOT = True)
    read_csv(Path("data/expdata/data_00/csv/mcd_slice_004.csv"), PLOT = True)

