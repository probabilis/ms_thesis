import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import pandas as pd
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure
from sklearn.mixture import GaussianMixture
from skimage.morphology import disk
from typing import Literal

from env_utils import plotting_style



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


def read_csv(_FILE_PATH, method : Literal["raw","standardize","clipped","gmm"], PLOT = False, CLEAN_DOMAINS = False):

    img = pd.read_csv(_FILE_PATH, header=None).values.astype(np.float32)
    img[np.isnan(img)] = 0

    # 1) remove background
    hp = img - ndi.gaussian_filter(img, sigma=200)

    # 2) crop ROI (your deltaN)
    deltaN = 180
    N = hp.shape[0]
    roi = hp[deltaN:N-deltaN, deltaN:N-deltaN]

    # 3) standardization 
    m, s = np.median(roi), np.median(np.abs(roi - np.median(roi))) + 1e-6
    z = (roi - m) / s
    
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

    z_clipped = np.where(roi > 0, 1, -1)

    if PLOT:
        plotting_style()
        fig, axs = plt.subplots(4,2, figsize = (12,12))

        axs[0,0].imshow(img, cmap='gray')
        axs[0,1].set_title("MCD image")
        axs[0,1].hist(img.ravel(), bins=256, density = True, color ='gray')

        axs[1,0].imshow(z, cmap='gray')
        axs[1,1].set_title("Cropped + Standardized + Gaussian Filter (GF)")
        axs[1,1].hist(z.ravel(), bins=256, density = True, color='gray')

        axs[2,1].set_title("Cropped + GF + Clipped to $\\pm 1$")
        axs[2,0].imshow(z_clipped, cmap='gray')
        axs[2,1].hist(z_clipped.ravel(), bins=256, density = True, color='gray')
        axs[2,1].set_xlim(-1.1, +1.1)

        axs[3,0].imshow(y, cmap="gray")
        axs[3,1].set_title("Cropped + GF + GMM posterior → $[-1,1]$")
        axs[3,1].hist(y.ravel(), bins=256, density = True, color='gray')
        axs[3,1].axvline(filters.threshold_otsu(y), ls='--') 
        axs[3,1].set_xlim(-1.1, +1.1)

        for ii in range(4):
            axs[ii, 1].grid(color = "gray")
            axs[ii, 0].axes.get_xaxis().set_ticks([])
            axs[ii, 0].axes.get_yaxis().set_ticks([])

        fig.tight_layout()
        fig.savefig(_FILE_PATH.with_suffix(".png"), dpi = 300)
        plt.show()

    if method == "raw":
        return torch.from_numpy(roi.astype(np.float32))

    if method == "standardize":
        return torch.from_numpy(z.astype(np.float32))

    if method == "clipped":
        return torch.from_numpy(z_clipped.astype(np.float32))

    if method == "gmm":
        return torch.from_numpy(y.astype(np.float32))



if __name__ == "__main__":
    read_csv(Path("data/expdata/data_00/csv/mcd_slice_001.csv"), "standardize", PLOT = True)

