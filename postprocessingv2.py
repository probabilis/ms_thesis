import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib.pyplot as plt

def extract_domain_wall(u, eps=0.02, smooth_sigma=1.0):
    """
    Extract domain wall as a binary mask from scalar field u.
    
    Parameters
    ----------
    u : 2D ndarray
        Scalar field (experimental or simulated)
    eps : float
        Threshold around zero defining the wall
    smooth_sigma : float
        Gaussian smoothing before wall extraction
    
    Returns
    -------
    wall : 2D bool ndarray
        Domain wall mask
    """
    u_smooth = gaussian_filter(u, smooth_sigma)
    wall = np.abs(u_smooth) < eps

    plt.imshow(wall, cmap = "gray")
    plt.show()
    return wall


def symmetric_wall_distance(wall_exp, wall_sim):
    """
    Compute symmetric mean distance between two wall masks.
    
    Distances are in pixels.
    """
    # Distance to nearest wall pixel
    dist_to_exp = distance_transform_edt(~wall_exp)
    dist_to_sim = distance_transform_edt(~wall_sim)

    # Distances from wall pixels
    d_exp_to_sim = dist_to_sim[wall_exp]
    d_sim_to_exp = dist_to_exp[wall_sim]

    all_distances = np.concatenate([d_exp_to_sim, d_sim_to_exp])

    return {
        "mean_distance_px": np.mean(all_distances),
        "median_distance_px": np.median(all_distances),
        "p95_distance_px": np.percentile(all_distances, 95),
        "max_distance_px": np.max(all_distances),
    }


# -----------------------------
# Example usage
# -----------------------------

# u_exp and u_sim are your raw 2D arrays
# u_exp = ...
# u_sim = ...




from env_utils import read_sim_dat_from_csv, PATHS
import json
from dataclasses import replace
from read import read_csv
from params import get_DataParameters, labyrinth_data_params

if __name__ == "__main__":
    dataset = "data_00"
    recording = "003"
    ENERGY_DIFF_STOP_TOL = "0.01"

    read_types = ["raw", "standardize", "shift", "clipped"]

    INPUT_PATH = PATHS.BASE_EXPDATA

    # ---------------------------------------------------------------

    for read_type in read_types:
    
        OUTPUT_PATH = PATHS.BASE_EXPDATA / dataset / "opt" / recording / ENERGY_DIFF_STOP_TOL / read_type
        
        with open(INPUT_PATH / f"{dataset}" / "params_file.json", "r") as _file:
            params_file = json.load(_file)

        gamma_ls = params_file[recording]
        _lambda_ls = [0.06, 0.1, 0.2]

        print("Gamma's: ", gamma_ls)
        print("Lambda's: ", _lambda_ls)

        num_iters = 5000
        N_exp = 664
        _lambda = 0.06

        labyrinth_data_params = replace(labyrinth_data_params, N = N_exp)
        labyrinth_data_params = replace(labyrinth_data_params, gamma = 0.01)

        gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)

        u_exp = read_csv(INPUT_PATH / f"{dataset}/csv/mcd_slice_{recording}.csv", "raw")

        df_energies, u_sim = read_sim_dat_from_csv(OUTPUT_PATH, N, num_iters, gamma, epsilon, _lambda)



        wall_exp = extract_domain_wall(u_exp, eps=0.02)
        wall_sim = extract_domain_wall(u_sim, eps=0.02)

        metrics = symmetric_wall_distance(wall_exp, wall_sim)

        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")

        print("-------------------")