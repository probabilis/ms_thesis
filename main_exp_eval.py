from dataclasses import replace, asdict
import matplotlib.pyplot as plt

from utils.env_utils import PATHS, print_bars, get_args, plotting_style
from exp_data_processing.read import read_csv

from params.opt_params import exp_data_params, get_DataParameters, get_SimulationParamters
from params.opt_params import pgd_sim_params as ngd_sim_params

from utils.pattern_formation import initialize_u0_random

from exp_data_eval.evaluation import gradient_descent_nesterov_evaluation
from exp_data_eval.evaluation_sweep import grid_sweep_over_lambdas_and_gammas


def main_single_paramter_constellation(exp_data_params, ngd_sim_params, dataset, recording):

    plotting_style()
    
    INPUT_PATH = PATHS.BASE_EXPDATA
    OUTPUT_PATH = PATHS.PATH_EVALUATION

    args = get_args()
    LIVE_PLOT = args.live_plot
    DATA_LOG = args.data_log

    # ---------------------------------------------------------------

    INPUT_FILE_PATH = PATHS.BASE_EXPDATA / f"{dataset}/csv/mcd_slice_{recording}.csv"
    print(f"Reading {INPUT_FILE_PATH} as experimental image data.")
    u_exp = read_csv(INPUT_FILE_PATH, PLOT = True)

    if u_exp.shape[0] != u_exp.shape[1]:
        raise ValueError("Experimental data should be quadratic (NxN tensor).")
    N_exp = u_exp.shape[0]

    # ---------------------------------------------------------------

    num_iters = 5000
    ENERGY_STOP_TOL = 1e-12

    exp_data_params = replace(exp_data_params, gamma = 0.008) 
    ngd_sim_params = replace(ngd_sim_params, num_iters = num_iters, tau = 0.001) # smaller tau because of image

    gridsize, N, th, epsilon, gamma = get_DataParameters(exp_data_params)
    u0 = initialize_u0_random(N, REAL=True)

    print_bars()
    print(exp_data_params)
    print(ngd_sim_params)
    print_bars()

    _lambda = 0.1
    print(f"Learning Rate Lambda: {_lambda}")
    print_bars()

    # ---------------------------------------------------------------

    u, history = gradient_descent_nesterov_evaluation(u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH,**asdict(exp_data_params),**asdict(ngd_sim_params), 
                                                      LOSS_TYPE = "MSE", STOP_BY_TOL=True, ENERGY_STOP_TOL=ENERGY_STOP_TOL)

    # ---------------------------------------------------------------

    fig, axs = plt.subplots(2,2 , figsize = (12,8) )

    from utils.env_utils import main_colormap
    
    axs[0, 0].imshow(u_exp.cpu().numpy(), cmap = main_colormap, origin="lower", extent = (0,1,0,1) )
    axs[0, 1].imshow(u.cpu().numpy(), cmap = main_colormap,origin="lower", extent=(0,1,0,1) )    

    axs[0, 0].set_box_aspect(1)
    axs[0, 1].set_box_aspect(1)
    
    #axs[0].set_title(f"$\\gamma = {gamma}, \\lambda = {_lambda}$")

    axs[1, 0].loglog(history["E_total"], label = "$E_{total}$")

    axs[1, 0].loglog(history["E_grad"], label= "$E_{\\nabla}$")
    axs[1, 0].loglog(history["E_dw"], label= "$E_{W}$")
    axs[1, 0].loglog(history["E_fm"], label= "$E_{\\mathcal{F}}$")
    axs[1, 0].loglog(history["E_data"], label="$E_{Data}$")    
    axs[1, 0].grid(color = "gray")
    axs[1, 0].legend()

    material = "Ta/CoFeB/MgO"

    txt = (
        f"Material: {material}\n"
        f"gamma     = {gamma}\n"
        f"epsilon   = {epsilon}\n"
        f"lambda    = {_lambda}\n"
        f"iters     = {len(history['E_total'])-1}\n"
        f"E_final   = {history['E_total'][-1]:.3e}"
    )
    axs[1, 1].axis("off")
    axs[1, 1].text(0.0, 1.0, txt, va="top", ha="left", fontsize=11)

    #axs[1].set_title(f"$\\Delta E < {ENERGY_STOP_TOL}$")
    #axs[1].set_yscale('log')

    fig.tight_layout()
    #plt.savefig(OUTPUT_PATH / f"evaluation_gamma={gamma}_lambda={_lambda:.2f}_num-iters={num_iters}.png", dpi = 300)
    plt.show()


def main_evaluation_sweep(exp_data_params, ngd_sim_params):
    grid_sweep_over_lambdas_and_gammas(exp_data_params, ngd_sim_params, "simulate")


if __name__ == "__main__":

    #main_single_paramter_constellation(exp_data_params, ngd_sim_params, dataset = "data_01", recording = "001")
    main_evaluation_sweep(exp_data_params, ngd_sim_params)
