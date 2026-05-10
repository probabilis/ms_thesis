from pathlib import Path
from dataclasses import asdict, replace

from utils.env_utils import plotting_style, PATHS, print_bars
from utils.pattern_formation import initialize_u0_random
from params.opt_params import labyrinth_data_params, gd_sim_params, pgd_sim_params, cn_sim_params, get_DataParameters
from params.opt_params import pgd_sim_params as ngd_sim_params

from optimization.gd_autograd import gradient_descent_backtracking
from optimization.gradient_descent import gradient_descent
from optimization.gd_proximal import gradient_descent_proximal
from optimization.gd_nesterov import gradient_descent_nesterov
from optimization.crank_nicolson import adapted_crank_nicolson


def test_autograd(FOLDER_PATH):
    gradient_descent_backtracking(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params), num_iters=500_000, c0 = 9/32)

def test_gradient_descent(FOLDER_PATH):
    gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params), **asdict(gd_sim_params))

def test_gradient_descent_proximal(FOLDER_PATH):
    gradient_descent_proximal(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH,**asdict(labyrinth_data_params),**asdict(pgd_sim_params))

def test_gradient_descent_nesterov(FOLDER_PATH):
    gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), LAPLACE_SPECTRAL=False, ENERGY_STOP_TOL=1e-12)

def test_crank_nicolson(FOLDER_PATH):
    adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params), **asdict(cn_sim_params))





if __name__ == "__main__":

    plotting_style()
    
    LIVE_PLOT = True
    DATA_LOG = True
    #args = get_args()
    #LIVE_PLOT = args.live_plot
    #DATA_LOG = args.data_log

    u0 = initialize_u0_random(labyrinth_data_params.N)
    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)

    print_bars()
    print(labyrinth_data_params)
    #print(gd_sim_params)
    print(cn_sim_params)
    print_bars()

    """
    todo: implement lipschitz constant as parameter return for dataclass
    """
    #test_crank_nicolson(FOLDER_PATH = PATHS.PATH_CN)
    #test_autograd()
    test_gradient_descent(FOLDER_PATH = PATHS.PATH_GD)
    #test_gradient_descent_proximal(FOLDER_PATH = PATHS.PATH_PGD)
    #test_gradient_descent_nesterov(FOLDER_PATH = PATHS.PATH_NEST)

