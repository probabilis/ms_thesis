import torch
import matplotlib.pyplot as plt

from utils.pattern_formation import initialize_u0_random
from utils.env_utils import PATHS, plotting_style, print_bars

from optimization.crank_nicolson import adapted_crank_nicolson
from optimization.gradient_descent import gradient_descent
from optimization.gd_proximal import gradient_descent_proximal
from optimization.gd_nesterov import gradient_descent_nesterov



def convergence_comparison(WITH_CN_AS_REFERENCE = False):
    """
    Convergence comparison of implemented algorithms 
    WITH_CN_AS_REFERENCE : boolean ... Crank Nicolson scheme from Condette can also be shown, used as reference
    """

    STOP_BY_TOL = True # False : which means full iterations are done, independent of energy convergence value
    
    if WITH_CN_AS_REFERENCE:
        # multiple CN instances
        energies_cn = []
        
        for max_it_fixpoint in max_it_fixpoint_ls:
            for dt in dt_ls:
                print(f"CN constellation: max_it_fixpoint = {max_it_fixpoint} | dt = {dt}")
                max_it = num_iters
                _, energies = adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0, STOP_BY_TOL)
                energies_cn.append(energies)
                print_bars()

    _, energies_gd = gradient_descent(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, c0, alpha, num_iters, LAPLACE_SPECTRAL=False, STOP_BY_TOL = STOP_BY_TOL)
    print_bars()
    _, energies_prox = gradient_descent_proximal(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, STOP_BY_TOL)
    print_bars()
    _, energies_nest = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, gamma, epsilon, tau, c0, num_iters, prox_newton_iters, tol_newton, LAPLACE_SPECTRAL=False, STOP_BY_TOL = STOP_BY_TOL)
    print_bars()

    # ---------------------------------------------------------------

    fig, ax = plt.subplots(1,1, figsize = (8,6))

    colors_cn = ['cornflowerblue', 'royalblue','blue', 'black']
    colors_gd = ['lightcoral', 'mediumseagreen', 'red']

    #ax.vlines(len(energies_gd)-1, y_min, y_max, linestyle = "--", linewidth = 3, color =  colors_gd[0] )
    if WITH_CN_AS_REFERENCE:
        ii = 0
        for max_it_fixpoint in max_it_fixpoint_ls:
                for dt in dt_ls:
                    ax.plot(torch.arange(1, len(energies_cn[ii])+1, 1), energies_cn[ii], label= fr"Crank Nicolson $N_{{fixpoint}}$ = {max_it_fixpoint} $|$ $dt$ = {dt}", linewidth = 3, color = colors_cn[ii])
                    ii += 1 

    ax.plot(torch.arange(1, len(energies_gd)+1, 1), energies_gd, label="Gradient Descent", linewidth = 3, color =  "blue")
    ax.plot(torch.arange(1, len(energies_prox)+1, 1), energies_prox, label="Proximal Gradient Descent", linewidth = 3, linestyle = ":", color = "black")
    ax.plot(torch.arange(1, len(energies_nest)+1, 1), energies_nest, label="Nesterov Proximal GD", linewidth = 3, color = colors_gd[2])

    ax.set_xlabel("iterations $n$")
    ax.set_ylabel("energy value $E(u_n(x,y)$")
    #ax.set_title("Energy convergence comparison")

    #ax.set_yscale("log")
    ax.set_xscale("log")

    ax.legend(loc = "upper right")
    ax.grid(True)
    fig.tight_layout()
    base_file_name = f"energy_convergence_comparison_N={N}_nmax={num_iters}_gamma={gamma}_eps={epsilon}"
    if WITH_CN_AS_REFERENCE:
        file_name = base_file_name + "with_cn" + ".png"
    else:
        file_name = base_file_name + ".png"
    
    plt.savefig(FOLDER_PATH / file_name, dpi = 300)
    plt.show()




if __name__ == "__main__":

    # ---------------------------------------------------------------

    plotting_style()
    FOLDER_PATH = PATHS.PATH_COMPARISON
    LIVE_PLOT = False
    DATA_LOG = False

    # ---------------------------------------------------------------

    gridsize = 1.0
    N = 64
    epsilon = 1/100
    gamma = 1/2000

    c0 = 9/32
    th = 1.0
    from params.lipschitz import evaluate_lipschitz_constant
    alpha = tau = 0.1 #evaluate_lipschitz_constant(gamma, epsilon, N, gridsize=1.0, th=1.0)       # step size for plain GD &  # step size for prox-based methods 
            
    ENERGY_STOP_TOL = 1e-8  # energy stopping tolerance if STOP_BY_TOL is true for methods 
    stop_limit = ENERGY_STOP_TOL
    tol = 1e-4

    prox_newton_iters = 10
    tol_newton = 1e-6

    max_it_fixpoint_ls = [20]
    dt_ls = [0.1, 0.5, 1.0]

    num_iters = 100_000  # number of iterations for all methods

    # ---------------------------------------------------------------
    
    u0 = initialize_u0_random(N)
    convergence_comparison()