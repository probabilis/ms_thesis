import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict, replace

from utils.pattern_formation import initialize_u0_random

from params.opt_params import labyrinth_data_params, get_DataParameters, get_SimulationParamters, sim_config
from params.opt_params import pgd_sim_params as ngd_sim_params
from params.lipschitz import evaluate_lipschitz_constant

from utils.env_utils import PATHS, print_bars, plotting_style, log_data

from optimization.gd_nesterov import gradient_descent_nesterov


"""
functional with new dir layout
"""


if __name__ == "__main__":

    plotting_style()

    FOLDER_PATH = PATHS.PATH_PARAMS_STUDY

    LIVE_PLOT = False
    DATA_LOG = False


    labyrinth_data_params = replace(labyrinth_data_params, N = 100)

    gridsize, N, th, epsilon, gamma = get_DataParameters(labyrinth_data_params)

    ngd_sim_params = replace(ngd_sim_params, num_iters = 20_000)
    
    print_bars()
    print(labyrinth_data_params)
    print(ngd_sim_params)
    print(sim_config)
    print_bars()

    N_est = 1

    #gamma_ls = np.linspace(1, 0, 10)
    gamma_ls = np.array([1/500, 1/800, 1/1000, 1/1500, 1/2000, 1/3000, 1/4000, 1/5000, 1/8000, 1/12000])
    energies_ls = []

    u0 = initialize_u0_random(N, REAL = True)

    for ii in range(N_est):
        for gamma in gamma_ls:
            print_bars()
            print("Gamma: ", gamma)

            eta = evaluate_lipschitz_constant(gamma, epsilon, N, gridsize)
            print("eta", eta)

            labyrinth_data_params = replace(labyrinth_data_params, gamma = gamma)
            ngd_sim_params = replace(ngd_sim_params, tau = eta)
            
            u, energies = gradient_descent_nesterov(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, **asdict(labyrinth_data_params),**asdict(ngd_sim_params), **asdict(sim_config))
            
            energies_ls.append( energies[-1])

    
    def algebraic_scaling(gamma):
        """
        Theoretical scaling of the energy over gamma
        -> power law with gamma^(1/2) as Condette described in his thesis
        """
        return gamma**(1/2)


    plt.figure(figsize = (8,6))

    print(gamma_ls)
    print(energies_ls)

    plt.scatter(gamma_ls, energies_ls, label = "simul.")
    plt.plot(gamma_ls, algebraic_scaling(gamma_ls), linestyle = "--", label = "theor.")

    plt.xlabel("$\\gamma$")
    plt.ylabel("$E_\\mathrm{opt}$")
    
    #plt.yscale("log")
    
    #plt.ylim(np.min(energies_ls)-1e-1,np.max(energies_ls)+1e+1)

    plt.grid(color = "gray")
    plt.legend(loc = "lower right")
    plt.savefig(FOLDER_PATH / "asymptotic_energy.png", dpi = 300)
    plt.show()