
from dataclasses import dataclass, replace, asdict
from typing import Optional, Union
import os
from env_utils import term_size, bcolors
from pattern_formation import define_spaces

# ----------------------------------------- #

def print_data_class(dataclass_instance):
    print('─' * term_size.columns) 
    print(dataclass_instance)
    for name, value in asdict(dataclass_instance).items():
        print(f"{name}: {value}")
    print('─' * term_size.columns)

@dataclass
class DataParameters:
    gridsize: int # gridsize
    N: int # nr. of grid-points
    th : float # sample thickness
    epsilon: float # tuning 1
    gamma: float # tuning 2


@dataclass
class CN_SimulationParameters:
    dt: float
    max_it_fixpoint: int
    max_it: int
    tol: float
    stop_limit: float
    c0 : float # integral constant


@dataclass
class PGD_SimulationParameters:
    tau : float
    num_iters : int
    prox_newton_iters: int
    tol_newton: float
    c0 : float # integral constant

@dataclass
class GD_SimulationParameters:
    num_iters : int
    c0 : float # integral constant

# ----------------------------------------- #


def get_DataParameters(dp : Optional[DataParameters]):
    #print_data_class(dp)
    gridsize = dp.gridsize
    N = dp.N
    th = dp.th
    epsilon = dp.epsilon
    gamma = dp.gamma

    return gridsize, N, th, epsilon, gamma

def get_SimulationParamters(sp : Optional[Union[CN_SimulationParameters,GD_SimulationParameters,PGD_SimulationParameters]]):
    #print_data_class(sp)
    dt = sp.dt
    max_it_fixpoint = sp.max_it_fixpoint
    max_it = sp.max_it
    tol = sp.tol
    stop_limit = sp.stop_limit
    c0 = sp.c0

    return dt, max_it_fixpoint, max_it, tol, stop_limit, c0

# ----------------------------------------- #

# Labyrinth data parameters
labyrinth_data_params = DataParameters(
gridsize = 1,
N = 128,
th = 1.0,
epsilon = 1/20,
gamma = 1/200
)

# Sinus initial config data params
sin_data_params = replace(labyrinth_data_params, gamma = 1/50)

# ----------------------------------------- #
# Simulation params

# Crank Nicolson
cn_sim_params = CN_SimulationParameters(
dt = 1/10,
max_it_fixpoint = 40,
max_it = 50_000,
tol = 1e-6,
stop_limit = 1e-8,
c0 = 9/32
)

# Gradient Descent params
gd_sim_params = GD_SimulationParameters(
num_iters = 200_000,
c0 = 9/32
)

# PGD sim params
pgd_sim_params = PGD_SimulationParameters(
tau = 5e-3,              # proximal gradient step size
num_iters = 100_000,     # total iterations )
prox_newton_iters = 20,  # iterations for prox Newton
tol_newton = 1e-8,       # stop tol inside prox
c0 = 9/32)


# ----------------------------------------- #

