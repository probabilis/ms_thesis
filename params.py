
from dataclasses import dataclass, replace, asdict
from typing import Optional
import os
from env_utils import term_size, bcolors



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
class SimulationParameters:
    dt: float
    max_it_fixpoint: int
    max_it: int
    tol: float
    stop_limit: float
    c0 : float # integral constant


# ----------------------------------------- #


def get_DataParameters(dp : Optional[DataParameters]):
    print_data_class(dp)
    gridsize = dp.gridsize
    N = dp.N
    th = dp.th
    epsilon = dp.epsilon
    gamma = dp.gamma

    return gridsize, N, th, epsilon, gamma

def get_SimulationParamters(sp : Optional[SimulationParameters]):
    print_data_class(sp)
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

# Simulation params
sim_params = SimulationParameters(
dt = 1/10,
max_it_fixpoint = 40,
max_it = 50_000,
tol = 1e-6,
stop_limit = 1e-8,
c0 = 9/32
)

# Sinus initial config data params
sin_data_params = replace(labyrinth_data_params, gamma = 1/50)


# ----------------------------------------- #

