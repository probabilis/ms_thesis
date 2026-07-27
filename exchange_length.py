import math
import numpy as np
import matplotlib.pyplot as plt


"""
Ta/CoFeB/MgO sample

data_01
recording 4
"""

gamma_exp = 0.0002 # best optimization
eps = 0.01


N = 664
N_total = 1024

exp_image_width = 37.24 * 1e-6
reduced_exp_image_width = exp_image_width * N/N_total

print("Reduced Image Width [mum]: ", reduced_exp_image_width * 1e6)
print(f"Reduced Pixel Width [mum]: {reduced_exp_image_width * 1e6 / N}, [nm]: {reduced_exp_image_width * 1e9 / N} \n")


def calculate(gamma_exp, gamma_unc, eps):
    #print("EL: ", math.sqrt(eps * gamma_exp * reduced_exp_image_width / 2))    

    exchange_length_norm = math.sqrt(eps * gamma_exp/4 )
    print("Exchange length normalized [1]: ", exchange_length_norm)

    exchange_length = exchange_length_norm * reduced_exp_image_width
    #print("Exchange length in [m]: ", exchange_length)
    print("Exchange length in [nm]: ", exchange_length * 1e9)

    quality_factor = gamma_exp / (2 * exchange_length)
    #print("Quality Factor Q: ", quality_factor**2)

    exchange_length_unc_norm = 1/2 * (gamma_exp)**(-0.5) * math.sqrt(eps / 4) * gamma_unc # größtunsicherheitsmethode
    exchange_length_unc = exchange_length_unc_norm * reduced_exp_image_width
    print("Delta exchange length [nm]:", exchange_length_unc * 1e9)

    print("-----------------------------------------")
    return exchange_length, exchange_length_unc, quality_factor



def theoretical_exchange_length(As, Ms):
    MU0 = 4.0 * math.pi * 1e-7
    return math.sqrt(2 * As / (Ms**2 * MU0) )


if __name__ == "__main__":

    PLOT = True

    d_ls = []
    d_unc_ls = []

    gamma_exp = [0.008, 0.008, 0.002, 0.0004]
    gamma_unc = [0.002, 0.003, 0.001, 0.0002]

    for ii, gamma in enumerate(gamma_exp):
        print("gamma: ", gamma)
        d, d_unc, Q = calculate(gamma, gamma_unc[ii], eps)    
        d_ls.append(d)
        d_unc_ls.append(d_unc)
    print("\n")
    

    #print(f"Mean: {np.mean(d_ls) * 1e9} with Std: {np.std(d_ls) * 1e9}")
    
    As = 44 * 10**(-12)
    
    Ms = (1.2 * 10**6)

    l_ex = theoretical_exchange_length(As, Ms)
    print("l_ex = ", l_ex * 10**9)