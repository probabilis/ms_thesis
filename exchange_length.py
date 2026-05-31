import math
import numpy as np
"""
Ta/CoFeB/MgO sample

data_00
recording 4
"""

gamma_exp = 0.0002 # best optimization
eps = 0.01
N = 664
N_total = 1024

exp_image_width = 37.24 * 1e-6
reduced_exp_image_width = exp_image_width * N/N_total

def calculate(gamma_exp, eps, N, N_total, exp_image_width):
    #print("EL: ", math.sqrt(eps * gamma_exp * reduced_exp_image_width / 2))
    
    print("Reduced Image Width [mum]: ", reduced_exp_image_width * 1e6)
    print(f"Reduced Pixel Width [mum]: {reduced_exp_image_width * 1e6 / N}, [nm]: {reduced_exp_image_width * 1e9 / N}")
    exchange_length_norm = math.sqrt(eps * gamma_exp/2 )
    print("Exchange length normalized [1]: ", exchange_length_norm)


    exchange_length = exchange_length_norm * reduced_exp_image_width
    print("Exchange length in [m]: ", exchange_length)
    print("Exchange length in [nm]: ", exchange_length * 1e9)
    print("-----------------------------------------")

    quality_factor = gamma_exp / (2 * exchange_length)
    print("Quality Factor Q: ", quality_factor**2)
    print(1/664)

    return exchange_length, quality_factor


if __name__ == "__main__":

    d_ls = []

    for gamma in [0.008, 0.004, 0.002, 0.0004]:
        d, Q = calculate(gamma, eps, N, N_total, exp_image_width)    
        d_ls.append(d)

    print(np.mean(d_ls))