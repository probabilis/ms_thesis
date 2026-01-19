import math

"""
Ta/CoFeB/MgO sample

data_00
recording 4
"""

gamma_exp = 0.0002 # best optimization
eps = 0.01

exp_image_width = 37.24 * 1e-6
reduced_exp_image_width = exp_image_width * 664/1024

exchange_length_norm = math.sqrt(eps * gamma_exp/4)

exchange_length = exchange_length_norm * reduced_exp_image_width
print("Exchange length in [m]: ", exchange_length)
print("Exchange length in [nm]: ", exchange_length * 1e9)