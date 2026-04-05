import torch
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit



domain_pattern_lengths = [17.07508226, 5.69169425, 5.69169425, 3.41501639, 1.89723127, 1.31346779]



def kaplan_gehring_limit0(L, b, D0):
    return L * math.exp(math.pi * b /2 + 1) * torch.exp(torch.pi * D0 /2 * L )

def kaplan_gehring_limit1(L, b, Ms, a0, J):
    return L * math.exp(math.pi * b /2 + 1) * math.exp(J / 4 * Ms**2 * a0**2 )


def func(t, As, Keff, Ms):
        MU0 = 4.0 * torch.pi * 1e-7
        return 2 * (As/Keff)**0.5 * torch.exp(4 * torch.pi * MU0 * (As * Keff)**0.5 / (Ms**2 * t)  ) 

def funcNP(t, As, Keff, Ms):
    MU0 = 4.0 * np.pi * 1e-7
    return 2 * (As/Keff)**0.5 * np.exp(4 * np.pi * MU0 * (As * Keff)**0.5 / (Ms**2 * t)  ) 


def func_reduced(t, a, b, c):
    return a * np.exp( b / t ) * np.exp(c) * t



As = 12e-12
Keff = 4e4
Ms = 1.2e6

thickness = [1.3, 1.31, 1.32, 1.34, 1.36, 1.4]
thickness = [x*1e-9 for x in thickness]
thickness = np.array(thickness)

#domain_pattern_lengths = [x*1e-6 for x in domain_pattern_lengths]
domain_pattern_lengths = np.array(domain_pattern_lengths)

#popt, pcov = curve_fit(func_reduced, thickness, domain_pattern_lengths)
#print(popt)
#print(pcov)

plt.plot(thickness, domain_pattern_lengths)
#plt.plot(thickness, func_reduced(thickness, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))#
#plt.plot(thickness, func_reduced(thickness, 1e-6,1e-9,1) )
plt.show()

#y = kaplan_gehring_limit0(x, -0.666, 1e2)