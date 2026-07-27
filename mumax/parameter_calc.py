import math

MU0 = 4.0 * math.pi * 1e-7

C = 7.64 * 10**(-15)
DELTA_C = 0.12 * 10**(-16)


B =  3.0 * 10**(-8)
DELTA_B = 0.2 * 10**(-9)


C = C + DELTA_C
B = B + DELTA_B


def As(Keff):
    return (C/2)**2 * Keff

def Ms(Keff):
    return math.sqrt(4 * math.pi * MU0 * math.sqrt( (C/2)**2 * Keff**2 ) / B )

def delta_As(Keff):
    return C/2 * Keff * DELTA_C

def delta_Ms(Keff):
    return 1/2 * Ms(Keff) * 1/B * DELTA_B + Ms(Keff) * 1/C * DELTA_C 


Keff_values = [
    20e3,
    40e3,
    80e3,
]

As_values = []
Ms_values = []

delta_As_values = []
delta_Ms_values = []

for keff in Keff_values:
    As_values.append(As(keff))
    Ms_values.append(Ms(keff))

    delta_As_values.append(delta_As(keff))
    delta_Ms_values.append(delta_Ms(keff)) 

print("As:" , As_values)   
print("dAs: ", delta_As_values)

print("Ms: ", Ms_values)
print("dMs: ", delta_Ms_values)