import torch
import numpy as np
from env_utils import tensor_type


# -----------------------------
# torch related
dtype_real = torch.float64
dtype_complex = torch.complex128
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------

def fourier_multiplier(A):
    sig = torch.zeros_like(A, dtype=dtype_real)
    small = torch.abs(A) < 1e-8
    large = ~small
    sig[small] = 1 - np.pi * torch.abs(A[small])
    sig[large] = (1 - torch.exp(-2 * np.pi * torch.abs(A[large]))) / (2 * np.pi * torch.abs(A[large]))
    return sig


def double_well_potential(u, c0):
    u2 = 1 - torch.abs(u) ** 2
    return c0 * (u2 ** 2)



def energy_value(gamma, epsilon, N, u, th, modk, modk2, c0):
    W = double_well_potential(u, c0)
    ftu = torch.fft.fft2(u) / (N ** 2)

    S = fourier_multiplier(th * modk)
    print('max sigma:', torch.max(S).item())
    print('min sigma:', torch.min(S).item())
    print('mean sigma:', torch.mean(S).item())

    energy = (gamma / epsilon) * torch.sum(W).real / (N ** 2)
    energy += 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu) ** 2).real
    return energy.item()



def N_eps(U_np1, U_n, epsilon, gamma, c0):
    return 2 * gamma * c0 / epsilon * (U_np1 + U_n) * (1 - (torch.abs(U_np1) ** 2 + torch.abs(U_n) ** 2) / 2)



def fixpoint(U_0, L_eps, dt, N, epsilon, gamma, Nmax, tol, c0):
    
    _ones = torch.ones(N)

    G_m = (_ones - dt / 2 * L_eps)
    G_p = (_ones + dt / 2 * L_eps)

    CT = torch.fft.ifft2( G_m / G_p * torch.fft.fft2(U_0)).real


    U_n = U_0.clone()
    
    error = 10.0
    ii = 0
    conv = False
    
    print('max L:', torch.max(L_eps).item())
    print('max |CT|:', torch.max(torch.abs(CT)).item())
    print('mean |u0|:', torch.mean(torch.abs(U_0)).item())
    print('mean |u0|^2:', torch.mean(torch.abs(U_0)**2).item())


    while ii < Nmax and error > tol:

        non_linear = N_eps(U_n, U_0, epsilon, gamma, c0) # for fixed U_0 (initial image config.)

        print('max |NL|:', torch.max(torch.abs(non_linear)).item())

        U_np1 = torch.fft.ifft2( torch.fft.fft2(dt * non_linear) / G_p ).real + CT
        
        error = torch.max(torch.abs(U_np1 - U_n)).item()
        
        print("error", error)
        U_0 = U_n
        U_n = U_np1
        ii += 1

    
    if error < tol:
        conv = True

    return ii, U_n, error, conv