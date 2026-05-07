import torch
import numpy as np
from env_utils import tensor_type
from PIL import Image

# ------------------------------------------------------------------
# helper functions for Magnetic Pattern formation

# ------------------------------------------------------------------
# torch related
dtype_real = torch.float64
dtype_complex = torch.complex128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------------

def fft2_real(x):
    return torch.fft.fft2(x)

# ------------------------------------------------------------------

def ifft2_real(x_hat):
    return torch.fft.ifft2(x_hat).real

# ------------------------------------------------------------------

def fourier_multiplier(K):
    """
    Fourier Multiplier Dipolar
    """
    sig = torch.zeros_like(K)
    zero_freq = (torch.abs(K) < 1e-14)
    small = (torch.abs(K) >= 1e-14) & (torch.abs(K) < 1e-6)
    large = (torch.abs(K) >= 1e-6)
    
    sig[zero_freq] = 1
    sig[small] = 1 - torch.pi * torch.abs(K[small])
    sig[large] = (1 - torch.exp(-2 * torch.pi * torch.abs(K[large]))) / (2 * torch.pi * torch.abs(K[large]))
    
    return sig

# ------------------------------------------------------------------

def double_well_potential(u, c0):
    u2 = 1 - torch.abs(u) ** 2
    return c0 * (u2 ** 2)

# ------------------------------------------------------------------

def double_well_prime(u, c0):
    return -4.0 * c0 * u * (1.0 - u*u)

# ------------------------------------------------------------------

import torch.nn.functional as F

def laplacian2d(input_tensor):

    # Ensure input is (Batch, Channel, Height, Width) -> needed for Conv2D API
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # defined 3x3 kernel 
    kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                             [1.0, -4.0, 1.0],
                             [0.0, 1.0, 0.0]]]], dtype=input_tensor.dtype, device=input_tensor.device)

    # {Link: Conv2d https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv2d.html} [14]
    return F.conv2d(input_tensor, kernel, padding=1) # (padding=1 keeps same size)


def laplacian(u, dx, CONV = False):
    """
    with periodic BC because of the torch.roll() implementation (last element will be rolled over to first element)
    see.: https://docs.pytorch.org/docs/stable/generated/torch.roll.html
    """
    if not CONV:
        lap_u = (torch.roll(u, 1, dims=0) + torch.roll(u, -1, dims=0) + torch.roll(u, 1, dims=1) + torch.roll(u, -1, dims=1) - 4 * u) / dx**2
    else:
        lap_u = laplacian2d(u).squeeze(0).squeeze(0)
        #print(lap_u.shape)
    
    return lap_u


def laplacian_neumann(u: torch.Tensor, dx: float) -> torch.Tensor:

    # 5-point Laplacian with homogeneous Neumann BC (zero normal derivative).
    # https://www.12000.org/my_notes/neumman_BC/Neumman_BC.htm

    lap = torch.zeros_like(u)
    lap[1:-1, 1:-1] = u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]

    # edges are implemented via mirror across boundary / imaginary nodes represented by mirroring
    # top row i=0:      u[-1] -> u[1]   
    lap[0, 1:-1] = u[1, 1:-1] + u[1, 1:-1] + u[0, 2:] + u[0, :-2] - 4.0 * u[0, 1:-1]

    # bottom row i=H-1: u[H] -> u[H-2]
    lap[-1, 1:-1] = u[-2, 1:-1] + u[-2, 1:-1] + u[-1, 2:] + u[-1, :-2] - 4.0 * u[-1, 1:-1]

    # left col j=0: u[:, -1] -> u[:, 1]
    lap[1:-1, 0] = u[2:, 0] + u[:-2, 0] + u[1:-1, 1] + u[1:-1, 1] - 4.0 * u[1:-1, 0]

    # right col j=W-1: u[:, W] -> u[:, W-2]
    lap[1:-1, -1] = u[2:, -1] + u[:-2, -1] + u[1:-1, -2] + u[1:-1, -2] - 4.0 * u[1:-1, -1]

    # corners (mirror in both directions)
    lap[0, 0] = 2 * u[1, 0] + 2 * u[0, 1] - 4.0 * u[0, 0]
    lap[0, -1] = 2 * u[1, -1] + 2 * u[0, -2] - 4.0 * u[0, -1]
    lap[-1, 0] = 2 * u[-2, 0] + 2 * u[-1, 1] - 4.0 * u[-1, 0]
    lap[-1, -1] = 2 * u[-2, -1] + 2 * u[-1, -2] - 4.0 * u[-1, -1]

    return lap / (dx ** 2)


# ------------------------------------------------------------------

def define_spaces(gridsize, N):
    # -- k, x, kx, ky, modk & modk2 --

    x = gridsize / N * torch.arange(N, dtype=dtype_real, device=device) # position array
    h = gridsize / N
    k = torch.fft.fftfreq(N, d=h).to(device) #* 2 * torch.pi -> excluded it for same structure as condette proposed (for consistent Fourier Multiplier)
    # the same as: torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device), torch.arange(-N // 2, 0, dtype=dtype_real, device = device)])

    kx, ky = torch.meshgrid(k, k, indexing='ij')
    modk2 = (kx**2 + ky**2).to(dtype_real)
    modk = torch.sqrt(modk2).to(dtype_real)
    
    return x, k, modk, modk2

# ------------------------------------------------------------------

def define_spaces_adapted(gridsize, N):
    
    x = gridsize / N * torch.arange(N, dtype=dtype_real, device=device) # position array

    SCALING_FACTOR = 1

    k = torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device)/SCALING_FACTOR, torch.arange(-N // 2, 0, dtype=dtype_real, device = device)/SCALING_FACTOR])

    PERIODIC_ = False
    if PERIODIC_:
        period_factor=1.0
        amplitude=1.0
        k = amplitude * torch.sin(2 * torch.pi * k / (N / period_factor))

    EXP = True
    if EXP:
        exp_base=1.2
        k_pos = torch.logspace(0, np.log(N//2)/np.log(exp_base), N//2, 
                            base=exp_base, dtype=dtype_real, device=device)

        # mirror to negative side
        k_neg = -torch.flip(k_pos, dims=[0])

        # concatenate negative and positive
        k = torch.cat([k_pos, k_neg])

    xi, eta = torch.meshgrid(k, k, indexing='ij')
    modk2 = (xi ** 2 + eta ** 2).to(dtype_real)
    modk = torch.sqrt(modk2).to(dtype_real)
    return x, k, modk, modk2

# ------------------------------------------------------------------

def initialize_u0_random(N, REAL = True):
    amplitude = 10.0
    if REAL:
        u0 = amplitude * (2 * torch.rand(N, N, dtype=dtype_real, device=device) - 1) 
    else:
        u0 = amplitude * (2 * torch.rand(N, N, dtype=dtype_real, device=device) - 1) + amplitude * 1j * (2*torch.rand(N, N, dtype=dtype_real, device=device) - 1)
    return u0

# ------------------------------------------------------------------

def initialize_u0_sin(N, x, noise_level = 0.01):
    x1, x2 = torch.meshgrid(x, x)

    x1 = x1 + noise_level * (torch.rand(N, N, dtype=dtype_real, device=device) - 0.5)
    x2 = x2 + noise_level * (torch.rand(N, N, dtype=dtype_real, device=device) - 0.5)

    u0 = torch.sin(8 * torch.pi * x1) * torch.sin(8 * torch.pi * x2)
    return u0

# ------------------------------------------------------------------

def grad_g(u, M_k):
    """
    gradient of g(u) via spectral multiplication 
    grad_g = iFFT[ ( FM(|k|) + gamma*eps*|k|² ) * FFT(u) ] 
    """
    Fu = torch.fft.fft2(u, norm='ortho')
    return torch.fft.ifft2(M_k * Fu, norm='ortho').real

# ------------------------------------------------------------------

def grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC = True, DW_TERM = False):
    """
    Gradient of Energy functional with Finite Difference method
    """
    # Local FD gradient (–γ ε Δu)
    if PBC:
        lap = laplacian(u, gridsize/N)
    else:
        lap = laplacian_neumann(u, gridsize/N)
    grad_loc = - (gamma * epsilon) * lap

    # Nonlocal gradient (σ_k * Fu)
    Fu = torch.fft.fft2(u, norm='ortho')
    grad_nl = torch.fft.ifft2(sigma_k * Fu, norm='ortho').real

    grad_dw = 0
    if DW_TERM:
        # Double-well gradient ((γ/ε) W′(u))
        grad_dw = (gamma / epsilon) * double_well_prime(u, c0)

    return grad_loc + grad_nl + grad_dw

# ------------------------------------------------------------------

def grad_fd_neumann_centered(u: torch.Tensor, dx: float):
    # gradient with open boundary (for von neumann)

    uy = torch.zeros_like(u)
    ux = torch.zeros_like(u)

    uy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*dx) # 2*dx spacing here
    ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dx)

    # one-sided near boundary (Neumann normal is 0)
    neuman_normal = 0.0
    uy[0, :]  = neuman_normal
    uy[-1, :] = neuman_normal
    ux[:, 0]  = neuman_normal
    ux[:, -1] = neuman_normal
    return ux, uy


def grad_fd_pbc(u: torch.Tensor, dx : float):
    uy = torch.zeros_like(u)
    ux = torch.zeros_like(u)
    
    ux = ( u - torch.roll(u, 1, 0) ) / dx
    uy = ( u - torch.roll(u, 1, 1) ) / dx
    return ux, uy


def energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC = True, RETURN_SEPERATE = False):
    """
    Energy functional with finite differences
    E = LaPlace + DW + FM
    """

    dx = 1/N
    if PBC: # Periodic boundary condition
        ux, uy = grad_fd_pbc(u, dx)
    else:   # Von Neumann BC
        ux, uy = grad_fd_neumann_centered(u, dx)

    # local gradient energy 
    E_GRAD = 0.5 * (gamma * epsilon) * torch.sum(ux*ux + uy*uy) / (N**2) # normalized

    # nonlocal Fourier energy
    ftu = torch.fft.fft2(u, norm = 'ortho') / (N**2) # normalized
    E_FM = 0.5 * torch.sum(sigma_k * torch.abs(ftu)**2)

    # double-well energy
    W = double_well_potential(u, c0)
    E_DW = (gamma / epsilon) * torch.sum(W) / N**2 # normalized

    if RETURN_SEPERATE:
        return E_GRAD, E_DW, E_FM
    else:
        return (E_GRAD + E_DW + E_FM).item()

# ------------------------------------------------------------------

def energy_value(gamma, epsilon, N, u, M_k, c0):
    """
    Energy functional with spectral variant
    E = LaPlace + DW + FM 
    """

    W = double_well_potential(u, c0)
    ftu = torch.fft.fft2(u, norm = 'ortho') #/ N**2
    
    E_DW = (gamma / epsilon) * torch.sum(W) / N**2 
    E_LPFM = 0.5 * torch.sum( M_k * torch.abs(ftu)**2 )

    return (E_LPFM + E_DW).item()

# ------------------------------------------------------------------

def energy_tensor(u, gamma, epsilon, N, th, modk, modk2, c0, sigma_k): 
    # same as energy_value but returns a torch scalar (not .item()) for Torch autograd backtracking as reference
    W = double_well_potential(u, c0)
    ftu = torch.fft.fft2(u) / (N ** 2)                 
    S = sigma_k
    e1 = (gamma / epsilon) * torch.sum(W) / (N ** 2)    
    e2 = 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu) ** 2)
    return e1 + e2

# ------------------------------------------------------------------

def prox_h(v, tau, gamma, eps, c0, maxiter, tol):

    # --- proximal operator for h(x) = (gamma/epsilon) * c0 * (1 - x^2)^2 ---
    # via vectorized Newton method, returns prox evaluated elementwise
    # Ref.: https://stackoverflow.com/questions/30191851/vectorize-a-newton-method-in-python-numpy
    # minimize 0.5*(x-v)^2 + tau*(gamma/eps)*c0*(1-x^2)^2

    lam = tau * (gamma / eps) * c0
    x = v.clone()

    for i in range(maxiter):

        grad = x - v - 4.0 * lam * x * (1.0 - x * x)
        hess = 1.0 + lam * (4.0 * (x * x - 1) + 8.0 * x * x)
        hess_safe = torch.where(torch.abs(hess) < 1e-12, torch.sign(hess) * 1e-12, hess)
        step = grad / hess_safe # ratio for newtons method

        # damped update (clamp step to avoid runaway)
        # use backtracking-like damping factor to ensure phi decreases (simple safeguard)
        # Ref.: claude.ai + stackoverflow
        alpha = 1.0
        x_new = x - alpha * step

        max_jump = 0.5
        delta = x_new - x
        overshoot = torch.abs(delta) > max_jump
        if overshoot.any():
            # scale down the step where overshooting
            scale = max_jump / (torch.abs(delta) + 1e-16)
            x_new = x + delta * torch.where(overshoot, scale, torch.ones_like(scale))

        # check convergence (max abs difference)
        if torch.max(torch.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    return x

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Condette

def N_eps(U_np1, U_n, epsilon, gamma, c0):
    return 2 * gamma * c0 / epsilon * (U_np1 + U_n) * (1 - (torch.abs(U_np1) ** 2 + torch.abs(U_n) ** 2) / 2)

# ------------------------------------------------------------------

def fixpoint(U_0, L_eps, dt, N, epsilon, gamma, Nmax, tol, c0):
    DEBUG = False

    _ones = torch.ones(N)

    G_m = (_ones - dt / 2 * L_eps)
    G_p = (_ones + dt / 2 * L_eps)

    CT = torch.fft.ifft2( G_m / G_p * torch.fft.fft2(U_0)).real

    U_n = U_0.clone()    
    error = 10.0
    ii = 0
    conv = False

    energies_fixpoint = []

    if DEBUG:
        print('max L:', torch.max(L_eps).item())
        print('max |CT|:', torch.max(torch.abs(CT)).item())
        print('mean |u0|:', torch.mean(torch.abs(U_0)).item())
        print('mean |u0|^2:', torch.mean(torch.abs(U_0)**2).item())


    while ii < Nmax and error > tol:

        non_linear = N_eps(U_n, U_0, epsilon, gamma, c0) # for fixed U_0 (initial image config.)

        if DEBUG:
            print('max |NL|:', torch.max(torch.abs(non_linear)).item())

        U_np1 = torch.fft.ifft2( torch.fft.fft2(dt * non_linear) / G_p ).real + CT
        error = torch.max(torch.abs(U_np1 - U_n)).item()

        U_0 = U_n
        U_n = U_np1
        ii += 1

        curr_energy = energy_value(gamma, epsilon, N, U_n, L_eps, c0)
        energies_fixpoint.append(curr_energy)

    if error < tol:
        conv = True

    return ii, U_n, error, conv, energies_fixpoint

# ------------------------------------------------------------------