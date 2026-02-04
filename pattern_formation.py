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

def laplacian(u, dx):
    """
    with periodic BC because of the torch.roll() implementation (last element will be rolled over to first element)
    see.: https://docs.pytorch.org/docs/stable/generated/torch.roll.html
    """
    lap_u = (torch.roll(u, 1, dims=0) + torch.roll(u, -1, dims=0) +
             torch.roll(u, 1, dims=1) + torch.roll(u, -1, dims=1) - 4 * u) / dx**2
    return lap_u


def laplacian_neumann(u: torch.Tensor, dx: float) -> torch.Tensor:
    """
    2D 5-point Laplacian with homogeneous Neumann BC (zero normal derivative).
    Implemented by mirroring the boundary-adjacent values (reflect/replicate),
    i.e., ghost cells satisfy u[-1]=u[1], u[N]=u[N-2], etc.

    u: shape (H, W) (or any tensor where the last two dims are y,x if you adapt dims)
    dx: grid spacing (assumed same in both directions)
    """

    lap = torch.zeros_like(u)

    # interior
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
    )

    # edges (Neumann: mirror across boundary)
    # top row (i=0): u[-1] -> u[1]
    lap[0, 1:-1] = (
        u[1, 1:-1] + u[1, 1:-1] + u[0, 2:] + u[0, :-2] - 4.0 * u[0, 1:-1]
    )
    # bottom row (i=H-1): u[H] -> u[H-2]
    lap[-1, 1:-1] = (
        u[-2, 1:-1] + u[-2, 1:-1] + u[-1, 2:] + u[-1, :-2] - 4.0 * u[-1, 1:-1]
    )
    # left col (j=0): u[:, -1] -> u[:, 1]
    lap[1:-1, 0] = (
        u[2:, 0] + u[:-2, 0] + u[1:-1, 1] + u[1:-1, 1] - 4.0 * u[1:-1, 0]
    )
    # right col (j=W-1): u[:, W] -> u[:, W-2]
    lap[1:-1, -1] = (
        u[2:, -1] + u[:-2, -1] + u[1:-1, -2] + u[1:-1, -2] - 4.0 * u[1:-1, -1]
    )

    # corners (mirror in both directions)
    lap[0, 0] = (u[1, 0] + u[1, 0] + u[0, 1] + u[0, 1] - 4.0 * u[0, 0])
    lap[0, -1] = (u[1, -1] + u[1, -1] + u[0, -2] + u[0, -2] - 4.0 * u[0, -1])
    lap[-1, 0] = (u[-2, 0] + u[-2, 0] + u[-1, 1] + u[-1, 1] - 4.0 * u[-1, 0])
    lap[-1, -1] = (u[-2, -1] + u[-2, -1] + u[-1, -2] + u[-1, -2] - 4.0 * u[-1, -1])

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

def initialize_u0_random(N, REAL = False):
    amplitude = 0.1
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

def grad_fd(u, sigma_k, N, gridsize, gamma, epsilon, c0, PBC = True):
    """
    Gradient of Energy functional with Finite Difference method
    """
    PBC = True
    # Local FD gradient (–γ ε Δu)
    if PBC:
        lap = laplacian(u, gridsize/N)
    else:
        lap = laplacian_neumann(u, gridsize/N)
    grad_loc = -(gamma * epsilon) * lap

    # Nonlocal gradient (σ_k * Fu)
    Fu = torch.fft.fft2(u, norm='ortho')
    grad_nl = torch.fft.ifft2(sigma_k * Fu, norm='ortho').real

    # Double-well gradient ((γ/ε) W′(u))
    grad_dw = (gamma / epsilon) * double_well_prime(u, c0)

    return grad_loc + grad_nl + grad_dw

# ------------------------------------------------------------------

def grad_neumann_centered(u: torch.Tensor, dx: float):
    uy = torch.zeros_like(u)
    ux = torch.zeros_like(u)

    # centered interior (more stable than )
    uy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*dx)
    ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dx)

    # one-sided near boundary (then enforce Neumann normal=0)
    uy[0, :]  = 0.0
    uy[-1, :] = 0.0
    ux[:, 0]  = 0.0
    ux[:, -1] = 0.0

    return uy, ux


def energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC = True):
    """
    Energy functional with finite difference
    """
    if PBC: # Periodic boundary condition
        ux = u - torch.roll(u, 1, 0)
        uy = u - torch.roll(u, 1, 1)
    
    else:
        uy, ux = grad_neumann_centered(u, 1/N)

    # local gradient energy
    E_loc = 0.5 * (gamma * epsilon) * torch.sum(ux*ux + uy*uy) / (N**2)

    # nonlocal Fourier energy
    ftu = torch.fft.fft2(u) / (N**2)
    E_nl = 0.5 * torch.sum(sigma_k * torch.abs(ftu)**2)

    # double-well energy
    W = double_well_potential(u, c0)
    E_dw = (gamma / epsilon) * torch.sum(W) / N**2

    return (E_loc + E_nl + E_dw).item()

# ------------------------------------------------------------------

def energy_value(gamma, epsilon, N, u, M_k, c0):
    """
    E = DW + LaPlace + FM
    spectral variant
    """

    W = double_well_potential(u, c0)
    ftu = torch.fft.fft2(u) / N**2
    
    energy = (gamma / epsilon) * torch.sum(W) / N**2
    energy += 0.5 * torch.sum( M_k * torch.abs(ftu)**2 )

    return energy.item()

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

    if error < tol:
        conv = True

    return ii, U_n, error, conv

# ------------------------------------------------------------------