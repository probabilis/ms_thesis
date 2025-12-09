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

def fourier_multiplier(A):
    """
    Fourier Multiplier Dipolar
    """
    sig = torch.zeros_like(A)
    zero_freq = (torch.abs(A) < 1e-14)
    small = (torch.abs(A) >= 1e-14) & (torch.abs(A) < 1e-6)
    large = (torch.abs(A) >= 1e-6)
    
    sig[zero_freq] = 1
    sig[small] = 1 - torch.pi * torch.abs(A[small])
    sig[large] = (1 - torch.exp(-2 * torch.pi * torch.abs(A[large]))) / (2 * torch.pi * torch.abs(A[large]))
    
    return sig

# ------------------------------------------------------------------

def fourier_multiplier_simple(k_mag):
    """ 
    1 / k fourier multiplier
    """
    k_safe = torch.where(k_mag < 1e-12, torch.tensor(1e-12), k_mag)
    return 1 / k_safe

# ------------------------------------------------------------------

def fourier_multiplier_dipolar(k_mag, th):
    """
    Dipolar interaction kernel for magnetic thin films
    For thin films, the demagnetization factor depends on thickness th
    
    Physical form: σ(k) = 1 - exp(-|k|*th) for thin films
    """
    # Avoid numerical issues at k=0
    k_safe = torch.where(k_mag < 1e-12, torch.tensor(1e-12, dtype=dtype_real, device=device), k_mag)
    
    # Dipolar kernel for thin films
    sigma = 1.0 - torch.exp(-k_safe * th)
    
    # Handle k=0 case properly
    sigma = torch.where(k_mag < 1e-12, torch.tensor(0.0, dtype=dtype_real, device=device), sigma)
    
    return sigma

# ------------------------------------------------------------------

def double_well_potential(u, c0):
    u2 = 1 - torch.abs(u) ** 2
    return c0 * (u2 ** 2)

# ------------------------------------------------------------------

def double_well_prime(u, c0):
    return -4.0 * c0 * u * (1.0 - u*u)

# ------------------------------------------------------------------

def laplacian(u, dx):
    # with periodic BC
    lap_u = (torch.roll(u, 1, dims=0) + torch.roll(u, -1, dims=0) +
             torch.roll(u, 1, dims=1) + torch.roll(u, -1, dims=1) - 4 * u) / dx**2
    return lap_u

# ------------------------------------------------------------------

def define_spaces(gridsize, N, LAPLACE_SPECTRAL = False):
    x = gridsize / N * torch.arange(N, dtype=dtype_real, device=device) # position array
    
    """
    k_scaled = False
    if k_scaled:
        k = torch.fft.fftfreq(N, d = 1/N).to(device) * 2 * torch.pi # FFT frequencies scaled
    else:
        k = torch.fft.fftfreq(N, d = 1/N).to(device) 
    """

    # -- k, x, xi, eta, modk & modk2 --
    #k = np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])
    
    # torch.arange(-N // 2 + 1, 1) ... # +[0..+N,-N..0]
    
    #k = torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device), torch.arange(-N // 2, 0, dtype=dtype_real, device = device)])
    h = gridsize / N
    if LAPLACE_SPECTRAL:
        k = torch.fft.fftfreq(N, d=h).to(device)
    else:
        k = torch.fft.fftfreq(N, d=h).to(device) * 2 * torch.pi
    print("k", k)

    xi, eta = torch.meshgrid(k, k, indexing='ij')
    modk2 = (xi ** 2 + eta ** 2).to(dtype_real)
    modk = torch.sqrt(modk2).to(dtype_real)
    
    return x, k, modk, modk2

# ------------------------------------------------------------------

def define_spaces_adapted(gridsize, N):
    x = gridsize / N * torch.arange(N, dtype=dtype_real, device=device) # position array
    
    """
    k_scaled = False
    if k_scaled:
        k = torch.fft.fftfreq(N, d = 1/N).to(device) * 2 * torch.pi # FFT frequencies scaled
    else:
        k = torch.fft.fftfreq(N, d = 1/N).to(device) 
    """

    # -- k, x, xi, eta, modk & modk2 --
    #k = np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])

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

def initialize_u0_image(file_path):
    image = Image.open(file_path).convert('L')
    np_array = np.array(image)
    u0 = torch.from_numpy(np_array).float() / 255.0  # PyTorch tensor -> normalize to [0.0, 1.0]
    #print(u0.shape)
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
    # gradient of g(u) via spectral multiplication
    # 
    # grad_g = iFFT[ ( FM(|k|) + gamma*eps*|k|² ) * FFT(u) ] 
    #
    #Fu = torch.fft.fft2(u, norm='ortho') 
    Fu = fft2_real(u)
    grad_hat = M_k * Fu
    return ifft2_real(grad_hat)
    #return torch.fft.ifft2(M_k * Fu, norm='ortho').real

# ------------------------------------------------------------------

def grad_fd_mix(u, sigma_k, N, gamma, epsilon, c0):
    dx = 1.0 / N

    # Local FD gradient (–γ ε Δu)
    lap = laplacian(u, dx)
    grad_loc = -(gamma * epsilon) * lap #* 0.1

    # Nonlocal gradient (σ_k * Fu)
    Fu = torch.fft.fft2(u, norm='ortho')
    grad_nl = torch.fft.ifft2(sigma_k * Fu, norm='ortho').real

    # Double-well gradient ((γ/ε) W′(u))
    grad_dw = (gamma / epsilon) * double_well_prime(u, c0)

    return grad_loc + grad_nl + grad_dw

# ------------------------------------------------------------------

def energy_value_fd_mix(u, sigma_k, N, gamma, epsilon, c0):
    # local FD energy

    dx = 1.0 / N

    ux = u - torch.roll(u, 1, 0)
    uy = u - torch.roll(u, 1, 1)

    E_loc = 0.5 * (gamma * epsilon) * torch.sum(ux*ux + uy*uy) / (dx**2)

    # nonlocal Fourier energy (same normalization as your code)
    ftu = torch.fft.fft2(u) / (N**2)
    # Fu = torch.fft.fft2(u, norm='ortho')
    E_nl = 0.5 * torch.sum(sigma_k * torch.abs(ftu)**2)

    # double-well energy
    W = c0 * (1 - u*u)**2
    E_dw = (gamma / epsilon) * torch.sum(W)

    return (E_loc + E_nl + E_dw).item()

# ------------------------------------------------------------------

def energy_value(gamma, epsilon, N, u, th, modk, modk2, c0):
    DEBUG = False

    W = double_well_potential(u, c0)
    ftu = torch.fft.fft2(u) / (N ** 2)
    S = fourier_multiplier(th * modk)

    energy = (gamma / epsilon) * torch.sum(W) / (N ** 2)
    energy += 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu) ** 2)

    if DEBUG:
        print('max modk:', torch.max(modk).item())
        print('min modk:', torch.min(modk).item())
        print('mean modk:', torch.mean(modk).item())
        print('max sigma:', torch.max(S).item())
        print('min sigma:', torch.min(S).item())
        print('mean sigma:', torch.mean(S).item())

    return energy.item()

# ------------------------------------------------------------------

def energy_tensor(u, gamma, epsilon, N, th, modk, modk2, c0, sigma_k): 
    # same as energy_value but returns a torch scalar (not .item()) for autograd backtracking
    W = double_well_potential(u, c0)
    ftu = torch.fft.fft2(u) / (N ** 2)                 
    S = sigma_k
    e1 = (gamma / epsilon) * torch.sum(W) / (N ** 2)    
    e2 = 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu) ** 2)
    return e1 + e2

# ------------------------------------------------------------------

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