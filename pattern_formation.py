import torch
import numpy as np
from env_utils import tensor_type
from PIL import Image

# ------------------------------------------------------------------
# torch related
dtype_real = torch.float64
dtype_complex = torch.complex128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------------

def fft2_real(x):
    return torch.fft.fft2(x)

def ifft2_real(x_hat):
    return torch.fft.ifft2(x_hat).real

# ------------------------------------------------------------------

def fourier_multiplier(A):
    sig = torch.zeros_like(A)
    zero_freq = (torch.abs(A) < 1e-14)
    small = (torch.abs(A) >= 1e-14) & (torch.abs(A) < 1e-6)
    large = (torch.abs(A) >= 1e-6)
    
    sig[zero_freq] = 1
    sig[small] = 1 - np.pi * torch.abs(A[small])
    sig[large] = (1 - torch.exp(-2 * np.pi * torch.abs(A[large]))) / (2 * np.pi * torch.abs(A[large]))
    
    return sig


def double_well_potential(u, c0):
    u2 = 1 - torch.abs(u) ** 2
    return c0 * (u2 ** 2)


def double_well_prime(u, c0):
    return -4.0 * c0 * u * (1.0 - u*u)

def laplacian(u, dx):
    # with periodic BC
    lap_u = (torch.roll(u, 1, dims=0) + torch.roll(u, -1, dims=0) +
             torch.roll(u, 1, dims=1) + torch.roll(u, -1, dims=1) - 4 * u) / dx**2
    return lap_u

# ------------------------------------------------------------------

def define_spaces(gridsize, N):
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
    k = torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device), torch.arange(-N // 2, 0, dtype=dtype_real, device = device)])

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


def initialize_u0_image(file_path):
    image = Image.open(file_path).convert('L')
    np_array = np.array(image)
    u0 = torch.from_numpy(np_array).float() / 255.0  # PyTorch tensor -> normalize to [0.0, 1.0]
    print(u0.shape)
    return u0


def initialize_u0_sin(N, x, noise_level = 0.01):
    x1, x2 = torch.meshgrid(x, x)

    x1 = x1 + noise_level * (torch.rand(N, N, dtype=dtype_real, device=device) - 0.5)
    x2 = x2 + noise_level * (torch.rand(N, N, dtype=dtype_real, device=device) - 0.5)

    u0 = torch.sin(8 * torch.pi * x1) * torch.sin(8 * torch.pi * x2)
    return u0

# ------------------------------------------------------------------

def grad_g(u, M_k):
    # gradient of g(u) via spectral multiplication
    Fu = fft2_real(u)          
    grad_hat = M_k * Fu
    grad_real = ifft2_real(grad_hat)
    return grad_real

# ------------------------------------------------------------------

def energy_value(gamma, epsilon, N, u, th, modk, modk2, c0):
    W = double_well_potential(u, c0)
    ftu = torch.fft.fft2(u) / (N ** 2)

    S = fourier_multiplier(th * modk)
    #print('max sigma:', torch.max(S).item())
    #print('min sigma:', torch.min(S).item())
    #print('mean sigma:', torch.mean(S).item())

    energy = (gamma / epsilon) * torch.sum(W) / (N ** 2)
    energy += 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu) ** 2)
    return energy.item()

def energy_tensor(u, gamma, epsilon, N, th, modk, modk2, c0, sigma_k): 
    # returns a torch scalar (not .item()), matching your energy_value implementation
    # NB: keep operations in torch (no .item())
    W = double_well_potential(u, c0)                    # tensor
    ftu = torch.fft.fft2(u) / (N ** 2)                   # match your energy_value style
    S = sigma_k
    e1 = (gamma / epsilon) * torch.sum(W) / (N ** 2)     # use same normalization as energy_value
    e2 = 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu) ** 2)
    return e1 + e2


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
    
    #print('max L:', torch.max(L_eps).item())
    #print('max |CT|:', torch.max(torch.abs(CT)).item())
    #print('mean |u0|:', torch.mean(torch.abs(U_0)).item())
    #print('mean |u0|^2:', torch.mean(torch.abs(U_0)**2).item())


    while ii < Nmax and error > tol:

        non_linear = N_eps(U_n, U_0, epsilon, gamma, c0) # for fixed U_0 (initial image config.)

        #print('max |NL|:', torch.max(torch.abs(non_linear)).item())

        U_np1 = torch.fft.ifft2( torch.fft.fft2(dt * non_linear) / G_p ).real + CT
        
        error = torch.max(torch.abs(U_np1 - U_n)).item()
        
        #print("error", error)
        U_0 = U_n
        U_n = U_np1
        ii += 1

    
    if error < tol:
        conv = True

    return ii, U_n, error, conv


# -------------------------------------