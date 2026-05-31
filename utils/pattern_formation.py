import torch

# ------------------------------------------------------------------
# helper functions for Magnetic Pattern formation via Reduced Micromagnetic Functional
# implemented via python torch arrays

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
    """
    Double well potential with normalization constant c0
    """
    u2 = 1 - torch.abs(u) ** 2
    return c0 * (u2 ** 2)

# ------------------------------------------------------------------

def double_well_prime(u, c0):
    """
    Analytic derivative of Double well potential
    """
    return -4.0 * c0 * u * (1.0 - u*u)

# ------------------------------------------------------------------


def laplacian2d(input_tensor):
    """
    2D Laplacian by convulution
    """
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
    return torch.nn.functional.conv2d(input_tensor, kernel, padding=1) # (padding=1 keeps same size)


def laplacian(u, dx, CONV = False):
    """
    2D Laplacian with periodic BC
    -> torch.roll() implementation (last element will be rolled over to first element)
    see.: https://docs.pytorch.org/docs/stable/generated/torch.roll.html
    """
    if not CONV:
        lap_u = (torch.roll(u, 1, dims=0) + torch.roll(u, -1, dims=0) + torch.roll(u, 1, dims=1) + torch.roll(u, -1, dims=1) - 4 * u) / dx**2
    else:
        lap_u = laplacian2d(u).squeeze(0).squeeze(0)
    
    return lap_u


def laplacian_neumann(u: torch.Tensor, dx: float) -> torch.Tensor:
    """
    2D Laplacian with VonNeumann BC
    """
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
    """
    define cartesian and wavevector space
    cartesian space : x
    wavevector space : k, modk, modk2
    """
    x = gridsize / N * torch.arange(N, dtype=dtype_real, device=device) # position array
    h = gridsize / N
    k = torch.fft.fftfreq(N, d=h).to(device) #* 2 * torch.pi -> excluded it for same structure as condette proposed (for consistent Fourier Multiplier)
    # returns the same as: torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device), torch.arange(-N // 2, 0, dtype=dtype_real, device = device)])

    kx, ky = torch.meshgrid(k, k, indexing='ij')
    modk2 = (kx**2 + ky**2).to(dtype_real)
    modk = torch.sqrt(modk2).to(dtype_real)
    
    return x, k, modk, modk2

# ------------------------------------------------------------------

def initialize_u0_random(N, REAL = True):
    """
    intialize a quadratic grid with uniform sampled values between [-1,+1]
    """
    amplitude = 1.0
    if REAL:
        u0 = amplitude * (2 * torch.rand(N, N, dtype=dtype_real, device=device) - 1) 
    else:
        u0 = amplitude * (2 * torch.rand(N, N, dtype=dtype_real, device=device) - 1) + amplitude * 1j * (2*torch.rand(N, N, dtype=dtype_real, device=device) - 1)
    return u0

# ------------------------------------------------------------------

def initialize_u0_sin(N, x, noise_level = 0.01):
    """
    initialize a quadratic grid with uniform sampled values between [-1,+1] modulated by a sin() function
    """
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
    gradient of energy functional via Finite Difference method
    """
    # Local FD Laplace
    if PBC:
        lap = laplacian(u, gridsize/N)
    else:
        lap = laplacian_neumann(u, gridsize/N)
    grad_loc = - (gamma * epsilon) * lap

    # Nonlocal Fourier multiplier
    Fu = torch.fft.fft2(u, norm='ortho')
    grad_nl = torch.fft.ifft2(sigma_k * Fu, norm='ortho').real

    grad_dw = 0
    if DW_TERM:
        # Double-well gradient
        grad_dw = (gamma / epsilon) * double_well_prime(u, c0)

    return grad_loc + grad_nl + grad_dw

# ------------------------------------------------------------------

def grad_fd_neumann_centered(u: torch.Tensor, dx: float):
    """
    gradient implemented with open boundary (for von neumann)
    """
    uy = torch.zeros_like(u)
    ux = torch.zeros_like(u)

    uy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*dx) # 2*dx spacing here
    ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dx)

    neuman_normal = 0.0  # one-sided near boundary (Neumann normal is 0)
    uy[0, :]  = neuman_normal
    uy[-1, :] = neuman_normal
    ux[:, 0]  = neuman_normal
    ux[:, -1] = neuman_normal
    return ux, uy


def grad_fd_pbc(u: torch.Tensor, dx : float):
    """
    gradient implemented with PBC 
    """
    uy = torch.zeros_like(u)
    ux = torch.zeros_like(u)
    
    ux = ( u - torch.roll(u, 1, 0) ) / dx
    uy = ( u - torch.roll(u, 1, 1) ) / dx
    return ux, uy


def energy_value_fd(u, sigma_k, N, gamma, epsilon, c0, PBC = True, RETURN_SEPERATE = False):
    """
    Energy functional with finite differences
    E = Gradient + DW + FM
    """

    dx = 1/N
    if PBC: # Periodic boundary condition
        ux, uy = grad_fd_pbc(u, dx)
    else:   # Von Neumann BC
        ux, uy = grad_fd_neumann_centered(u, dx)

    # local gradient energy 
    E_GRAD = 0.5 * (gamma * epsilon) * torch.sum(ux*ux + uy*uy) / (N**2) # normalized

    # nonlocal Fourier energy
    ftu = torch.fft.fft2(u, norm = 'ortho')
    E_FM = 0.5 * torch.sum(sigma_k * torch.abs(ftu)**2) / (N**2)

    # double-well energy
    W = double_well_potential(u, c0)
    E_DW = (gamma / epsilon) * torch.sum(W) / (N**2) # normalized

    if RETURN_SEPERATE:
        return E_GRAD.item(), E_DW.item(), E_FM.item()
    else:
        return (E_GRAD + E_DW + E_FM).item()


# ------------------------------------------------------------------


def energy_value(gamma, epsilon, N, u, c0, sigma_k, modk2, RETURN_SEPERATE = False):
    """
    Energy functional with spectral variant
    E = LaPlace + DW + FM 
    """
    ftu = torch.fft.fft2(u, norm = 'ortho')

    E_FM = 0.5 * torch.sum( sigma_k * torch.abs(ftu)**2 ) / N**2 # normalized FM energy

    # Condette used the definition of the laplace term in the discrete energy evaluation
    # both methods work, in my opinion we should use the gradient term but the spectral laplce also does its job in this case
    E_LP = 0.5 * torch.sum( gamma * epsilon * (2 * torch.pi)**2 * modk2 * torch.abs(ftu)**2 ) / (N**2) # normalized
    E_GRAD = E_LP # commented above the reason behind it, otherwise use the blank E_GRAD
    #ux, uy = grad_fd_pbc(u, 1/N)
    #E_GRAD = 0.5 * (gamma * epsilon) * torch.sum(ux*ux + uy*uy) / (N**2) # normalized

    W = double_well_potential(u, c0)
    E_DW = (gamma / epsilon) * torch.sum(W) / (N**2) # normalized

    if RETURN_SEPERATE:
        return E_GRAD.item(), E_DW.item(), E_FM.item()
    else:
        return (E_GRAD + E_DW + E_FM).item()



# ------------------------------------------------------------------

def energy_tensor(u, gamma, epsilon, N, th, modk, modk2, c0, sigma_k): 
    # same as energy_value but returns a torch scalar (not .item()) for Torch autograd backtracking as reference
    # just for review purposes
    W = double_well_potential(u, c0)
    ftu = torch.fft.fft2(u) / (N ** 2)                 
    S = sigma_k
    e1 = (gamma / epsilon) * torch.sum(W) / (N ** 2)    
    e2 = 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu) ** 2)
    return e1 + e2

# ------------------------------------------------------------------

def prox_h(v, tau, gamma, eps, c0, maxiter, tol):

    # --- proximal operator for double well potentialh(u) = (gamma/epsilon) * c0 * (1 - u^2)^2 ---
    # via vectorized Newton method, returns prox evaluated elementwise
    # Ref.: https://stackoverflow.com/questions/30191851/vectorize-a-newton-method-in-python-numpy
    # minimizes the following objective:
    # 0.5*(x - v)^2 + tau*(gamma/eps)*c0*(1 - x^2)^2   ->    min()

    lam = tau * (gamma / eps) * c0
    x = v.clone()

    for _ in range(maxiter):

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
