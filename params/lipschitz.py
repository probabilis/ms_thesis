import torch
import math
from utils.pattern_formation import dtype_real, fourier_multiplier, device, define_spaces



def evaluate_lipschitz_constant(gamma, eps, N, gridsize, th = 1.0, DEBUG = False):

    h = gridsize / N
    x, k, modk, modk2 = define_spaces(gridsize, N)

    # Discrete Laplacian eigenvalues (periodic 2D) / indices 0..N-1
    idx = torch.arange(N)
    kx = math.pi * idx / N  # kx*h/2 = π i/N => we store θ = kx*h/2
    ky = math.pi * idx / N
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")

    lambda_lap = -4.0 / h**2 * (torch.sin(KX)**2 + torch.sin(KY)**2)
    rho_lap = torch.max(torch.abs(lambda_lap))  # ≈ 8/h^2

    # -------------------------------------------

    # 1) LaPlace term
    L_laplace = gamma * eps * rho_lap

    # 2) Fourier Multiplier Term
    sigma_hat = fourier_multiplier(th * k).to(dtype_real).to(device)
    # Nonlocal operator norm
    print("sigma", sigma_hat)
    L_fouriermult = torch.max(torch.abs(sigma_hat))

    # 3) Double Well term
    # Nonlinear double-well bound (on u in [-1,1])
    Wpp_max = 2.25
    L_double_well = gamma * Wpp_max / eps

    # total
    L_total = L_laplace + L_fouriermult + L_double_well
    eta_safe = 1.0 / L_total

    if DEBUG:
        print("LaPlace", rho_lap)
        print("8/h^2", 8/h**2)
        print("LaPlace", L_laplace)
        print("Fourier Multiplier", L_fouriermult )
        print("Double well", L_double_well)
        print("Lipschitz upper bound L =", L_total.item())
        print("Safe step size eta ~", eta_safe.item())

    return float(eta_safe)


if __name__ == "__main__":
    print("Lipschitz constant test: ")
    evaluate_lipschitz_constant(gamma = 0.002, eps = 0.01, N = 100, gridsize = 1.0, DEBUG=True)