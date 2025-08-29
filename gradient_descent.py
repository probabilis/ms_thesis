import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pattern_formation import fourier_multiplier, energy_value, dtype_complex, dtype_real, device
from params import labyrinth_data_params, sim_params, get_DataParameters, get_SimulationParamters, sin_data_params
from main import initialize_u0_random


folder_path = r"out_gd/"

# -----------------------------------------------------

L = 1.0
N = 100

epsilon = 1/20
gamma = 1/200

c0 = 9/32

alpha = 1e-7 # learning rate
num_iter = 1_000_000


#k = np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])
k = torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device), torch.arange(-N // 2, 0, dtype=dtype_real, device = device)])
xi, eta = torch.meshgrid(k, k, indexing='ij')
modk2 = (xi ** 2 + eta ** 2).to(dtype_real)
modk = torch.sqrt(modk2).to(dtype_real)


# -----------------------------------------------------


def double_well_potential(u, c0):
    u2 = 1 - torch.abs(u) ** 2
    return c0 * (u2 ** 2)

def double_well_prime(u, c0):
    return -2 * c0 * u * (1 - torch.abs(u)**2)


def laplacian(u, dx):
    # with periodic BC
    lap_u = (torch.roll(u, 1, dims=0) + torch.roll(u, -1, dims=0) +
             torch.roll(u, 1, dims=1) + torch.roll(u, -1, dims=1) - 4 * u) / dx**2
    return lap_u

# -----------------------------------------------------

if __name__ == "__main__":
    sigma_k = fourier_multiplier(modk)
    u = initialize_u0_random(N)

    h = L / N
    th = 1

    energies = []

    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)

    plt.ion()

    LACPLACE_SPECTRAL = True
    LIVE_PLOT = False

    try:
        # -- Gradient descent looop --
        for n in tqdm(range(num_iter)):

            # 1. Local gradient term
            lap_u = torch.zeros_like(u)
            # via spectral method
            if LACPLACE_SPECTRAL:
                lap_u = torch.fft.ifft2(-modk2 * torch.fft.fft2(u)).real
            # via finite difference
            #else:
                lap_u = laplacian(u, h)

            grad_local = - epsilon * lap_u

            # 2. Double well potential derivative
            Wprime = double_well_prime(u, c0)
            grad_double_well = (1/epsilon) * Wprime

            # 3. Nonlocal gradient term
            Fu = torch.fft.fft2(u)
            nonlocal_term = torch.fft.ifft2(sigma_k * Fu).real

            # 4. Total gradient
            grad_E = grad_local + grad_double_well + nonlocal_term

            # 5. Gradient update
            u -= alpha * grad_E

            E = energy_value(gamma,epsilon,N,u,th,modk,modk2,c0)
            energies.append(E)

            if LIVE_PLOT and (n % 1000) == 0:
                ax1.clear()
                ax2.clear()
                #print(f"Iteration {n}")
                ax1.imshow(u.real, cmap='gray',extent=(0, 1, 0, 1))
                ax1.set_title(f"Iteration {n}")
                fig1.savefig(folder_path + f"image_graddescent_N={N}_nmax={num_iter}_alpha={alpha}_gamma={gamma}_eps={epsilon}.png")
                
                ax2.plot(torch.arange(0,len(energies), 1), energies)
                #ax2.set_yscale('log')

                ax2.set_title("energy evolution")
                fig2.savefig(folder_path + f"energy_graddescent_N={N}_nmax={num_iter}_alpha={alpha}_gamma={gamma}_eps={epsilon}.png")
                
                plt.pause(0.1)


    except KeyboardInterrupt:
        print("Exit.")  


    ax1.imshow(u.real, cmap='gray',extent=(0, 1, 0, 1))
    ax1.set_title(f"Iteration {n}")
    fig1.savefig(folder_path + f"image_graddescent_N={N}_nmax={num_iter}_alpha={alpha}_gamma={gamma}_eps={epsilon}.png")

    ax2.plot(torch.arange(0,len(energies), 1), energies)
    #ax2.set_yscale('log')

    ax2.set_title("energy evolution")
    fig2.savefig(folder_path + f"energy_graddescent_N={N}_nmax={num_iter}_alpha={alpha}_gamma={gamma}_eps={epsilon}.png")
