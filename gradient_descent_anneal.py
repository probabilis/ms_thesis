import torch
import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import laplacian, double_well_prime

from pattern_formation import fourier_multiplier, energy_value, dtype_complex, dtype_real, device
from main import initialize_u0_random

# ---------------------------------------------------------------
# parameters

N = 128                   
L = 1.0   
th = 1.0           
dx = L / N             
epsilon = 1/20          
gamma =1/200           
c0 = 9/32                
dt = 0.000001                
num_steps = 200_000          

# annealing parameters
sigma0 = 0.001             
lambda_anneal = 1e-4 

# ---------------------------------------------------------------

# field
u = initialize_u0_random(N)

#k = np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])
k = torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device), torch.arange(-N // 2, 0, dtype=dtype_real, device = device)])
xi, eta = torch.meshgrid(k, k, indexing='ij')
modk2 = (xi ** 2 + eta ** 2).to(dtype_real)
modk = torch.sqrt(modk2).to(dtype_real)

S = fourier_multiplier(modk)

# ---------------------------------------------------------------

energy_history = []

fig1, ax1 = plt.subplots(1,1)
fig2, ax2 = plt.subplots(1,1)

LACPLACE_SPECTRAL = True


for step in range(num_steps):

    sigma_noise = sigma0 * np.exp(-lambda_anneal * step * dt)
    print("sigma noise", sigma_noise)
    # Local gradient term
    lap_u = torch.zeros_like(u)
    # via spectral method
    if LACPLACE_SPECTRAL:
        lap_u = torch.fft.ifft2(-modk2 * torch.fft.fft2(u)).real
    else:
        lap_u = laplacian(u, dx)
    
    local_term = -epsilon * lap_u + (gamma / epsilon) * double_well_prime(u, c0)
    
    # Non-local gradient term
    u_hat = torch.fft.fft2(u) / (N**2)
    nonlocal_hat = (S + gamma * epsilon * modk2) * u_hat
    nonlocal_term = torch.fft.ifft2(nonlocal_hat).real * (N**2)
    
    # Gradient descent step with annealing
    gradE = local_term + nonlocal_term
    noise = sigma_noise * torch.randn(N, N)

    # update
    u -= dt * gradE + np.sqrt(dt) * noise

    if step % 10 == 0:
        E = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
        energy_history.append(E)
        print(f"Step {step}, Energy = {E:.6f}")

    if step % 100 == 0:
        ax1.imshow(u.real, cmap='gray',extent=(0, 1, 0, 1))
        ax1.set_title(f"Iteration {step}")
        fig1.savefig("image_graddescent_anneal.png")
        
        ax2.plot(torch.arange(0,len(energy_history), 1), energy_history)
        #ax2.set_yscale('log')
        
        ax2.set_title("energie evolution")
        fig2.savefig("energy_graddescent_anneal.png")
        
        plt.pause(0.1)

