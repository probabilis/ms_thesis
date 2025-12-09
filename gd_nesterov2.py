import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt

from pattern_formation import dtype_real, fourier_multiplier, prox_h, laplacian
from pattern_formation import energy_value_fd_mix, fourier_multiplier_simple, fourier_multiplier_dipolar

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------
# params:

N = 64
L = 1.0
h = L / N

th = 1.0
c0 = 9/32

eps = 0.02
gamma = 0.0001 #* 100

steps = 500_000

# Fourier wave numbers
kx = torch.fft.fftfreq(N, d=h).to(device) * 2 *  torch.pi
ky = torch.fft.fftfreq(N, d=h).to(device) * 2 * torch.pi
KX, KY = torch.meshgrid(kx, ky, indexing='ij')
K = torch.sqrt(KX**2 + KY**2)



def fourier_multiplier_double(k_mag):
    """
    Fourier Multiplier Dipolar
    """

    sig = torch.zeros_like(k_mag)
    zero_freq = (torch.abs(k_mag) < 1e-14)
    small = (torch.abs(k_mag) >= 1e-14) & (torch.abs(k_mag) < 1e-6)
    large = (torch.abs(k_mag) >= 1e-6)
    
    sig[zero_freq] = 1
    sig[small] = 1 - torch.pi * torch.abs(k_mag[small])
    sig[large] = (1 - torch.exp(-2 * torch.pi * torch.abs(k_mag[large]) ** 2)) / (2 * torch.pi * torch.abs(k_mag[large]))
    
    return sig


x = torch.arange(-10,10, 0.1)

y_double = fourier_multiplier_double(x)
y_fourier = fourier_multiplier(x)

y_ls = [y_double, y_fourier]

labels = ["FM double", "FM standard"]

for ii, y in enumerate(y_ls):
    plt.plot(x, torch.abs(y), label = f"{labels[ii]}")

plt.legend()
plt.ylim(-0.1,+1)
plt.show()

#exit(0)
sigma_hat = fourier_multiplier(th * K).to(dtype_real).to(device)
#sigma_hat = fourier_multiplier_double(th * K).to(dtype_real).to(device)

def grad_energy(u):
    """
    form of:
    grad(E) = gamma * eps * laplacian - 1/eps * DW - iFTT(u)
    """

    # local terms
    Wp = (u**3 - u) * c0 * 4 # samer as PF
    lap = laplacian(u, h) # PF
    grad_local = gamma * (- eps * lap + Wp / eps) # PF
    # nonlocal term
    u_hat = fft.fft2(u, norm='ortho')
    _nonlocal = fft.ifft2(sigma_hat * u_hat, norm='ortho').real

    return grad_local + _nonlocal


amplitude = 0.1
u = amplitude * (2 * torch.rand(N, N, dtype=dtype_real, device=device) - 1)
v = torch.zeros_like(u)

plt.ion()

t_prev = 1.0
u_prev = u.clone()
u_curr = u.clone()
tau = 5e-3

energies = [energy_value_fd_mix(u, sigma_hat, N, gamma, eps, c0)]

fig, axs = plt.subplots(2,2)

for n in range(steps):
    t_curr = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev)**0.5)
    beta = (t_prev - 1.0) / t_curr
    y = u_curr + beta * (u_curr - u_prev)


    g = grad_energy(y)

    v = y - tau * g

    u_next = prox_h(v, tau, gamma, eps, c0, 20, 1e-8)

    u_prev = u_curr
    u_curr = u_next
    t_prev = t_curr

    E = energy_value_fd_mix(u_curr, sigma_hat, N, gamma, eps, c0)
    
    energy_diff = energies[-1] - E
    energies.append(E)

    print("dE", energy_diff)
    #print(energies)
    if n % 1000 == 0:
        print(f"step {n}")

        u_hat = fft.fft2(u_curr, norm='ortho')
        xf = torch.fft.fftfreq(N)
        yf = torch.diagonal(u_hat)
        yf = torch.abs(yf)

        axs[0, 0].imshow(u_curr)
        axs[0, 1].imshow(torch.abs(u_hat))
        axs[1, 1].plot(xf[0:N//2], 2.0/N * torch.abs(yf[0:N//2]))
        axs[1, 0].plot(np.arange(0, len(energies), 1), energies)
        plt.pause(1)


plt.ioff()