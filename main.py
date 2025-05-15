import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from pathlib import Path

from pattern_formation import fourier_multiplier, energy_value, fixpoint, dtype_complex, dtype_real, device


folder_path = r"out/"

# -- Custom Setup Parameters --
# -----------------------------


# data parameters
gridsize = 1 # gridsize
N = 1024 # nr. of grid-points

c0 = 9/32 # integral constant

th = 0.1 # sample parameters
epsilon = 1/20
gamma = 1/400

# simulation parameters
# dt = 1/3000
dt = 1/1000

max_it_fixpoint = 50
max_it = 200_000

NUMERICAL_THRESHOLD = 1e-12
tol = 1e-8

stop_crit = 1e-12


x = gridsize / N * torch.arange(N, dtype=dtype_real, device=device) # position array

# k = torch.fft.fftfreq(N, d=1 / N).to(device) * N  # FFT frequencies scaled
k = torch.cat([torch.arange(0, N // 2, dtype=dtype_real, device = device),torch.arange(-N // 2, 0, dtype=dtype_real, device = device)])

xi, eta = torch.meshgrid(k, k, indexing='ij')
modk2 = (xi ** 2 + eta ** 2).to(dtype_real)
modk = torch.sqrt(modk2)


# -----------------------------
# -- u0 Initialization --

u0 = (torch.randn(N, N, dtype=dtype_real, device=device) )# + 1j * torch.randn(N, N, dtype=dtype_real, device=device)).to(dtype_complex)

#image = Image.open('input_test.png').convert('L')
#np_array = np.array(image)
# Convert to PyTorch tensor and normalize to [0.0, 1.0]
# u0 = torch.from_numpy(np_array).float() / 255.0  # values now in [0.0, 1.0]
# print(u0.shape)

# u0 = (torch.tanh(10 * (torch.rand(N, N, dtype=dtype_real) - 0.5)) + 1j * torch.tanh(10 * (torch.rand(N, N, dtype=dtype_real) - 0.5)))

# -----------------------------


L = (2 * np.pi) ** 2 * gamma * epsilon * modk2 + fourier_multiplier(th * modk)
time_vector = [0.0]
energy_list = [energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)]

u_n = u0.clone()

energy_diff = 1000
ii = 0


plt.ion()
fig1, ax1 = plt.subplots(figsize = (14,12))
fig2, ax2 = plt.subplots(figsize = (10,10))

fp_iterations = []

try:
    while ii < max_it: # energy_diff > stop_crit and
        ii_fp, u_np1, err, conv = fixpoint(u_n, L, dt, N, epsilon, gamma, max_it_fixpoint, tol, c0)
        fp_iterations.append(ii_fp)
        print("----------------")
        print("Energy diff: ", energy_diff)
        if conv:
            curr_energy = energy_value(gamma, epsilon, N, u_np1, th, modk, modk2, c0)
            u_diff = torch.max(torch.abs(u_np1 - u_n)).item()
            energy_diff = energy_list[-1] - curr_energy
            energy_list.append(curr_energy)
            _time = time_vector[-1] + dt
            time_vector.append(_time)
            print("time: ", _time)

            LIVE_PLOT = False
            
            if LIVE_PLOT:
                ax1.clear()
                #ax.imshow(torch.abs(u).cpu().numpy(), cmap='gray', extent=(0, 1, 0, 1))
                ax1.imshow(u_n.cpu().numpy(), cmap='gray', extent=(0, 1, 0, 1))
                ax1.set_title(f"Time = {_time:.8f}")


                plt.pause(0.0001)

            u_n = u_np1
            ii += 1
            
        else:
            dt = dt / 4
            if dt < 1e-12:
                print("exit.")
                raise RuntimeError("Time step too small. Exiting.")

except KeyboardInterrupt:
    print("Exit.")

# -----------------------------

print("Final energy difference: ", energy_diff)
print(fp_iterations)


ax1.set_title(f"time = {_time:.8f}")
ax1.imshow(u_n.cpu().numpy(), cmap='gray', extent=(0, 1, 0, 1))

ax2.plot(time_vector, energy_list)
ax2.set_title("energy")
ax2.set_yscale('log')

plt.ioff()

fig1.savefig(folder_path + f"image_gamma={gamma}_eps={epsilon}_dt={dt}_ii={ii}.png")
fig2.savefig(folder_path + f"energy_gamma={gamma}_eps={epsilon}_dt={dt}_ii={ii}.png")
plt.show()
