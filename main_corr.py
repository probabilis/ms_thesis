import numpy as np
import torch
import torch.fft
import matplotlib.pyplot as plt

# -- CUSTOM SET UP --
gridsize = 1
N = 300  # Number of Gridpoints
dt = 1 / 10  # Reduced initial time step
c0 = 9 / 32  # Normalization constant for double well

# CORRECTED PARAMETERS
epsilon = 1 / 22
gamma = 1 / 200  # Changed from 1/100 to match paper

th = 1.0  # Thickness parameter
Nmax = 40  # Max iterations for fixed point
tol = 10**(-6)  # Tighter tolerance for fixed point convergence

stop_crit = 10**(-8)  # Tighter stopping criterion
max_it = 1000

# -- GENERIC SET UP --
x = torch.arange(N, dtype=torch.float32) * gridsize / N
k = torch.cat([torch.arange(0, N//2), torch.arange(-N//2, 0)])  # Proper wave number vector for FFT
xi, eta = torch.meshgrid(k, k)  # 2D wave numbers
modk2 = xi**2 + eta**2
modk = torch.sqrt(modk2)

# IMPROVED SIGMA FUNCTION
def sigma(A):
    sig = torch.zeros_like(A)
    zero_freq = (torch.abs(A) < 1e-14)
    small = (torch.abs(A) >= 1e-14) & (torch.abs(A) < 1e-6)
    large = (torch.abs(A) >= 1e-6)
    
    sig[zero_freq] = 1
    sig[small] = 1 - np.pi * torch.abs(A[small])
    sig[large] = (1 - torch.exp(-2 * np.pi * torch.abs(A[large]))) / (2 * np.pi * torch.abs(A[large]))
    
    return sig

# FIXED POINT ITERATION FUNCTION
def fixpt(u0, L, dt, N, epsilon, gamma, Nmax, tol, c0):
    def NL(u, v, epsilon, gamma, c0):
        return 2 * gamma * c0 / epsilon * (u + v) * (1 - (torch.abs(u)**2 + torch.abs(v)**2) / 2)
    
    # Precompute FFT of initial condition
    fft_u0 = torch.fft.fft2(u0)
    
    # Constant term for iteration - CORRECTED
    denominator = torch.ones(N) + dt / 2 * L
    numerator = torch.ones(N) - dt / 2 * L
    
    # Avoid division by zero
    denominator[denominator == 0] = 1e-14
    CT = torch.real(torch.fft.ifft2(numerator / denominator * fft_u0))
    
    # Initialization
    j = 1
    error = 10
    u_int = u0
    u_int2 = torch.zeros_like(u0)
    convergence = 0
    
    # Fixed Point Iteration Loop
    while j <= Nmax and error > tol:
        nl_term = NL(u_int, u0, epsilon, gamma, c0)
        
        # Improved numerical stability
        fft_nl = torch.fft.fft2(nl_term)
        u_int2 = torch.real(torch.fft.ifft2(dt * fft_nl / denominator) + CT)
        
        error = torch.max(torch.abs(u_int2 - u_int))
        
        # Add relaxation for better convergence
        alpha = 0.7  # relaxation parameter
        u_int = alpha * u_int2 + (1 - alpha) * u_int
        j += 1
    
    if error <= tol:
        convergence = 1
    
    return j, u_int2, error, convergence

# ENERGY VALUE FUNCTION
def energy_value(gamma, epsilon, N, u, th, modk, modk2, c0):
    W = double_well(u, c0)
    ftu = torch.fft.fft2(u) / N**2  # Proper normalization
    
    S = sigma(th * modk)
    
    # CORRECTED energy calculation
    kinetic_energy = 0.5 * torch.sum((S + gamma * epsilon * modk2) * torch.abs(ftu)**2)
    potential_energy = (gamma / epsilon) * torch.sum(W) / N**2
    
    return potential_energy + kinetic_energy

# DOUBLE WELL FUNCTION
def double_well(u, c0):
    u2 = 1 - torch.abs(u)**2
    return c0 * (u2**2)

# IMPROVED INITIAL CONDITION
# Use smaller amplitude random initial condition
amplitude = 0.1
u0 = amplitude * (2 * torch.rand(N, N) - 1) + 1j * amplitude * (2 * torch.rand(N, N) - 1)

# Add small perturbation to break symmetry
x1, x2 = torch.meshgrid(x, x)
perturbation = 0.01 * torch.sin(2 * np.pi * x1) * torch.cos(2 * np.pi * x2)

u0 = u0 + perturbation

# CORRECTED LINEAR OPERATOR
L = gamma * epsilon * modk2 + sigma(th * modk)
L[0, 0] = sigma(torch.tensor(0.0))

# -- TIME STEPPING LOOP INITIALIZATION --
n_it = 1
time = 0
i = 1
u_int = u0
Du = 1
Denergy = 1000

# -- TIME VECTOR VALUES --
time_vector = torch.zeros(max_it)
time_vector[0] = 0

# -- ENERGY COMPUTATION - INITIALIZATION --
Energy = torch.zeros(max_it)
Energy[0] = energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0)

# -- TIME STEPPING LOOP --
while n_it < max_it and abs(Denergy) > stop_crit:
    # cranked nicolson comparison
    #u_np1 = crank_nicolson_step(u_n, L, dt, N, epsilon, gamma, c0, modk, modk2, th)
    k_fp, u, err, conv = fixpt(u_int, L, dt, N, epsilon, gamma, Nmax, tol, c0)
    
    if conv == 1:
        Energy[i] = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
        Du = torch.max(torch.abs(u - u_int))
        Denergy = Energy[i - 1] - Energy[i]
        print(Denergy)
        
        time_vector[i] = time_vector[i - 1] + dt
        
        # Visualization every 100 iterations to speed up
        if n_it % 100 == 0:
            plt.figure(1)
            plt.imshow(torch.real(u).numpy(), cmap='gray')
            plt.colorbar()
            plt.title(f'time = {time_vector[i]}')
            
            plt.figure(2)
            plt.semilogy(time_vector[:i], Energy[:i])
            plt.xlabel('Time')
            plt.ylabel('Energy')
            plt.title('Energy Evolution')
            plt.draw()
            plt.pause(1.0)
        
        # Update values
        u_int = u
        n_it += 1
        i += 1
        
        # Print progress
        if n_it % 1000 == 0:
            print(f'Iteration {n_it}, Time = {time_vector[i-1]}, Energy = {Energy[i-1]}, Energy Change = {Denergy}')
    
    elif conv == 0:
        dt *= 0.5  # Reduce time step more conservatively
        print(f'Reducing time step to {dt:.2e}')
        
        if dt < 1e-15:
            print('Time step became too small. Stopping simulation.')
            break

plt.clear()

# Final visualization
plt.figure(1)
plt.imshow(torch.real(u).numpy(), cmap='gray')
plt.colorbar()
plt.title(f'Final Pattern at time = {time_vector[i-1]}')

plt.figure(2)
plt.semilogy(time_vector[:i-1], Energy[:i-1])
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy Evolution')
plt.grid(True)

# Save figures
plt.figure(1)
plt.savefig(f'image_N={N}_max_it={max_it}_gamma={gamma}_eps={epsilon}_dt={dt}_th={th}.png')
plt.figure(2)
plt.savefig(f'energy_N={N}_max_it={max_it}_gamma={gamma}_eps={epsilon}_dt={dt}_th={th}.png')
