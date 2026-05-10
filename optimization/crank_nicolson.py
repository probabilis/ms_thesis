from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from utils.env_utils import plotting_schematic, log_data
from utils.pattern_formation import define_spaces, fourier_multiplier, double_well_potential


# ---------------------------------------------------------------

def N_eps(U_np1, U_n, epsilon, gamma, c0):
    return 2 * gamma * c0 / epsilon * (U_np1 + U_n) * (1 - (torch.abs(U_np1) ** 2 + torch.abs(U_n) ** 2) / 2)

# ------------------------------------------------------------------

def energy_value(gamma, epsilon, N, u, M_k, c0):
    """
    Energy functional with spectral variant
    E = LaPlace + DW + FM 
    """
    ftu = torch.fft.fft2(u, norm = 'ortho') #/ N**2 

    E_LPFM = 0.5 * torch.sum( M_k * torch.abs(ftu)**2 )
    
    W = double_well_potential(u, c0)
    E_DW = (gamma / epsilon) * torch.sum(W) / N**2 
     
    return (E_LPFM + E_DW).item()

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

    energies_fixpoint = []

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

        curr_energy = energy_value(gamma, epsilon, N, U_n, L_eps, c0)
        energies_fixpoint.append(curr_energy)

    if error < tol:
        conv = True

    return ii, U_n, error, conv, energies_fixpoint


# ---------------------------------------------------------------

def adapted_crank_nicolson(u0, LIVE_PLOT, DATA_LOG, FOLDER_PATH, gridsize, N, th, epsilon, gamma, dt, max_it_fixpoint, max_it, tol, stop_limit, c0, STOP_BY_TOL = True):
    
    """
    Adapted Crank-Nicolson Schematic as Mr. Condette implemented it in his thesis
    time splitting + fixpoint iteration schematic
    
    """
    
    x, k, modk, modk2 = define_spaces(gridsize, N)

    
    L = (2*torch.pi)**2 * gamma * epsilon * modk2 + fourier_multiplier(th * modk)
    #L[0, 0] = fourier_multiplier(torch.tensor(0.0))

    time_vector = [0.0]
    energies = [energy_value(gamma, epsilon, N, u0, L, c0)]

    u_n = u0

    energy_diff = 1000
    ii = 0

    if LIVE_PLOT or DATA_LOG:
        fig1, ax1 = plt.subplots(figsize = (14,12))
        fig2, ax2 = plt.subplots(figsize = (10,10))
        plt.ion()

    fp_iterations = []
    ii_updated = 0

    pbar = tqdm(total=max_it, desc = "Crank Nicolson")
    
    try:
        while ii_updated < max_it:

            if STOP_BY_TOL and energy_diff <= stop_limit: # only when STOP_BY_TOL is True, max_iterations will be cut
                print("Converged: ", energy_diff)
                break

            ii_fp, u_np1, err, conv, energies_fixpoint = fixpoint(u_n, L, dt, N, epsilon, gamma, max_it_fixpoint, tol, c0)
            fp_iterations.append(ii_fp)
            #print("ii fixpoint", ii_fp)
            
            if conv:
                curr_energy = energy_value(gamma, epsilon, N, u_np1, L, c0)
                u_diff = torch.max(torch.abs(u_np1 - u_n)).item()

                energy_diff = energies[-1] - curr_energy
                
                energies.extend(energies_fixpoint)
                _time = time_vector[-1] + dt
                time_vector.append(_time)

                u_n = u_np1
                ii += 1
                
                if LIVE_PLOT and ii % 100 == 0:
                    plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u_n, energies, N, max_it, gamma, epsilon, ii)
                    plt.pause(1)
                
            else:
                dt = dt / 4
                print("reduced dt to", dt)
                if dt < 1e-12:
                    print("exit.")
                    raise RuntimeError("Time step too small. Exiting.")
            
            ii_updated = sum(fp_iterations)
            pbar.update()
            #time.sleep(1)

    except KeyboardInterrupt:
        print("Exit.")

    pbar.close()
    print("ii total", ii_updated)
    plt.ioff()

    if DATA_LOG:
        log_data(FOLDER_PATH, u_n, energies, N, max_it, gamma, epsilon)
        plotting_schematic(FOLDER_PATH, ax1, fig1, ax2, fig2, u_n, energies, N, max_it, gamma, epsilon, ii)

    return u_n, energies

    