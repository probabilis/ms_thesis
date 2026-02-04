



def center_embed(u_small, factor=2, fill=0.0):
    """
    Embed u_small (N,N) into a bigger (factor*N, factor*N) tensor, centered.
    Returns: u_big, (top,left)
    """
    N = u_small.shape[0]
    Nb = factor * N
    top = (Nb - N) // 2
    left = (Nb - N) // 2

    u_big = torch.full((Nb, Nb), fill_value=fill, device=u_small.device, dtype=u_small.dtype)
    u_big[top:top+N, left:left+N] = u_small
    return u_big, top, left


def center_mask(N, factor=2, device=None, dtype=None):
    """
    Mask = 1 in centered ROI, 0 outside.
    Returns: M, (top,left)
    """
    Nb = factor * N
    top = (Nb - N) // 2
    left = (Nb - N) // 2
    M = torch.zeros((Nb, Nb), device=device, dtype=dtype)
    M[top:top+N, left:left+N] = 1.0
    return M, top, left



# --- padded FFT linear operator (open-boundary-ish) ---

def apply_linear_op_padded(U, Mk_pad):
    """
    Apply linear operator in Fourier space with 2x zero-padding to reduce wrap-around.
    U:      (Nb, Nb)
    Mk_pad: (2Nb, 2Nb) multiplier on padded grid
    returns (Nb, Nb)
    """
    Nb = U.shape[0]
    Np = 2 * Nb

    Up = torch.zeros((Np, Np), device=U.device, dtype=U.dtype)
    Up[:Nb, :Nb] = U

    FU = torch.fft.fft2(Up, norm="ortho")
    outp = torch.fft.ifft2(Mk_pad * FU, norm="ortho").real
    return outp[:Nb, :Nb]


def build_Mk_pad(gridsize_big, Nb, th, gamma, epsilon, dtype_real, device):
    """
    Build Mk on a padded grid of size 2Nb with physical size 2*gridsize_big.
    This keeps dx the same as in the big domain.
    """
    gridsize_pad = 2.0 * gridsize_big
    _, _, modk_pad, modk2_pad = define_spaces(gridsize_pad, 2 * Nb)
    sigma_pad = fourier_multiplier(th * modk_pad).to(dtype_real).to(device)
    Mk_pad = sigma_pad + gamma * epsilon * modk2_pad
    return Mk_pad



# --- energy consistent with padded-linear term (optional but useful) ---
def energy_value_with_data_embedded(
    gamma, epsilon,
    U, th,
    modk_pad, modk2_pad,  # on padded grid (2Nb)
    c0,
    lam, M, Uexp_big,
    N_roi):
    """
    Total energy (padded spectral linear term + DW + masked L2 data).
    Uses ortho FFT, matching your grad implementation.
    """
    Nb = U.shape[0]
    Np = 2 * Nb

    print(modk_pad) # [values]
    print(Uexp_big) # [0,0, ... 0,0]

    # DW term over big domain
    W = double_well_potential(U, c0)
    E_dw = (gamma / epsilon) * torch.sum(W) / (Nb**2)

    # spectral linear term on padded grid
    Up = torch.zeros((Np, Np), device=U.device, dtype=U.dtype)
    Up[:Nb, :Nb] = U
    FU = torch.fft.fft2(Up, norm="ortho")
    S_pad = fourier_multiplier(th * modk_pad).to(U.dtype).to(U.device)
    E_lin = 0.5 * torch.sum((S_pad + gamma * epsilon * modk2_pad) * (torch.abs(FU) ** 2))

    # masked data term, normalized by ROI size (so lam is stable vs embedding)
    E_data = 0.5 * lam * torch.sum(M * (U - Uexp_big) ** 2) / (N_roi ** 2)

    return (E_dw + E_lin + E_data).item()





def gradient_descent_nesterov_evaluation_centeredROI(
    u0, u_exp, _lambda,
    LIVE_PLOT, DATA_LOG, OUTPUT_PATH,
    gridsize, N, th, gamma, epsilon, tau, c0,
    num_iters, prox_newton_iters, tol_newton,
    STOP_BY_TOL=False, ENERGY_STOP_TOL=1e-6):


    if u_exp.shape[0] != u_exp.shape[1]:
        raise ValueError("u_exp must be square (N x N).")
    N_roi = u_exp.shape[0]

    if N != N_roi:
        raise ValueError(f"Parameter N={N} does not match experimental N={N_roi}.")

    Nb = 2 * N

    #u_exp = u_exp.to(device=device, dtype=dtype_real)
    #u0 = u0.to(device=device, dtype=dtype_real)

    gridsize_big = 2.0 * gridsize  # dx fixed

    print(f"Nb = {Nb}, gridsize big = {gridsize_big}")
    Uexp_big, top, left = center_embed(u_exp, factor=2, fill=0.0)
    M, _, _ = center_mask(N, factor=2, device=device, dtype=dtype_real)

    # embed u0 centered (or you can initialize random big and ignore u0 outside ROI)
    U0_big = initialize_u0_random(Nb, REAL=True) #, _, _ = center_embed(u0, factor=2, fill=0.0)

    Mk_pad = build_Mk_pad(
        gridsize_big=gridsize_big,
        Nb=Nb,
        th=th,
        gamma=gamma,
        epsilon=epsilon,
        dtype_real=dtype_real,
        device=device,
    )

    _, _, modk_pad, modk2_pad = define_spaces(2.0 * gridsize_big, 2 * Nb)

    def grad_g_with_data_big(Y):
        grad_lin = apply_linear_op_padded(Y, Mk_pad)
        grad_data = _lambda * M * (Y - Uexp_big)
        return grad_lin + grad_data

    U_prev = U0_big.clone()
    U_curr = U0_big.clone()
    t_prev = 1.0

    if LIVE_PLOT:
        plt.ion()
        fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
        fig2, ax2 = plt.subplots(1,1, figsize=(5,5))

    energies = [energy_value_with_data_embedded(
        gamma=gamma, epsilon=epsilon,
        U=U_curr, th=th,
        modk_pad=modk_pad, modk2_pad=modk2_pad,
        c0=c0,
        lam=_lambda, M=M, Uexp_big=Uexp_big,
        N_roi=N_roi
    )]

    # --- main loop ---
    for n in tqdm(range(1, num_iters + 1), desc="Nesterov GD (centered ROI)"):
        # 1) Nesterov extrapolation
        t_curr = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev) ** 0.5)
        beta = (t_prev - 1.0) / t_curr
        Y = U_curr + beta * (U_curr - U_prev)

        # 2) forward step
        ggrad = grad_g_with_data_big(Y)
        V = Y - tau * ggrad

        # 3) backward step (prox for double well)
        U_next = prox_h(V, tau, gamma, epsilon, c0, prox_newton_iters, tol_newton)

        # 4) update
        U_prev = U_curr
        U_curr = U_next
        t_prev = t_curr

        E = energy_value_with_data_embedded(
            gamma=gamma, epsilon=epsilon,
            U=U_curr, th=th,
            modk_pad=modk_pad, modk2_pad=modk2_pad,
            c0=c0,
            lam=_lambda, M=M, Uexp_big=Uexp_big,
            N_roi=N_roi
        )
        dE = energies[-1] - E
        energies.append(E)

        if STOP_BY_TOL and dE < ENERGY_STOP_TOL:
            break

        if LIVE_PLOT and plotting_schematic_eval is not None and (n % 10) == 0:
            u_roi = U_curr[top:top+N, left:left+N]
            plotting_schematic_eval(OUTPUT_PATH, ax1, fig1, ax2, fig2, u_roi, energies, N, num_iters, gamma, epsilon, _lambda, n)

    u_roi_fit = U_curr[top:top+N, left:left+N]

    if DATA_LOG and log_data is not None:
        log_data(OUTPUT_PATH, u_roi_fit, energies, N, num_iters, gamma, epsilon, _lambda)

    return U_curr, u_roi_fit, energies



if __name__ == "__main__":

    u_big, u, energies = gradient_descent_nesterov_evaluation_centeredROI(u0, u_exp, _lambda, LIVE_PLOT, DATA_LOG, OUTPUT_PATH,**asdict(exp_data_params),**asdict(ngd_sim_params), STOP_BY_TOL=False, ENERGY_STOP_TOL=ENERGY_STOP_TOL)
    ROI = True

    if ROI:
        axs[2].imshow(u_big.cpu().numpy(), cmap='gray',origin="lower", extent=(0,1,0,1))
        axs[2].set_box_aspect(1)
        axs[2].set_title(f"$\\gamma = {gamma}, \\lambda = {_lambda}$")