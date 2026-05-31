import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

t_nm = np.array([1.30, 1.33, 1.36, 1.40])  # nm
Dp_um = np.array([5.69169425, 3.41501639, 1.89723127, 1.31346779])  # µm

# Convert to SI units
t = t_nm * 1e-9      # m
Dp = Dp_um * 1e-6    # m

# ------------------------------------------------------------
# Model: Dp(t) = C * exp(B / t)
# ------------------------------------------------------------

def domain_period_model(t, C, B):
    return C * np.exp(B / t)

# Initial guesses
C0 = 1e-9       # m, prefactor scale
B0 = 1e-9       # m, exponential length scale

popt, pcov = curve_fit(
    domain_period_model,
    t,
    Dp,
    p0=[C0, B0],
    maxfev=10000
)

C_fit, B_fit = popt
C_err, B_err = np.sqrt(np.diag(pcov))

print("Fit results:")
print(f"C = {C_fit:.6e} ± {C_err:.6e} m")
print(f"B = {B_fit:.6e} ± {B_err:.6e} m")

print()
print("Equivalent:")
print(f"C = {C_fit * 1e9:.6f} nm")
print(f"B = {B_fit * 1e9:.6f} nm")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

t_fit_nm = np.linspace(t_nm.min(), t_nm.max(), 300)
t_fit = t_fit_nm * 1e-9
Dp_fit = domain_period_model(t_fit, C_fit, B_fit)

if __name__ == "__main__":
    

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t_nm, Dp_um, label="data", zorder=5)
    ax.plot(t_fit_nm, Dp_fit * 1e6, label="fit")

    ax.set_xlabel(r"$t~[\mathrm{nm}]$")
    ax.set_ylabel(r"$D_\mathrm{P}~[\mu\mathrm{m}]$")
    ax.set_title(r"Fit of $D_\mathrm{P}(t) = C \exp(B/t)$")
    ax.legend()

    plt.tight_layout()
    plt.show()