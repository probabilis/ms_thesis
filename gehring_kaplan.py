import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

thickness_nm = np.array([1.3, 1.31, 1.32, 1.34, 1.36, 1.40], dtype=float)
domain_width_um = np.array([17.07508226, 5.69169425, 5.69169425,
                            3.41501639, 1.89723127, 1.31346779], dtype=float)

t = thickness_nm * 1e-9
w = domain_width_um * 1e-6

def domain_width_model(t, A, B):
    return A * t * np.exp(B / t)

# linearized initial guess
x = 1.0 / t
y = np.log(w / t)
m, c = np.polyfit(x, y, 1)
p0 = [np.exp(c), m]

popt, pcov = curve_fit(
    domain_width_model,
    t,
    w,
    p0=p0,
    maxfev=20000
)

A_fit, B_fit = popt
A_err, B_err = np.sqrt(np.diag(pcov))

print(f"A = {A_fit:.6e} ± {A_err:.3e}")
print(f"B = {B_fit:.6e} ± {B_err:.3e} m")
print(f"B = {B_fit*1e9:.6f} ± {B_err*1e9:.6f} nm")

t_plot_nm = np.linspace(thickness_nm.min(), thickness_nm.max(), 400)
t_plot = t_plot_nm * 1e-9
w_plot_um = domain_width_model(t_plot, A_fit, B_fit) * 1e6

plt.figure(figsize=(8, 6))
plt.scatter(thickness_nm, domain_width_um, label="experimental data")
plt.plot(t_plot_nm, w_plot_um, label="Kaplan-Gehring fit")
plt.xlabel("film thickness t [nm]")
plt.ylabel("domain width w [µm]")
#plt.yscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()