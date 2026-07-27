import numpy as np
import matplotlib.pyplot as plt

from utils.env_utils import plotting_style
from pathlib import Path
plotting_style()

def hysteresis_loop(H, Hc, Ms, width):
    """
    Simple phenomenological hysteresis model.

    Increasing branch:
        M crosses zero at +Hc

    Decreasing branch:
        M crosses zero at -Hc
    """

    M_increasing = Ms * np.tanh((H - Hc) / width)
    M_decreasing = Ms * np.tanh((H + Hc) / width)

    return M_increasing, M_decreasing


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

Hmax = 1.2
H = np.linspace(-Hmax, Hmax, 1000)

# Soft magnetic material
Hc_soft = 0.08       # coercive field
Ms_soft = 1.0        # saturation magnetization
width_soft = 0.08    # switching smoothness

# Hard magnetic material
Hc_hard = 0.70
Ms_hard = 1.0
width_hard = 0.06


# ------------------------------------------------------------
# Generate curves
# ------------------------------------------------------------

M_soft_inc, M_soft_dec = hysteresis_loop(H, Hc_soft, Ms_soft, width_soft)
M_hard_inc, M_hard_dec = hysteresis_loop(H, Hc_hard, Ms_hard, width_hard)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

# Soft magnetic loop
plt.plot(H, M_soft_inc, color="blue", label="Soft magnetic", linewidth = 2)
plt.plot(H, M_soft_dec, color="blue", linewidth = 2)

# Hard magnetic loop
plt.plot(H, M_hard_inc, color="red", label="Hard magnetic", linewidth = 2)
plt.plot(H, M_hard_dec, color="red", linewidth = 2)

# Axes
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.text(0.1, 1.1, "$M$", fontsize = 24)
plt.text(1.0, 0.05, "$H$", fontsize = 24)

# Coercive fields
plt.axvline(Hc_soft, color="blue", linestyle="--", alpha=0.6)
plt.axvline(-Hc_soft, color="blue", linestyle="--", alpha=0.6)

plt.axvline(Hc_hard, color="red", linestyle="--", alpha=0.6)
plt.axvline(-Hc_hard, color="red", linestyle="--", alpha=0.6)

# Saturation magnetization
plt.axhline(Ms_soft, color="gray", linestyle=":", alpha=0.7)
plt.axhline(-Ms_soft, color="gray", linestyle=":", alpha=0.7)

plt.text(Hmax * 0.75, Ms_soft + 0.03, r"$M_s$", fontsize=18)
plt.text(Hc_soft + 0.02, -0.25, r"$H_{c,\mathrm{soft}}$", color="blue", fontsize=18)
plt.text(Hc_hard + 0.02, -0.45, r"$H_{c,\mathrm{hard}}$", color="red", fontsize=18)

plt.xlabel(r"Magnetic field $H$")
plt.ylabel(r"Magnetization $M$")
#.title("Comparison of Soft and Hard Magnetic Hysteresis Loops")

plt.xlim(-Hmax, Hmax)
plt.ylim(-1.2 * Ms_soft, 1.2 * Ms_soft)

plt.axis("off")

#plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
from utils.env_utils import PATHS
plt.savefig(PATHS.PATH_THESIS / "hystersis.png", dpi = 300)
plt.show()