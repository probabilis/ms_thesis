from pathlib import Path
from postprocess import load_mumax_ovf

import matplotlib.pyplot as plt

files = ["t_1.000nm_A_24.00pJpm_Keff_40.0kJpm3_Ms_1.200MApm",
         "t_1.300nm_A_44.00pJpm_Keff_40.0kJpm3_Ms_1.200MApm",
         "t_1.300nm_A_14.00pJpm_Keff_80.0kJpm3_Ms_1.200MApm",
         "t_1.400nm_A_14.00pJpm_Keff_40.0kJpm3_Ms_1.200MApm"]



titles = ["$\\delta = 1.0 \\mathrm{nm} \\quad A_s = 24 \\mathrm{pJ/m} \\quad K_\\mathrm{eff} = 40 \\mathrm{ kJ/m^3}$",
         "$\\delta = 1.3 \\mathrm{nm} \\quad A_s = 44 \\mathrm{pJ/m} \\quad K_\\mathrm{eff} = 40 \\mathrm{kJ/m^3}$",
         "$\\delta = 1.3 \\mathrm{nm} \\quad A_s = 14 \\mathrm{pJ/m} \\quad K_\\mathrm{eff} = 80 \\mathrm{kJ/m^3}$",
         "$\\delta = 1.4 \\mathrm{nm} \\quad A_s = 14 \\mathrm{pJ/m} \\quad K_\\mathrm{eff} = 40 \\mathrm{kJ/m^3}$"]


FOLDER_PATH = Path("data")

FOLDER_PATH = FOLDER_PATH / "mumax_test_final"


fig, axs = plt.subplots(2,2, figsize = (12,12) )

axs = axs.ravel()

for ii, file in enumerate(files):

    ovf_path = FOLDER_PATH / file / "run.out" /f"mz_final.ovf"

    x, y, mx, my, mz, meta = load_mumax_ovf(ovf_path)

    extent = [
        x.min() * 1e6,
        x.max() * 1e6,
        y.min() * 1e6,
        y.max() * 1e6,
    ]

    im = axs[ii].imshow(
        mz.T,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest")

    axs[ii].set_title(titles[ii], fontsize = 16)

    axs[ii].set_xlabel("$x \\quad \\mu\\mathrm{m}$", fontsize = 14)
    axs[ii].set_ylabel("$y \\quad \\mu\\mathrm{m}$", fontsize = 14)


plt.savefig(FOLDER_PATH / "final.png", dpi = 300)
plt.show()


