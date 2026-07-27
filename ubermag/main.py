import math
import time
import json
import csv
import re
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product
from typing import Tuple, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

import discretisedfield as df
import micromagneticmodel as mm


MU0 = 4.0 * math.pi * 1e-7


# ============================================================
# 1) Sample definition
# ============================================================

@dataclass
class SampleConfig:
    # Geometry
    Lx: float                  # sample width in x [m]
    Ly: float                  # sample width in y [m]
    t: float                   # thickness [m]
    dx: float                  # mesh size x [m]
    dy: float                  # mesh size y [m]

    # Material / known values
    Ms: float                  # saturation magnetization [A/m] / assumed
    easy_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Field during imaging (approx. 0)
    H: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Numerical
    random_seed: int = 0
    mz_threshold: float = 0.0


# ============================================================
# 2) Initial state
# ============================================================

def make_initial_state(sample: SampleConfig):
    rng = np.random.default_rng(sample.random_seed)

    def value_fun(pos):
        mz = 1.0 if rng.random() > 0.5 else -1.0
        mx = 0.03 * rng.standard_normal()
        my = 0.03 * rng.standard_normal()

        v = np.array([mx, my, mz], dtype=float)
        v /= np.linalg.norm(v)
        return tuple(v)

    return value_fun


# ============================================================
# 3) Small helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, max_len: int = 180) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\s\-.=]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len]


def build_run_name(
    sample: SampleConfig,
    A: float,
    Ku: float,
    prefix: str = "tacofebmgo",
) -> str:
    return (
        f"{prefix}_"
        f"t_{sample.t*1e9:.3f}nm_"
        f"A_{A*1e12:.3f}pJpm_"
        f"Ku_{Ku/1e6:.6f}MJpm3_"
        f"Ms_{sample.Ms/1e6:.3f}MApm"
    )
    #return f"{prefix}_0"


def compute_keff_from_ku(Ms: float, Ku: float) -> float:
    return Ku - 0.5 * MU0 * Ms**2


def compute_ku_from_keff(Ms: float, Keff: float) -> float:
    return Keff + 0.5 * MU0 * Ms**2


# ============================================================
# 4) Build Ubermag system
# ============================================================

def build_system(
    sample: SampleConfig,
    Ku: float,
    A: float,
    D: float = 0.0,
    system_name: str = "oommf_tacofebmgo",
):
    region = df.Region(p1=(0.0, 0.0, 0.0), p2=(sample.Lx, sample.Ly, sample.t))
    mesh = df.Mesh(region=region, cell=(sample.dx, sample.dy, sample.t))

    system = mm.System(name=system_name)

    system.m = df.Field(
        mesh=mesh,
        nvdim=3,
        value=make_initial_state(sample),
        norm=sample.Ms,
    )

    energy = mm.Exchange(A=A)
    energy += mm.UniaxialAnisotropy(K=Ku, u=sample.easy_axis)
    energy += mm.Demag()
    energy += mm.Zeeman(H=sample.H)

    if abs(D) > 0.0:
        energy += mm.DMI(D=D, crystalclass="Cnv_z")

    system.energy = energy
    return system


# ============================================================
# 5) Relaxation
# ============================================================

def relax_system_oommf(
    system,
    verbose: int = 2,
    max_steps: Optional[int] = None,
    stopping_mxHxm: Optional[float] = None,
):
    """
    verbose:
        0 -> quiet
        1/2 -> OOMMF usually prints more minimization info

    Note:
    Exact progress-bar style output depends on the installed OOMMF/OOMMFC
    version. In practice, verbose=2 is the usual way to get iteration output.
    """
    #driver_kwargs = {"verbose": verbose}
    driver_kwargs = {}

    if max_steps is not None:
        driver_kwargs["max_steps"] = max_steps

    if stopping_mxHxm is not None:
        driver_kwargs["stopping_mxHxm"] = stopping_mxHxm

    md = oc.MinDriver(**driver_kwargs)
    md.drive(system)
    return system


def relax_system_mumax(
    system,
    verbose: int = 2,
    DemagAccuracy: int = 6,
):
    md = mc.MinDriver(DemagAccuracy=DemagAccuracy)
    md.drive(system)
    return system

# ============================================================
# 6) Extract field data
# ============================================================

def system_to_m_arrays(system):
    xr = system.m.to_xarray()
    arr = xr.values

    if arr.ndim != 4:
        raise ValueError(f"Unexpected magnetization array shape: {arr.shape}")

    mx = np.mean(arr[..., 0], axis=2)
    my = np.mean(arr[..., 1], axis=2)
    mz = np.mean(arr[..., 2], axis=2)
    return mx, my, mz


def system_to_mz_2d(system):
    _, _, mz = system_to_m_arrays(system)
    return mz


# ============================================================
# 7) Plotting
# ============================================================

def build_plot_title(sample: SampleConfig, A: float, Ku: float, D: float = 0.0) -> str:
    Keff = compute_keff_from_ku(sample.Ms, Ku)
    return (
        "Ta/CoFeB/MgO relaxed domains\n"
        f"t={sample.t*1e9:.3f} nm, "
        f"A={A*1e12:.3f} pJ/m, "
        f"Ku={Ku/1e6:.6f} MJ/m³, "
        f"Keff={Keff/1e3:.3f} kJ/m³, "
        f"D={D*1e3:.3f} mJ/m²"
    )


def plot_domains(
    mz2d,
    sample: SampleConfig,
    title="Relaxed domain pattern",
    savepath: Optional[Path] = None,
    show: bool = True,
):
    extent = [0, sample.Lx * 1e6, 0, sample.Ly * 1e6]  # µm

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        mz2d.T,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label=r"$m_z$")
    ax.set_xlabel(r"$x\;[\mu m]$")
    ax.set_ylabel(r"$y\;[\mu m]$")
    ax.set_title(title)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 8) Save outputs
# ============================================================

def save_metadata_json(
    sample: SampleConfig,
    A: float,
    Ku: float,
    D: float,
    outpath: Path,
):
    meta = {
        "sample_config": asdict(sample),
        "material_parameters": {
            "A_J_per_m": A,
            "Ku_J_per_m3": Ku,
            "Keff_J_per_m3": compute_keff_from_ku(sample.Ms, Ku),
            "D_J_per_m2": D,
            "MU0_H_per_m": MU0,
        },
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_field_csv(
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    sample: SampleConfig,
    A: float,
    Ku: float,
    D: float,
    outpath: Path,
):
    """
    Saves one row per cell:
    i, j, x_m, y_m, mx, my, mz

    Metadata are written as commented header lines starting with '#'.
    """
    nx, ny = mz.shape

    with open(outpath, "w", newline="", encoding="utf-8") as f:
        # metadata as commented lines
        meta = {
            "sample_config": asdict(sample),
            "A_J_per_m": A,
            "Ku_J_per_m3": Ku,
            "Keff_J_per_m3": compute_keff_from_ku(sample.Ms, Ku),
            "D_J_per_m2": D,
            "nx": int(nx),
            "ny": int(ny),
        }
        for key, value in meta.items():
            f.write(f"# {key}: {value}\n")

        writer = csv.writer(f)
        writer.writerow(["i", "j", "x_m", "y_m", "mx", "my", "mz"])

        for i in range(nx):
            x = (i + 0.5) * sample.dx
            for j in range(ny):
                y = (j + 0.5) * sample.dy
                writer.writerow([i, j, x, y, mx[i, j], my[i, j], mz[i, j]])


def save_run_outputs(
    system,
    sample: SampleConfig,
    A: float,
    Ku: float,
    D: float,
    outdir: Path,
    show_plot: bool = False,
):
    outdir = ensure_dir(outdir)

    run_name = build_run_name(sample=sample, A=A, Ku=Ku)
    mx, my, mz = system_to_m_arrays(system)

    title = build_plot_title(sample=sample, A=A, Ku=Ku, D=D)
    png_name = f"{slugify(title)}.png"
    png_path = outdir / png_name

    csv_path = outdir / f"{run_name}_field.csv"
    json_path = outdir / f"{run_name}_metadata.json"

    save_field_csv(mx=mx, my=my, mz=mz, sample=sample, A=A, Ku=Ku, D=D, outpath=csv_path)
    save_metadata_json(sample=sample, A=A, Ku=Ku, D=D, outpath=json_path)
    plot_domains(mz, sample=sample, title=title, savepath=png_path, show=show_plot)

    return {
        "run_name": run_name,
        "png_path": png_path,
        "csv_path": csv_path,
        "json_path": json_path,
    }


# ============================================================
# 9) Utility: print derived quantities
# ============================================================

def print_parameter_summary(sample: SampleConfig, A: float, Ku: float, D: float):
    Keff = compute_keff_from_ku(sample.Ms, Ku)
    lex = math.sqrt(2 * A / (MU0 * sample.Ms**2)) 
    delta_eff = math.sqrt(A / max(Keff, 1e-30)) if Keff > 0 else float("inf")

    print("=" * 70)
    print("Ta/CoFeB/MgO simulation parameters")
    print("=" * 70)
    print(f"Ms              = {sample.Ms:.6e} A/m")
    print(f"A               = {A:.6e} J/m")
    print(f"Ku              = {Ku:.6e} J/m^3")
    print(f"Keff            = {Keff:.6e} J/m^3")
    print(f"D               = {D:.6e} J/m^2")
    print(f"thickness       = {sample.t*1e9:.6f} nm")
    print(f"cell            = ({sample.dx*1e9:.3f}, {sample.dy*1e9:.3f}, {sample.t*1e9:.6f}) nm")
    print(f"Lx, Ly          = ({sample.Lx*1e6:.3f}, {sample.Ly*1e6:.3f}) µm")
    print(f"exchange length = {lex*1e9:.6f} nm")
    if np.isfinite(delta_eff):
        print(f"sqrt(A/Keff)    = {delta_eff*1e9:.6f} nm")
    else:
        print("sqrt(A/Keff)    = inf (Keff <= 0)")
    print("=" * 70)


# ============================================================
# 10) Single run wrapper
# ============================================================

def run_single_simulation(
    sample: SampleConfig,
    A: float,
    Ku: float,
    D: float = 0.0,
    outdir: Path = Path("test_runs"),
    system_prefix: str = "tacofebmgo",
    verbose: int = 2,
    show_plot: bool = False,
    numerical_backend : str = "oommf"
):
    run_name = build_run_name(sample=sample, A=A, Ku=Ku, prefix=system_prefix)
    print_parameter_summary(sample=sample, A=A, Ku=Ku, D=D)
    print(f"Starting minimization: {run_name} with backend: {numerical_backend}")

    t0 = time.perf_counter()

    system = build_system(
        sample=sample,
        Ku=Ku,
        A=A,
        D=D,
        system_name=f"{numerical_backend}_{system_prefix}"
    )

    t1 = time.perf_counter()
    print(f"Build system: {t1 - t0:.2f} s")

    if numerical_backend == "oommf":
        system = relax_system_oommf(system, verbose=verbose)
    if numerical_backend == "mumax":
        system = relax_system_mumax(system, verbose=verbose)

    t2 = time.perf_counter()
    print(f"Relaxation:   {t2 - t1:.2f} s")

    saved = save_run_outputs(
        system=system,
        sample=sample,
        A=A,
        Ku=Ku,
        D=D,
        outdir=outdir,
        show_plot=show_plot,
    )

    t3 = time.perf_counter()
    print(f"Saving:       {t3 - t2:.2f} s")
    print(f"Total:        {t3 - t0:.2f} s")

    print(f"Saved CSV:  {saved['csv_path']}")
    print(f"Saved JSON: {saved['json_path']}")
    print(f"Saved PNG:  {saved['png_path']}")
    return system, saved


# ============================================================
# 11) Parameter grid search
# ============================================================

def run_parameter_grid_search(
    base_sample: SampleConfig,
    thickness_values: Iterable[float],
    A_values: Iterable[float],
    Ku_values: Iterable[float],
    D: float = 0.0,
    outdir: Path = Path("test_grid"),
    system_prefix: str = "grid",
    verbose: int = 1,
    show_plot: bool = False,
    numerical_backend: str = "oommf"
):
    """
    Loops over a grid in thickness, A, Ku.

    Parameters
    ----------
    thickness_values : iterable of float
        Thickness values in meters.
    A_values : iterable of float
        Exchange stiffness values in J/m.
    Ku_values : iterable of float
        Uniaxial anisotropy values in J/m^3.
    """
    outdir = ensure_dir(outdir)
    summary_rows = []

    all_cases = list(product(thickness_values, A_values, Ku_values))
    total = len(all_cases)

    print(f"Grid search: {total} runs")
    print("-" * 70)

    for idx, (t, A, Ku) in enumerate(all_cases, start=1):
        print(f"[{idx}/{total}] t={t*1e9:.3f} nm, A={A*1e12:.3f} pJ/m, Ku={Ku/1e6:.6f} MJ/m^3")

        sample = SampleConfig(
            Lx=base_sample.Lx,
            Ly=base_sample.Ly,
            t=t,
            dx=base_sample.dx,
            dy=base_sample.dy,
            Ms=base_sample.Ms,
            easy_axis=base_sample.easy_axis,
            H=base_sample.H,
            random_seed=base_sample.random_seed,
            mz_threshold=base_sample.mz_threshold,
        )

        try:
            system, saved = run_single_simulation(
                sample=sample,
                A=A,
                Ku=Ku,
                D=D,
                outdir=outdir,
                system_prefix=system_prefix,
                verbose=verbose,
                show_plot=show_plot,
                numerical_backend=numerical_backend
            )
            Keff = compute_keff_from_ku(sample.Ms, Ku)
            summary_rows.append({
                "status": "ok",
                "t_m": t,
                "t_nm": t * 1e9,
                "A_J_per_m": A,
                "A_pJ_per_m": A * 1e12,
                "Ku_J_per_m3": Ku,
                "Ku_MJ_per_m3": Ku / 1e6,
                "Keff_J_per_m3": Keff,
                "Keff_kJ_per_m3": Keff / 1e3,
                "Ms_A_per_m": sample.Ms,
                "csv_path": str(saved["csv_path"]),
                "json_path": str(saved["json_path"]),
                "png_path": str(saved["png_path"]),
            })
        except Exception as exc:
            summary_rows.append({
                "status": "failed",
                "t_m": t,
                "t_nm": t * 1e9,
                "A_J_per_m": A,
                "A_pJ_per_m": A * 1e12,
                "Ku_J_per_m3": Ku,
                "Ku_MJ_per_m3": Ku / 1e6,
                "Keff_J_per_m3": compute_keff_from_ku(base_sample.Ms, Ku),
                "Keff_kJ_per_m3": compute_keff_from_ku(base_sample.Ms, Ku) / 1e3,
                "Ms_A_per_m": base_sample.Ms,
                "csv_path": "",
                "json_path": "",
                "png_path": "",
                "error": str(exc),
            })
            print(f"Run failed: {exc}")

    summary_csv = outdir / "grid_summary.csv"
    fieldnames = [
        "status",
        "t_m",
        "t_nm",
        "A_J_per_m",
        "A_pJ_per_m",
        "Ku_J_per_m3",
        "Ku_MJ_per_m3",
        "Keff_J_per_m3",
        "Keff_kJ_per_m3",
        "Ms_A_per_m",
        "csv_path",
        "json_path",
        "png_path",
        "error",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    print("-" * 70)
    print(f"Grid search finished. Summary saved to: {summary_csv}")
    return summary_rows


# ============================================================
# 12) Main demo
# ============================================================


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend", choices=["mumax", "oommf"],
        help="Which backend is used, either mumax or oommf."
    )
    return parser.parse_args()



if __name__ == "__main__":

    base_sample = SampleConfig(
        Lx=10e-6,
        Ly=10e-6,
        t=1e-9,
        dx=5.0e-9,
        dy=5.0e-9,
        Ms=1.20e6,
        easy_axis=(0.0, 0.0, 1.0), # z- direction
        H=(0.0, 0.0, 0.0), # no external field
        random_seed=3, # some random seed for same re-iteration
    )
    sys_args = get_args()
    numerical_backend = sys_args.backend

    if numerical_backend == "oommf":
        import oommfc as oc
    elif numerical_backend == "mumax":
        import mumax3c as mc
    else:
        raise ValueError("Backend not found.")


    MAIN_PATH = Path(numerical_backend)

    SINGLE_RUN = True
    GRID_RUN = True

    if SINGLE_RUN:
        A = 18.0e-12
        Keff = 60.0e3
        Ku = compute_ku_from_keff(base_sample.Ms, Keff)
        print("Ku", Ku)
        D = 0.0
        
        run_single_simulation(
            sample=base_sample,
            A=A,
            Ku=Ku,
            D=D,
            outdir=MAIN_PATH / Path(f"{numerical_backend}_single_run"),
            system_prefix="tacofebmgo",
            verbose=2,
            show_plot=True,
            numerical_backend = numerical_backend
        )
        exit(0)


    if GRID_RUN:
        thickness_values = [1.0e-9, 1.2e-9, 1.4e-9] # [m]
        A_values = [10.0e-12, 12.0e-12, 18.0e-12] # [J/m]
        Keff_values = [1.0e4, 4.0e4, 8.0e4] # [J/m^3]

        Ku_values = [compute_ku_from_keff(base_sample.Ms, Keff) for Keff in Keff_values]

        run_parameter_grid_search(
            base_sample=base_sample,
            thickness_values=thickness_values,
            A_values=A_values,
            Ku_values=Ku_values,
            D=0.0, # no DMI interaction
            outdir=MAIN_PATH / Path(f"{numerical_backend}_grid_runs"),
            system_prefix="grid",
            verbose=1,
            show_plot=False,
            numerical_backend = numerical_backend
        )
