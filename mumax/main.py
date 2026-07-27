#!/usr/bin/env python3

import csv
import math
import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable


MU0 = 4.0 * math.pi * 1e-7


@dataclass
class MumaxDebugConfig:
    # Small proofing geometry
    Lx: float = 10e-6
    Ly: float = 10e-6
    dx: float = 20e-9
    dy: float = 20e-9

    # No PBC for OOMMF comparison
    pbc_x: int = 0
    pbc_y: int = 0

    # Material
    Ms: float = 1.20e6

    # Numerical
    random_seed: int = 3
    output_format: str = "OVF2_BINARY"

    # Minimize is preferred here: simple quasi-static energy minimization
    minimizer_stop: float = 1e-4
    minimizer_samples: int = 10
    minimize_wallclock_s: int = 10 * 60
    python_timeout_s: int = 15 * 60

    outdir: Path = Path("mumax_debug_oommf_reference")


def compute_ku_from_keff(Ms: float, Keff: float) -> float:
    return Keff + 0.5 * MU0 * Ms**2


def quality_factor(Ms: float, Ku: float) -> float:
    return 2.0 * Ku / (MU0 * Ms**2)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_name(t: float, A: float, Keff: float, Ms: float) -> str:
    return (
        f"t_{t*1e9:.3f}nm_"
        f"A_{A*1e12:.2f}pJpm_"
        f"Keff_{Keff/1e3:.1f}kJpm3_"
        f"Ms_{Ms/1e6:.3f}MApm"
    )


def make_mx3(
    cfg: MumaxDebugConfig,
    *,
    t: float,
    A: float,
    Keff: float,
    Ku: float,
    Ms: float,
) -> str:
    nx = int(round(cfg.Lx / cfg.dx))
    ny = int(round(cfg.Ly / cfg.dy))

    return f"""
// ============================================================
// Minimal MuMax3 proofing run
// Intended to reproduce thin-film Ta/CoFeB/MgO formation
// ============================================================

OutputFormat = {cfg.output_format}

// ---------- Mesh ----------
SetGridSize({nx}, {ny}, 1)
SetCellSize({cfg.dx:.16g}, {cfg.dy:.16g}, {t:.16g})
SetPBC({cfg.pbc_x}, {cfg.pbc_y}, 0)

// ---------- Material ----------
Msat = {Ms:.16g}
Aex  = {A:.16g}
Ku1  = {Ku:.16g}
anisU = vector(0, 0, 1)

// ---------- External field ----------
B_ext = vector(0, 0, 0)

// ---------- Solver ----------
MinimizeWallClockTime = {cfg.minimize_wallclock_s}
MinimizerStop = {cfg.minimizer_stop:.6g}
MinimizerSamples = {cfg.minimizer_samples}

// ---------- Initial condition ----------
// This is closest to your successful OOMMF random-start runs.
m = randomMagSeed({cfg.random_seed})

// Save initial state to verify that plotting works.
SaveAs(m, "m_initial")
SaveAs(m.Comp(2), "mz_initial")

// ---------- Table output ----------
TableAdd(E_exch)
TableAdd(E_anis)
TableAdd(E_demag)
TableAdd(E_Zeeman)
TableAdd(E_total)
TableAdd(maxTorque)
TableAdd(m.Comp(2))

print("t_m=", {t:.16g})
print("A_J_per_m=", {A:.16g})
print("Keff_J_per_m3=", {Keff:.16g})
print("Ku_J_per_m3=", {Ku:.16g})
print("Ms_A_per_m=", {Ms:.16g})
print("Nx=", {nx})
print("Ny=", {ny})

// ---------- Minimize ----------
ok := minimize()
print("minimize_converged=", ok)
TableSave()

// ---------- Save final fields ----------
SaveAs(m, "m_final")
SaveAs(m.Comp(2), "mz_final")
TableSave()
""".strip() + "\n"


def run_case(cfg: MumaxDebugConfig, *, t: float, A: float, Keff: float) -> dict:
    Ms = cfg.Ms
    Ku = compute_ku_from_keff(Ms, Keff)
    Q = quality_factor(Ms, Ku)

    name = run_name(t, A, Keff, Ms)
    run_dir = ensure_dir(cfg.outdir / name)

    row = {
        "run_name": name,
        "status": "pending",
        "error": "",
        "t_m": t,
        "t_nm": t * 1e9,
        "A_J_per_m": A,
        "A_pJ_per_m": A * 1e12,
        "Keff_J_per_m3": Keff,
        "Keff_kJ_per_m3": Keff / 1e3,
        "Ku_J_per_m3": Ku,
        "Ku_MJ_per_m3": Ku / 1e6,
        "Ms_A_per_m": Ms,
        "Ms_MA_per_m": Ms / 1e6,
        "Q": Q,
        "run_dir": str(run_dir),
        "mx3_path": str(run_dir / "run.mx3"),
        "ovf_path": str(run_dir / "run.out" / "m_final.ovf"),
        "mz_ovf_path": str(run_dir / "run.out" / "mz_final.ovf"),
        "table_path": str(run_dir / "run.out" / "table.txt"),
        "log_path": str(run_dir / "mumax_stdout_stderr.log"),
    }

    mx3_text = make_mx3(cfg, t=t, A=A, Keff=Keff, Ku=Ku, Ms=Ms)
    (run_dir / "run.mx3").write_text(mx3_text)

    try:
        result = subprocess.run(
            ["mumax3", "run.mx3"],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=cfg.python_timeout_s,
            check=False,
        )

        (run_dir / "mumax_stdout_stderr.log").write_text(result.stdout or "")

        if result.returncode == 0:
            row["status"] = "ok"
        else:
            row["status"] = "failed"
            row["error"] = f"mumax3 failed with return code {result.returncode}"

    except subprocess.TimeoutExpired as exc:
        row["status"] = "timeout"
        row["error"] = f"Python timeout after {cfg.python_timeout_s} s"
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        (run_dir / "mumax_stdout_stderr.log").write_text(
            stdout + "\n\n--- PYTHON TIMEOUT ---\n" + row["error"]
        )

    return row


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(
    cfg: MumaxDebugConfig,
    thickness_values: Iterable[float],
    A_values: Iterable[float],
    Keff_values: Iterable[float],
):
    ensure_dir(cfg.outdir)
    summary_path = cfg.outdir / "grid_summary.csv"

    cases = list(product(thickness_values, A_values, Keff_values))
    rows = []

    print(f"Total cases: {len(cases)}")
    print(f"Output: {cfg.outdir}")
    print(f"Geometry: {cfg.Lx*1e6:.1f} x {cfg.Ly*1e6:.1f} µm²")
    print(f"Cell: {cfg.dx*1e9:.1f} x {cfg.dy*1e9:.1f} nm²")
    print(f"PBC: ({cfg.pbc_x}, {cfg.pbc_y}, 0)")
    print("-" * 80)

    for idx, (t, A, Keff) in enumerate(cases, start=1):
        Ku = compute_ku_from_keff(cfg.Ms, Keff)
        Q = quality_factor(cfg.Ms, Ku)

        print(
            f"[{idx}/{len(cases)}] "
            f"t={t*1e9:.3f} nm, "
            f"A={A*1e12:.2f} pJ/m, "
            f"Keff={Keff/1e3:.1f} kJ/m³, "
            f"Ku={Ku/1e6:.4f} MJ/m³, "
            f"Q={Q:.4f}",
            flush=True,
        )

        row = run_case(cfg, t=t, A=A, Keff=Keff)
        rows.append(row)

        print(f"    -> {row['status']} {row['error']}", flush=True)
        write_summary(summary_path, rows)

    print("-" * 80)
    print(f"Summary written to: {summary_path}")
    return rows


if __name__ == "__main__":
    cfg = MumaxDebugConfig(
        Lx=10e-6,
        Ly=10e-6,
        dx=20e-9,
        dy=20e-9,
        pbc_x=0,
        pbc_y=0,
        Ms=1.20e6,
        random_seed=3,
        output_format="OVF2_BINARY",
        minimizer_stop=1e-4,
        minimizer_samples=10,
        minimize_wallclock_s=10 * 60,
        python_timeout_s=15 * 60,
        outdir=Path("mumax_test"),
    )

    # Keep this tiny. We only want to prove MuMax reproduces OOMMF-style behavior.
    thickness_values = [
        1.30e-9,
        1.40e-9,
    ]

    A_values = [
        12e-12,
        14e-12,
    ]

    Keff_values = [
        20e3,
        40e3,
        80e3,
    ]

    run_sweep(
        cfg=cfg,
        thickness_values=thickness_values,
        A_values=A_values,
        Keff_values=Keff_values,
    )