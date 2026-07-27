#!/usr/bin/env python3
"""
Postprocess mumax3 sweep outputs:
- reads grid_summary.csv
- finds each run's mz_final.ovf or m_final.ovf
- supports OVF2_BINARY and OVF2_TEXT
- extracts m_z
- saves one PNG per run
- optionally saves overview grid

Run:
    python3 plot_mumax_mz_binary.py --base-dir mumax_CoFeB_field_demag_runs --overview
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ============================================================
# 1) OVF header parsing
# ============================================================


plt.style.use('classic')

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif'
    })


def _parse_header_text(header_text: str) -> Dict[str, str]:
    header: Dict[str, str] = {}

    for line in header_text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue

        clean = s[1:].strip()

        if ":" in clean:
            key, value = clean.split(":", 1)
            header[key.strip().lower()] = value.strip()

    return header


def _get_int(header: Dict[str, str], key: str) -> int:
    if key not in header:
        raise KeyError(f"Missing OVF header key: {key}")
    return int(float(header[key]))


def _get_float(header: Dict[str, str], key: str, default: float = math.nan) -> float:
    if key not in header:
        return default
    return float(header[key])


def _get_valuedim(header: Dict[str, str]) -> int:
    if "valuedim" in header:
        return int(float(header["valuedim"]))
    return 3


# ============================================================
# 2) OVF TEXT reader
# ============================================================

def _load_ovf_text(path: Path, raw: bytes, data_marker: bytes):
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    header_text = []
    data_start = None

    marker_lower = data_marker.lower().decode("utf-8", errors="replace")

    for idx, line in enumerate(lines):
        header_text.append(line)
        if marker_lower in line.lower():
            data_start = idx + 1
            break

    if data_start is None:
        raise ValueError("Could not locate OVF text data block.")

    header = _parse_header_text("\n".join(header_text))

    nx = _get_int(header, "xnodes")
    ny = _get_int(header, "ynodes")
    nz = _get_int(header, "znodes")
    valuedim = _get_valuedim(header)

    values = []

    for line in lines[data_start:]:
        s = line.strip()

        if not s:
            continue

        if s.startswith("#"):
            if "end: data" in s.lower():
                break
            continue

        values.extend(float(v) for v in s.split())

    arr = np.asarray(values, dtype=float)

    expected = nx * ny * nz * valuedim
    if arr.size != expected:
        raise ValueError(
            f"Unexpected text OVF data size: got {arr.size}, expected {expected}."
        )

    return _reshape_ovf_data(arr, header, nx, ny, nz, valuedim)


# ============================================================
# 3) OVF BINARY reader
# ============================================================

def _find_binary_data_start(raw: bytes) -> Tuple[bytes, int, str]:
    """
    Finds binary data marker and returns:
        marker, data_start_byte_index, precision_tag
    """
    markers = [
        (b"# Begin: Data Binary 4", "binary4"),
        (b"# Begin: Data Binary 8", "binary8"),
    ]

    raw_lower = raw.lower()

    for marker, tag in markers:
        idx = raw_lower.find(marker.lower())
        if idx >= 0:
            line_end = raw.find(b"\n", idx)
            if line_end < 0:
                raise ValueError("Malformed OVF file: binary data marker has no newline.")
            return marker, line_end + 1, tag

    raise ValueError("Could not find '# Begin: Data Binary 4/8'.")


def _choose_binary_dtype(data_block: bytes, precision_tag: str):
    """
    OOMMF/MuMax binary OVF usually starts with a check number.

    Binary 4 check: 1234567.0
    Binary 8 check: 123456789012345.0

    Endianness is detected automatically.
    """
    if precision_tag == "binary4":
        candidates = [
            (">f4", 1234567.0),
            ("<f4", 1234567.0),
        ]
        width = 4
    elif precision_tag == "binary8":
        candidates = [
            (">f8", 123456789012345.0),
            ("<f8", 123456789012345.0),
        ]
        width = 8
    else:
        raise ValueError(f"Unknown binary precision tag: {precision_tag}")

    for dtype, check_value in candidates:
        check = np.frombuffer(data_block[:width], dtype=np.dtype(dtype), count=1)[0]
        if np.isfinite(check) and np.isclose(check, check_value, rtol=1e-5, atol=1e-5):
            return np.dtype(dtype), width

    # Fallback: try little endian without check, but warn through exception text.
    raise ValueError(
        "Could not identify OVF binary endianness/check value. "
        "This may be a nonstandard binary OVF file or a corrupted output."
    )


def _load_ovf_binary(path: Path, raw: bytes):
    marker, data_start, precision_tag = _find_binary_data_start(raw)

    header_text = raw[:data_start].decode("utf-8", errors="replace")
    header = _parse_header_text(header_text)

    nx = _get_int(header, "xnodes")
    ny = _get_int(header, "ynodes")
    nz = _get_int(header, "znodes")
    valuedim = _get_valuedim(header)

    n_values = nx * ny * nz * valuedim

    data_block = raw[data_start:]
    dtype, check_width = _choose_binary_dtype(data_block, precision_tag)

    bytes_per_value = dtype.itemsize
    n_bytes = check_width + n_values * bytes_per_value

    if len(data_block) < n_bytes:
        raise ValueError(
            f"Binary OVF data block too short: got {len(data_block)} bytes, "
            f"expected at least {n_bytes} bytes."
        )

    values = np.frombuffer(
        data_block[check_width:n_bytes],
        dtype=dtype,
        count=n_values,
    ).astype(float)

    return _reshape_ovf_data(values, header, nx, ny, nz, valuedim)


# ============================================================
# 4) Common reshape logic
# ============================================================

def _reshape_ovf_data(
    arr: np.ndarray,
    header: Dict[str, str],
    nx: int,
    ny: int,
    nz: int,
    valuedim: int,
):
    """
    MuMax writes data in x-fastest order.

    Raw shape:
        (nz, ny, nx, valuedim)

    Returned arrays:
        mx, my, mz with shape (nx, ny)
    """
    vec = arr.reshape((nz, ny, nx, valuedim))
    vec = np.transpose(vec, (2, 1, 0, 3))  # (nx, ny, nz, valuedim)

    if valuedim == 1:
        mx = None
        my = None
        mz = vec[..., 0].mean(axis=2)
    elif valuedim >= 3:
        mx = vec[..., 0].mean(axis=2)
        my = vec[..., 1].mean(axis=2)
        mz = vec[..., 2].mean(axis=2)
    else:
        raise ValueError(f"Unsupported OVF valuedim={valuedim}")

    xmin = _get_float(header, "xmin", 0.0)
    ymin = _get_float(header, "ymin", 0.0)

    xstepsize = _get_float(header, "xstepsize", math.nan)
    ystepsize = _get_float(header, "ystepsize", math.nan)

    # Fallback if step size is absent.
    if not np.isfinite(xstepsize):
        xmax = _get_float(header, "xmax", float(nx))
        xstepsize = (xmax - xmin) / nx

    if not np.isfinite(ystepsize):
        ymax = _get_float(header, "ymax", float(ny))
        ystepsize = (ymax - ymin) / ny

    x = xmin + (np.arange(nx) + 0.5) * xstepsize
    y = ymin + (np.arange(ny) + 0.5) * ystepsize

    return x, y, mx, my, mz, header


# ============================================================
# 5) Public OVF loader
# ============================================================

def load_mumax_ovf(path: Path):
    """
    Loads MuMax3 OVF2_TEXT or OVF2_BINARY file.

    Works for:
        m_final.ovf      vector field, valuedim=3
        mz_final.ovf     scalar field, valuedim=1

    Returns:
        x, y, mx, my, mz, meta
    """
    path = Path(path)
    raw = path.read_bytes()
    raw_lower = raw.lower()

    if b"# begin: data text" in raw_lower:
        return _load_ovf_text(path, raw, b"# Begin: Data Text")

    if b"# begin: data binary 4" in raw_lower or b"# begin: data binary 8" in raw_lower:
        return _load_ovf_binary(path, raw)

    raise ValueError(
        "Unsupported OVF data format. Could not find Data Text or Data Binary block."
    )


# ============================================================
# 6) Plotting
# ============================================================

def slugify(text: str, max_len: int = 180) -> str:
    text = str(text).replace("\n", " ")
    text = re.sub(r"[^\w\s\-.=]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len]


def plot_mz(
    mz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    savepath: Path,
    show: bool = False,
    SET_TITLE = False
):
    extent = [
        x.min() * 1e6,
        x.max() * 1e6,
        y.min() * 1e6,
        y.max() * 1e6,
    ]

    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    im = ax.imshow(
        mz.T,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_xlabel(r"$x\;[\mu m]$", fontsize = 12)
    ax.set_ylabel(r"$y\;[\mu m]$", fontsize = 12)
    
    if SET_TITLE:
        ax.set_title(title)

    fig.colorbar(im, ax=ax, label=r"$m_z$")
    fig.tight_layout()
    fig.savefig(savepath, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def radial_fourier_domain_size(
    mz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    period_min_um: float = 0.05,
    period_max_um: float | None = None,
    nbins: int | None = None,
    use_window: bool = True,
):
    """
    Estimate characteristic domain period D_p from radial Fourier spectrum.

    Parameters
    ----------
    mz:
        2D magnetization map, shape (nx, ny).
    x, y:
        1D coordinate arrays in meters.
    period_min_um:
        Smallest allowed period in µm.
    period_max_um:
        Largest allowed period in µm.
        If None, use half of the smaller box length.
    nbins:
        Number of radial bins. If None, chosen automatically.
    use_window:
        Apply Hann window before FFT to reduce edge artifacts.

    Returns
    -------
    result : dict
        Contains k_peak, Dp, spectrum diagnostics.
    radial_data : pd.DataFrame
        Radial spectrum with columns k_rad_per_m, power.
    """

    mz = np.asarray(mz, dtype=float)

    if mz.ndim != 2:
        raise ValueError(f"mz must be 2D, got shape {mz.shape}")

    nx, ny = mz.shape

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))

    Lx = nx * dx
    Ly = ny * dy

    if period_max_um is None:
        period_max_m = 0.5 * min(Lx, Ly)
    else:
        period_max_m = period_max_um * 1e-6

    period_min_m = period_min_um * 1e-6

    if period_min_m <= 0:
        raise ValueError("period_min_um must be positive.")

    if period_max_m <= period_min_m:
        raise ValueError("period_max_um must be larger than period_min_um.")

    # Remove DC component. This is essential, otherwise the mean magnetization dominates.
    mz0 = mz - np.mean(mz)

    mz_std = float(np.std(mz0))

    # If the state is homogeneous, no meaningful Fourier peak exists.
    if mz_std < 1e-8:
        result = {
            "mz_mean": float(np.mean(mz)),
            "mz_std": float(np.std(mz)),
            "up_fraction": float(np.mean(mz > 0)),
            "k_peak_rad_per_m": np.nan,
            "Dp_m": np.nan,
            "Dp_um": np.nan,
            "peak_power": np.nan,
            "total_power": 0.0,
            "valid_peak": False,
            "reason": "homogeneous_or_nearly_homogeneous_mz",
        }

        radial_data = pd.DataFrame(
            columns=["k_rad_per_m", "power", "counts"]
        )

        return result, radial_data

    if use_window:
        wx = np.hanning(nx)
        wy = np.hanning(ny)
        window = np.outer(wx, wy)
        mz0 = mz0 * window

    # FFT
    F = np.fft.fftshift(np.fft.fft2(mz0))
    power = np.abs(F) ** 2

    # Wave-vector grids in rad/m
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    KR = np.sqrt(KX**2 + KY**2)

    # Remove zero-frequency pixel
    nonzero = KR > 0

    k_vals = KR[nonzero].ravel()
    p_vals = power[nonzero].ravel()

    if nbins is None:
        nbins = max(50, min(nx, ny) // 2)

    k_max = np.max(k_vals)
    bins = np.linspace(0.0, k_max, nbins + 1)

    bin_idx = np.digitize(k_vals, bins) - 1

    radial_power = np.zeros(nbins, dtype=float)
    radial_counts = np.zeros(nbins, dtype=int)

    for ii in range(nbins):
        mask = bin_idx == ii
        radial_counts[ii] = int(np.sum(mask))
        if radial_counts[ii] > 0:
            radial_power[ii] = float(np.mean(p_vals[mask]))
        else:
            radial_power[ii] = np.nan

    k_centers = 0.5 * (bins[:-1] + bins[1:])

    radial_data = pd.DataFrame(
        {
            "k_rad_per_m": k_centers,
            "power": radial_power,
            "counts": radial_counts,
        }
    )

    # Restrict peak search to physically meaningful period window.
    k_min = 2.0 * np.pi / period_max_m
    k_max_allowed = 2.0 * np.pi / period_min_m

    search_mask = (
        np.isfinite(radial_power)
        & (k_centers >= k_min)
        & (k_centers <= k_max_allowed)
        & (radial_counts > 0)
    )

    if not np.any(search_mask):
        result = {
            "mz_mean": float(np.mean(mz)),
            "mz_std": float(np.std(mz)),
            "up_fraction": float(np.mean(mz > 0)),
            "k_peak_rad_per_m": np.nan,
            "Dp_m": np.nan,
            "Dp_um": np.nan,
            "peak_power": np.nan,
            "total_power": float(np.nansum(radial_power)),
            "valid_peak": False,
            "reason": "no_radial_bins_in_requested_period_window",
        }

        return result, radial_data

    search_indices = np.where(search_mask)[0]
    peak_idx = search_indices[np.nanargmax(radial_power[search_indices])]

    k_peak = float(k_centers[peak_idx])
    Dp_m = 2.0 * np.pi / k_peak
    Dp_um = Dp_m * 1e6

    result = {
        "mz_mean": float(np.mean(mz)),
        "mz_std": float(np.std(mz)),
        "up_fraction": float(np.mean(mz > 0)),
        "k_peak_rad_per_m": k_peak,
        "Dp_m": Dp_m,
        "Dp_um": Dp_um,
        "peak_power": float(radial_power[peak_idx]),
        "total_power": float(np.nansum(radial_power)),
        "valid_peak": True,
        "reason": "ok",
    }

    return result, radial_data



def row_to_title(row: pd.Series) -> str:
    t_nm = row.get("t_nm", np.nan)
    A_pJ = row.get("A_pJ_per_m", np.nan)
    Keff_kJ = row.get("Keff_kJ_per_m3", np.nan)
    Ku_MJ = row.get("Ku_MJ_per_m3", np.nan)
    Ks_mJ = row.get("Ks_mJ_per_m2", np.nan)
    Ms_MA = row.get("Ms_MA_per_m", np.nan)

    parts = []

    if pd.notna(t_nm):
        parts.append(f"$\\delta={t_nm:.3f}$ nm")

    if pd.notna(A_pJ):
        parts.append(f"$A_s={A_pJ:.2f}$ pJ/m")

    if pd.notna(Ks_mJ):
        parts.append(f"$K_s={Ks_mJ:.3f}$ mJ/m²")

    if pd.notna(Keff_kJ):
        parts.append(f"$K_eff={Keff_kJ:.1f}$ kJ/m³")
    elif pd.notna(Ku_MJ):
        parts.append(f"$K_u={Ku_MJ:.4f}$ MJ/m³")

    if pd.notna(Ms_MA):
        parts.append(f"$M_s={Ms_MA:.2f}$ MA/m")

    return ", ".join(parts) if parts else "mumax3 relaxed domain pattern"


# ============================================================
# 7) Find OVF files from grid summary
# ============================================================

def _candidate_if_exists(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def find_ovf_for_row(base_dir: Path, row: pd.Series) -> Path:
    """
    Prefer scalar mz_final.ovf if available.
    Otherwise fall back to vector m_final.ovf.
    """

    candidates = []

    # Preferred direct columns from updated sweep scripts.
    for col in ["mz_ovf_path", "ovf_path"]:
        if col in row and isinstance(row[col], str) and row[col]:
            p = Path(row[col])
            if not p.is_absolute():
                p = base_dir / p
            candidates.append(p)

    # run_dir column
    if "run_dir" in row and isinstance(row["run_dir"], str) and row["run_dir"]:
        run_dir = Path(row["run_dir"])
        if not run_dir.is_absolute():
            run_dir = base_dir / run_dir

        candidates.append(run_dir / "run.out" / "mz_final.ovf")
        candidates.append(run_dir / "run.out" / "m_final.ovf")

    # run_name fallback
    if "run_name" in row and isinstance(row["run_name"], str) and row["run_name"]:
        run_name = row["run_name"]
        candidates.extend(base_dir.glob(f"**/{run_name}/run.out/mz_final.ovf"))
        candidates.extend(base_dir.glob(f"**/{run_name}/run.out/m_final.ovf"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find mz_final.ovf or m_final.ovf for row.")


# ============================================================
# 8) Overview grid
# ============================================================

def make_overview_grid(plot_paths, outpath: Path, max_panels: int = 36):
    import matplotlib.image as mpimg

    plot_paths = list(plot_paths)[:max_panels]

    if not plot_paths:
        return

    n = len(plot_paths)
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for ax, path in zip(axs, plot_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path.stem[:35], fontsize=7)

    for ax in axs[len(plot_paths):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9) Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data") / Path("mumax_test_final"),
        help="Directory containing grid_summary.csv and run folders.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to grid_summary.csv. Defaults to <base-dir>/grid_summary.csv.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Defaults to <base-dir>/mz_plots.",
    )

    parser.add_argument(
        "--overview",
        action="store_true",
        help="Also create one overview PNG containing the first plots.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively while saving.",
    )


    parser.add_argument(
        "--domain-csv",
        type=Path,
        default=None,
        help="Output CSV for Fourier-estimated domain sizes. Defaults to <base-dir>/domain_size_summary.csv.",
    )

    parser.add_argument(
        "--period-min-um",
        type=float,
        default=0.05,
        help="Minimum allowed domain period for FFT peak search in µm.",
    )

    parser.add_argument(
        "--period-max-um",
        type=float,
        default=None,
        help="Maximum allowed domain period for FFT peak search in µm. Defaults to half the smaller box length.",
    )


    args = parser.parse_args()

    base_dir = args.base_dir
    summary_path = args.summary or (base_dir / "grid_summary.csv")
    out_dir = args.out_dir or (base_dir / "mz_plots")

    out_dir.mkdir(parents=True, exist_ok=True)


    domain_csv_path = args.domain_csv or (base_dir / "domain_size_summary.csv")
    spectrum_dir = base_dir / "radial_spectrum_plots"

    df = pd.read_csv(summary_path)

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    print(f"Loaded {len(df)} successful runs from {summary_path}")
    print(f"Saving plots to {out_dir}")

    plot_paths = []

    domain_rows = []

    CALCULATE_FOURIER = True

    for idx, row in df.iterrows():
        try:
            ovf_path = find_ovf_for_row(base_dir, row)
            x, y, mx, my, mz, meta = load_mumax_ovf(ovf_path)

            title = row_to_title(row)

            if "run_name" in row and isinstance(row["run_name"], str):
                base_name = slugify(row["run_name"])
                fname = base_name + "_mz.png"
            else:
                base_name = f"run_{idx:04d}"
                fname = base_name + "_mz.png"

            savepath = out_dir / fname

            plot_mz(
                mz=mz,
                x=x,
                y=y,
                title=title,
                savepath=savepath,
                show=args.show,
            )


            if CALCULATE_FOURIER:
                # ----------------------------------------------------
                # Fourier domain-size estimate
                # ----------------------------------------------------
                domain_result, radial_data = radial_fourier_domain_size(
                    mz=mz,
                    x=x,
                    y=y,
                    period_min_um=args.period_min_um,
                    period_max_um=args.period_max_um,
                    use_window=True,
                )

                # Combine original grid row + calculated metrics.
                domain_row = row.to_dict()
                domain_row.update(
                    {
                        "ovf_used": str(ovf_path),
                        "png_path": str(savepath),
                        **domain_result,
                    }
                )

                domain_rows.append(domain_row)

                print(
                    f"[OK] {savepath} | "
                    f"Dp={domain_result['Dp_um']:.3f} µm | "
                    f"valid={domain_result['valid_peak']} | "
                    f"reason={domain_result['reason']}"
                )

        except Exception as exc:
            error_row = row.to_dict()
            error_row.update(
                {
                    "ovf_used": "",
                    "png_path": "",
                    "mz_mean": np.nan,
                    "mz_std": np.nan,
                    "up_fraction": np.nan,
                    "k_peak_rad_per_m": np.nan,
                    "Dp_m": np.nan,
                    "Dp_um": np.nan,
                    "peak_power": np.nan,
                    "total_power": np.nan,
                    "valid_peak": False,
                    "reason": f"failed: {exc}",
                }
            )
            domain_rows.append(error_row)

            print(f"[FAILED] row={idx}: {exc}")

    if domain_rows:
        domain_df = pd.DataFrame(domain_rows)
        domain_df.to_csv(domain_csv_path, index=False)
        print(f"Saved domain-size summary: {domain_csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
