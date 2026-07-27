#!/usr/bin/env python3
"""
Plot selected MuMax/OOMMF domain configurations directly from raw field data.

Main use case:
    - Use grid_summary_nice.csv to select only the shortlisted runs.
    - Load the raw field from mz_final.ovf / m_final.ovf, or from a field CSV if present.
    - Plot the selected configurations in two rows: t = 1.3 nm and t = 1.4 nm.

Examples:
    python3 plot_raw_domain_gallery_selected.py \
        --summary grid_summary_nice.csv \
        --out selected_domain_gallery_raw.png

    python3 plot_raw_domain_gallery_selected.py \
        --summary grid_summary_nice.csv \
        --base-dir . \
        --out selected_domain_gallery_raw.png \
        --layout aligned

Notes:
    - The default layout is sequential: exactly two packed rows for t = 1.3 nm and t = 1.4 nm.
    - Use --layout aligned only if you explicitly want common (A, Keff) columns with empty cells where runs are missing.
    - Supports OVF2_TEXT, OVF2_BINARY, scalar mz_final.ovf and vector m_final.ovf.
    - Also supports CSV field files if the summary contains one of:
      csv_path, field_csv_path, raw_csv_path, mz_csv_path.
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


# ============================================================
# 1) Small helpers
# ============================================================

def slugify(text: str, max_len: int = 180) -> str:
    text = str(text).replace("\n", " ")
    text = re.sub(r"[^\w\s\-.=]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len]


def _try_existing_paths(paths: list[Path]) -> Optional[Path]:
    seen = set()
    for p in paths:
        try:
            p = p.expanduser()
        except Exception:
            pass
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            return p
    return None


def resolve_existing_path(raw_path: str | Path, summary_dir: Path, base_dir: Optional[Path]) -> Optional[Path]:
    """Resolve raw_path robustly against cwd, summary_dir, and optional base_dir."""
    if raw_path is None or (isinstance(raw_path, float) and np.isnan(raw_path)):
        return None

    s = str(raw_path).strip()
    if not s or s.lower() == "nan":
        return None

    p = Path(s)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        # 1) relative to current working directory
        candidates.append(p)
        # 2) relative to directory of the summary file
        candidates.append(summary_dir / p)
        # 3) relative to user-provided base-dir
        if base_dir is not None:
            candidates.append(base_dir / p)

        # Common case: summary path already starts with the base directory name,
        # but the script is launched from inside base_dir.
        if base_dir is not None and len(p.parts) > 1:
            if p.parts[0] == base_dir.name:
                candidates.append(base_dir / Path(*p.parts[1:]))

    return _try_existing_paths(candidates)


# ============================================================
# 2) CSV field reader
# ============================================================

def load_field_csv(path: Path):
    """
    Load a saved field CSV.

    Supported formats:
        1) i,j,x_m,y_m,mx,my,mz
        2) x_m,y_m,mz
        3) x,y,mz

    Returns:
        x, y, mx, my, mz, meta
    where x,y are 1D coordinate arrays in meters and mz has shape (nx, ny).
    """
    path = Path(path)
    df = pd.read_csv(path, comment="#")

    # Normalize column names for flexible matching.
    colmap = {c.lower().strip(): c for c in df.columns}

    def find_col(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    i_col = find_col("i", "ix", "x_index")
    j_col = find_col("j", "iy", "y_index")
    x_col = find_col("x_m", "x", "x[m]", "x_meter", "x_meters")
    y_col = find_col("y_m", "y", "y[m]", "y_meter", "y_meters")
    mx_col = find_col("mx", "m_x")
    my_col = find_col("my", "m_y")
    mz_col = find_col("mz", "m_z")

    if mz_col is None:
        raise ValueError(f"Could not find mz column in CSV: {path}")

    if i_col is not None and j_col is not None:
        i = df[i_col].to_numpy(dtype=int)
        j = df[j_col].to_numpy(dtype=int)
        nx = int(i.max()) + 1
        ny = int(j.max()) + 1

        mz = np.full((nx, ny), np.nan, dtype=float)
        mx = np.full((nx, ny), np.nan, dtype=float) if mx_col is not None else None
        my = np.full((nx, ny), np.nan, dtype=float) if my_col is not None else None

        mz[i, j] = df[mz_col].to_numpy(dtype=float)
        if mx_col is not None:
            mx[i, j] = df[mx_col].to_numpy(dtype=float)
        if my_col is not None:
            my[i, j] = df[my_col].to_numpy(dtype=float)

        if x_col is not None:
            x2d = np.full((nx, ny), np.nan, dtype=float)
            x2d[i, j] = df[x_col].to_numpy(dtype=float)
            x = np.nanmean(x2d, axis=1)
        else:
            x = np.arange(nx, dtype=float)

        if y_col is not None:
            y2d = np.full((nx, ny), np.nan, dtype=float)
            y2d[i, j] = df[y_col].to_numpy(dtype=float)
            y = np.nanmean(y2d, axis=0)
        else:
            y = np.arange(ny, dtype=float)

        return x, y, mx, my, mz, {"source_format": "csv_indexed"}

    if x_col is None or y_col is None:
        raise ValueError(
            f"CSV must contain either i,j indices or x/y coordinates. File: {path}"
        )

    x_vals = np.sort(df[x_col].unique())
    y_vals = np.sort(df[y_col].unique())
    nx = len(x_vals)
    ny = len(y_vals)

    x_index = {v: k for k, v in enumerate(x_vals)}
    y_index = {v: k for k, v in enumerate(y_vals)}

    mz = np.full((nx, ny), np.nan, dtype=float)
    mx = np.full((nx, ny), np.nan, dtype=float) if mx_col is not None else None
    my = np.full((nx, ny), np.nan, dtype=float) if my_col is not None else None

    for _, r in df.iterrows():
        ii = x_index[r[x_col]]
        jj = y_index[r[y_col]]
        mz[ii, jj] = float(r[mz_col])
        if mx_col is not None:
            mx[ii, jj] = float(r[mx_col])
        if my_col is not None:
            my[ii, jj] = float(r[my_col])

    return x_vals, y_vals, mx, my, mz, {"source_format": "csv_xy"}


# ============================================================
# 3) OVF reader: supports OVF2_TEXT and OVF2_BINARY
# ============================================================

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


def _reshape_ovf_data(
    arr: np.ndarray,
    header: Dict[str, str],
    nx: int,
    ny: int,
    nz: int,
    valuedim: int,
):
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

    if not np.isfinite(xstepsize):
        xmax = _get_float(header, "xmax", float(nx))
        xstepsize = (xmax - xmin) / nx

    if not np.isfinite(ystepsize):
        ymax = _get_float(header, "ymax", float(ny))
        ystepsize = (ymax - ymin) / ny

    x = xmin + (np.arange(nx) + 0.5) * xstepsize
    y = ymin + (np.arange(ny) + 0.5) * ystepsize

    return x, y, mx, my, mz, header


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
            f"Unexpected text OVF data size in {path}: got {arr.size}, expected {expected}."
        )

    return _reshape_ovf_data(arr, header, nx, ny, nz, valuedim)


def _find_binary_data_start(raw: bytes) -> Tuple[bytes, int, str]:
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
    if precision_tag == "binary4":
        candidates = [(">f4", 1234567.0), ("<f4", 1234567.0)]
        width = 4
    elif precision_tag == "binary8":
        candidates = [(">f8", 123456789012345.0), ("<f8", 123456789012345.0)]
        width = 8
    else:
        raise ValueError(f"Unknown binary precision tag: {precision_tag}")

    for dtype, check_value in candidates:
        check = np.frombuffer(data_block[:width], dtype=np.dtype(dtype), count=1)[0]
        if np.isfinite(check) and np.isclose(check, check_value, rtol=1e-5, atol=1e-5):
            return np.dtype(dtype), width

    raise ValueError("Could not identify OVF binary endianness/check value.")


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

    n_bytes = check_width + n_values * dtype.itemsize
    if len(data_block) < n_bytes:
        raise ValueError(
            f"Binary OVF data block too short in {path}: got {len(data_block)}, expected at least {n_bytes}."
        )

    values = np.frombuffer(
        data_block[check_width:n_bytes],
        dtype=dtype,
        count=n_values,
    ).astype(float)

    return _reshape_ovf_data(values, header, nx, ny, nz, valuedim)


def load_mumax_ovf(path: Path):
    path = Path(path)
    raw = path.read_bytes()
    raw_lower = raw.lower()

    if b"# begin: data text" in raw_lower:
        return _load_ovf_text(path, raw, b"# Begin: Data Text")

    if b"# begin: data binary 4" in raw_lower or b"# begin: data binary 8" in raw_lower:
        return _load_ovf_binary(path, raw)

    raise ValueError("Unsupported OVF data format. Could not find Data Text or Data Binary block.")


# ============================================================
# 4) Resolve raw field data for one summary row
# ============================================================

def find_raw_field_for_row(row: pd.Series, summary_dir: Path, base_dir: Optional[Path]) -> Path:
    """Prefer raw CSV files if present; otherwise use OVF files."""
    csv_cols = [
        "csv_path",
        "field_csv_path",
        "raw_csv_path",
        "mz_csv_path",
        "field_path",
    ]

    ovf_cols = [
        "mz_ovf_path",
        "ovf_path",
        "m_ovf_path",
    ]

    # 1) explicit CSV columns
    for col in csv_cols:
        if col in row:
            p = resolve_existing_path(row[col], summary_dir=summary_dir, base_dir=base_dir)
            if p is not None:
                return p

    # 2) explicit OVF columns
    for col in ovf_cols:
        if col in row:
            p = resolve_existing_path(row[col], summary_dir=summary_dir, base_dir=base_dir)
            if p is not None:
                return p

    # 3) run_dir fallback
    if "run_dir" in row:
        run_dir = resolve_existing_path(row["run_dir"], summary_dir=summary_dir, base_dir=base_dir)
        if run_dir is None:
            # run_dir is a directory, so resolve_existing_path may fail because it checks files.
            s = str(row["run_dir"]).strip()
            if s and s.lower() != "nan":
                p = Path(s)
                dirs = []
                if p.is_absolute():
                    dirs.append(p)
                else:
                    dirs.append(p)
                    dirs.append(summary_dir / p)
                    if base_dir is not None:
                        dirs.append(base_dir / p)
                        if len(p.parts) > 1 and p.parts[0] == base_dir.name:
                            dirs.append(base_dir / Path(*p.parts[1:]))
                for d in dirs:
                    if d.exists() and d.is_dir():
                        run_dir = d
                        break

        if run_dir is not None:
            candidates = [
                run_dir / "run.out" / "mz_final.csv",
                run_dir / "run.out" / "m_final.csv",
                run_dir / "run.out" / "mz_final.ovf",
                run_dir / "run.out" / "m_final.ovf",
            ]
            p = _try_existing_paths(candidates)
            if p is not None:
                return p

    # 4) run_name recursive fallback
    run_name = str(row.get("run_name", "")).strip()
    if run_name:
        roots = [summary_dir]
        if base_dir is not None:
            roots.append(base_dir)
        roots.append(Path.cwd())

        patterns = [
            f"**/{run_name}/run.out/mz_final.csv",
            f"**/{run_name}/run.out/m_final.csv",
            f"**/{run_name}/run.out/mz_final.ovf",
            f"**/{run_name}/run.out/m_final.ovf",
        ]
        for root in roots:
            for pattern in patterns:
                matches = list(root.glob(pattern))
                if matches:
                    return matches[0]

    raise FileNotFoundError(f"Could not resolve raw field file for run: {row.get('run_name', 'unknown')}")


def load_raw_field(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_field_csv(path)
    if suffix == ".ovf" or suffix == ".omf":
        return load_mumax_ovf(path)
    raise ValueError(f"Unsupported raw field file extension: {path}")


# ============================================================
# 5) Gallery plotting
# ============================================================

def make_panel_label(row: pd.Series, show_dp: bool = True) -> str:
    A = row.get("A_pJ_per_m", np.nan)
    Keff = row.get("Keff_kJ_per_m3", np.nan)
    Dp = row.get("Dp_um", np.nan)

    parts = []
    if pd.notna(A):
        parts.append(fr"$A={float(A):.0f}$ pJ/m")
    if pd.notna(Keff):
        parts.append(fr"$K_{{eff}}={float(Keff):.0f}$ kJ/m$^3$")
    if show_dp and pd.notna(Dp):
        parts.append(fr"$D_p={float(Dp):.2f}\,\mu$m")

    return "\n".join(parts)


def plot_one_mz(ax, mz: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, show_axes: bool):
    extent = [x.min() * 1e6, x.max() * 1e6, y.min() * 1e6, y.max() * 1e6]

    im = ax.imshow(
        mz.T,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=8)

    if show_axes:
        ax.set_xlabel(r"$x\,[\mu m]$", fontsize=8)
        ax.set_ylabel(r"$y\,[\mu m]$", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    return im


def get_selected_dataframe(summary_path: Path, thicknesses_nm: list[float]) -> pd.DataFrame:
    df = pd.read_csv(summary_path)

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    if "t_nm" not in df.columns:
        raise KeyError("summary CSV must contain a t_nm column")

    tvals = pd.to_numeric(df["t_nm"], errors="coerce")
    mask = np.zeros(len(df), dtype=bool)
    for t in thicknesses_nm:
        mask |= np.isclose(tvals, t, atol=1e-6)

    df = df[mask].copy()

    sort_cols = [c for c in ["t_nm", "A_pJ_per_m", "Keff_kJ_per_m3"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).copy()

    return df.reset_index(drop=True)


def plot_two_row_gallery(
    df: pd.DataFrame,
    summary_dir: Path,
    base_dir: Optional[Path],
    outpath: Path,
    layout: str = "aligned",
    title: str = "Selected MuMax domain patterns from raw field data",
    dpi: int = 220,
    show_dp: bool = True,
    show_axes: bool = False,
):
    t_values = sorted(pd.to_numeric(df["t_nm"], errors="coerce").dropna().unique())
    if len(t_values) == 0:
        raise ValueError("No selected thickness values found.")

    if layout == "aligned":
        if not {"A_pJ_per_m", "Keff_kJ_per_m3"}.issubset(df.columns):
            raise KeyError("aligned layout requires A_pJ_per_m and Keff_kJ_per_m3 columns")

        col_keys = sorted(
            set(
                zip(
                    pd.to_numeric(df["A_pJ_per_m"], errors="coerce"),
                    pd.to_numeric(df["Keff_kJ_per_m3"], errors="coerce"),
                )
            ),
            key=lambda k: (k[0], k[1]),
        )

        nrows = len(t_values)
        ncols = len(col_keys)
        fig_w = max(2.15 * ncols, 8.0)
        fig_h = max(2.35 * nrows, 4.8)

        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        last_im = None

        for r, t in enumerate(t_values):
            axs[r, 0].set_ylabel(
                fr"$t={t:.2f}$ nm",
                fontsize=11,
                rotation=0,
                labelpad=40,
                va="center",
            )

            for c, (A, Keff) in enumerate(col_keys):
                ax = axs[r, c]
                mask = (
                    np.isclose(pd.to_numeric(df["t_nm"], errors="coerce"), t)
                    & np.isclose(pd.to_numeric(df["A_pJ_per_m"], errors="coerce"), A)
                    & np.isclose(pd.to_numeric(df["Keff_kJ_per_m3"], errors="coerce"), Keff)
                )
                sub = df[mask]

                if sub.empty:
                    ax.axis("off")
                    continue

                row = sub.iloc[0]
                try:
                    raw_path = find_raw_field_for_row(row, summary_dir=summary_dir, base_dir=base_dir)
                    x, y, mx, my, mz, meta = load_raw_field(raw_path)
                    last_im = plot_one_mz(ax, mz, x, y, title=make_panel_label(row, show_dp=show_dp), show_axes=show_axes)

                except Exception as exc:
                    ax.text(0.5, 0.5, f"failed\n{exc}", ha="center", va="center", fontsize=7)
                    ax.axis("off")

    elif layout == "sequential":
        groups = []
        max_cols = 0
        for t in t_values:
            sub = df[np.isclose(pd.to_numeric(df["t_nm"], errors="coerce"), t)].copy()
            sub = sub.sort_values(by=[c for c in ["A_pJ_per_m", "Keff_kJ_per_m3"] if c in sub.columns])
            groups.append((t, sub))
            max_cols = max(max_cols, len(sub))

        nrows = len(groups)
        ncols = max_cols
        fig_w = max(2.25 * ncols, 8.0)
        fig_h = max(2.55 * nrows, 4.8)

        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        last_im = None

        for r, (t, sub) in enumerate(groups):
            axs[r, 0].set_ylabel(
                fr"$t={t:.2f}$ nm",
                fontsize=11,
                rotation=0,
                labelpad=40,
                va="center",
            )

            for c in range(ncols):
                ax = axs[r, c]
                if c >= len(sub):
                    ax.axis("off")
                    continue

                row = sub.iloc[c]
                try:
                    raw_path = find_raw_field_for_row(row, summary_dir=summary_dir, base_dir=base_dir)
                    x, y, mx, my, mz, meta = load_raw_field(raw_path)
                    last_im = plot_one_mz(
                        ax,
                        mz,
                        x,
                        y,
                        title=make_panel_label(row, show_dp=show_dp),
                        show_axes=show_axes,
                    )
                except Exception as exc:
                    ax.text(0.5, 0.5, f"failed\n{exc}", ha="center", va="center", fontsize=7)
                    ax.axis("off")
    else:
        raise ValueError(f"Unknown layout: {layout}")

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.985, 0.965])

    if last_im is not None:
        cbar_ax = fig.add_axes([0.988, 0.14, 0.008, 0.72])
        fig.colorbar(last_im, cax=cbar_ax, label=r"$m_z$")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6) Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("grid_summary_nice.csv"),
        help="CSV containing the selected runs. Default: grid_summary_nice.csv",
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Optional base directory for resolving raw field paths. If omitted, uses paths relative to the summary directory and cwd.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("selected_domain_gallery_raw.png"),
        help="Output image path.",
    )

    parser.add_argument(
        "--thicknesses-nm",
        type=float,
        nargs="+",
        default=[1.0, 1.3, 1.4],
        help="Thickness values to plot as rows. Default: 1.3 1.4",
    )

    parser.add_argument(
        "--layout",
        choices=["sequential", "aligned"],
        default="sequential",
        help="sequential: each thickness row is packed independently. aligned: columns are unique (A,Keff) pairs with blanks for missing runs.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Selected MuMax domain patterns from raw field data",
        help="Figure title.",
    )

    parser.add_argument("--dpi", type=int, default=220, help="Output DPI.")
    parser.add_argument("--show-axes", action="store_true", help="Show x/y axes in every panel.")
    parser.add_argument("--no-dp", action="store_true", help="Do not annotate Dp_um even if present.")

    args = parser.parse_args()

    summary_path = args.summary.resolve()
    summary_dir = summary_path.parent
    base_dir = args.base_dir.resolve() if args.base_dir is not None else None

    outpath = args.out
    if not outpath.is_absolute():
        outpath = summary_dir / outpath

    df = get_selected_dataframe(summary_path, thicknesses_nm=args.thicknesses_nm)

    print(f"Loaded {len(df)} selected rows from {summary_path}")
    print(f"Thickness rows: {args.thicknesses_nm}")
    if len(df) > 0:
        print("Actual t_nm values used:", sorted(pd.to_numeric(df["t_nm"], errors="coerce").dropna().unique()))
    print(f"Output: {outpath}")

    plot_two_row_gallery(
        df=df,
        summary_dir=summary_dir,
        base_dir=base_dir,
        outpath=outpath,
        layout=args.layout,
        title=args.title,
        dpi=args.dpi,
        show_dp=not args.no_dp,
        show_axes=args.show_axes,
    )

    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()