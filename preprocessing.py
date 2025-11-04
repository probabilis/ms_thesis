#!/usr/bin/env python3
"""
Read left/right circular polarization image stacks together with the
associated spectroscopy metadata and reproduce the magneto-optical
contrast plot from the MATLAB reference scripts.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Container bundling the image stack and auxiliary metadata."""

    stack: torch.Tensor  # shape: (num_slices, height, width)
    energy: torch.Tensor  # concatenated spectrum (arbitrary length)
    fov: float  # averaged field of view in micrometres
    ef: float  # Fermi level (arb.)


def _find_files(folder: Path | str, pattern: str) -> List[Path]:
    folder = Path(folder)
    variants = {pattern}
    variants.add(pattern.lower())
    variants.add(pattern.upper())

    seen = {}
    for variant in variants:
        for path in folder.glob(variant):
            seen[path] = None

    paths = sorted(seen.keys())
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} (any case) found in {folder}")
    return paths


def _read_energy_and_fov(dat_files: Iterable[Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    energies: List[float] = []
    fovs: List[float] = []

    for path in dat_files:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            # Skip header (90 lines in MATLAB version)
            for _ in range(90):
                next(handle, "")
            for raw in handle:
                # lines with strings or comments are ignored
                parts = raw.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    energies.append(float(parts[0]))
                    fovs.append(float(parts[8]))
                except ValueError:
                    continue

    if not energies:
        raise ValueError("No spectral data extracted from DAT files.")

    return torch.tensor(energies, dtype=torch.float32), torch.tensor(fovs, dtype=torch.float32)


def _read_ef(dat_files: Iterable[Path]) -> float:
    for dat_file in dat_files:
        with dat_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(21):
                next(handle, "")
            for raw in handle:
                if "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip().lower()
                if key.startswith("ef"):
                    try:
                        return float(value.strip().split()[0])
                    except ValueError:
                        break

    LOGGER.warning("Fermi level (Ef) not found in DAT headers; defaulting to NaN.")
    return float("nan")


def _pil_image_to_tensor(path: Path) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as img:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mode = img.mode
        if mode in {"I;16", "I"}:
            tensor = torch.frombuffer(img.tobytes(), dtype=torch.uint16)
        else:
            tensor = torch.frombuffer(img.tobytes(), dtype=torch.uint8)

        tensor = tensor.reshape(img.height, img.width).to(torch.float32)
        return tensor


def _read_tiff_stack(folder: Path) -> torch.Tensor:
    tif_files = _find_files(folder, "*.TIF")
    slices = [_pil_image_to_tensor(path) for path in tif_files]
    stack = torch.stack(slices, dim=0)
    return stack


def load_dataset(folder: Path | str) -> Dataset:
    folder = Path(folder).expanduser()
    dat_files = _find_files(folder, "*.DAT")
    energy, fovs = _read_energy_and_fov(dat_files)
    ef = _read_ef(dat_files)
    stack = _read_tiff_stack(folder)
    print("Stack shape: ",stack.shape)

    return Dataset(
        stack=stack,
        energy=energy,
        fov=float(fovs.mean().item()),
        ef=ef,
    )


def _average_filter(stack: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive.")

    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    pad = (pad_left, pad_right, pad_left, pad_right)

    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=stack.dtype, device=stack.device)
    kernel /= kernel.numel()

    padded = torch.nn.functional.pad(stack.unsqueeze(1), pad=pad, mode="reflect")
    filtered = torch.nn.functional.conv2d(padded, kernel).squeeze(1)
    return filtered


def compute_mcd(rcp: torch.Tensor, lcp: torch.Tensor, mask_threshold: float = 250.0, kernel_size: int = 250) -> torch.Tensor:
    if rcp.shape != lcp.shape:
        raise ValueError("RCP and LCP stacks must share the same shape.")

    print(rcp.shape)

    mask = (rcp[0] > mask_threshold).float()
    mask = mask.unsqueeze(0)
    rcp_masked = rcp * mask
    lcp_masked = lcp * mask

    flat_rcp = _average_filter(rcp_masked, kernel_size).clamp_min(1e-6)
    flat_lcp = _average_filter(lcp_masked, kernel_size).clamp_min(1e-6)

    norm_rcp = rcp_masked / flat_rcp
    norm_lcp = lcp_masked / flat_lcp
    mcd = (norm_rcp - norm_lcp) / (norm_rcp + norm_lcp + 1e-6)
    return mcd


def save_slices(
    stack: torch.Tensor,
    output_dir: Path | str,
    prefix: str = "mcd_slice",
    fmt: str = "csv",
) -> None:
    """Persist each slice of a tensor stack as a CSV or TXT file."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fmt_normalized = fmt.lower()
    if fmt_normalized not in {"csv", "txt"}:
        raise ValueError("fmt must be either 'csv' or 'txt'.")

    suffix = ".csv" if fmt_normalized == "csv" else ".txt"
    delimiter = "," if fmt_normalized == "csv" else " "

    for idx in range(stack.shape[0]):
        filename = output_path / f"{prefix}_{idx:03d}{suffix}"
        slice_array = stack[idx].detach().cpu().numpy()
        np.savetxt(filename, slice_array, delimiter=delimiter)
        LOGGER.info("Saved slice %s to %s", idx, filename)


def plot_results(
    dataset_rcp: Dataset,
    dataset_lcp: Dataset,
    save_path: Path | None = None,
    slice_format: str = "csv",
) -> None:
    mcd = compute_mcd(dataset_rcp.stack, dataset_lcp.stack)
    num_slices = mcd.shape[0]
    cols = min(num_slices, 4)
    rows = (num_slices + cols - 1) // cols
    x_extent = dataset_lcp.fov / 2.0

    if save_path is not None:
        save_slices(mcd, save_path, fmt=slice_format)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes_iter = [axes]
    elif rows == 1 or cols == 1:
        axes_iter = list(axes)
    else:
        axes_iter = [ax for row in axes for ax in row]

    for idx in range(num_slices):
        ax = axes_iter[idx]
        image = mcd[idx].clamp(-0.05, 0.05).detach().cpu().numpy()
        ax.imshow(
            image,
            cmap="gray",
            origin="lower",
            extent=(-x_extent, x_extent, -x_extent, x_extent),
            interpolation="nearest",
        )
        ax.set_title(f"MCD slice {idx}")
        ax.set_aspect("equal")
        ax.set_xlabel("µm")
        ax.set_ylabel("µm")
        ax.grid(False)

    for ax in axes_iter[num_slices:]:
        ax.axis("off")

    fig.suptitle(
        f"MCD overview — Ef LCP: {dataset_lcp.ef:.2f}, Ef RCP: {dataset_rcp.ef:.2f}",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path / "raw_data.png", dpi=300)
        LOGGER.info("Saved figure to %s", save_path)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LCP/RCP stacks and their MCD contrast.")
    parser.add_argument("--lcp", required=True, type=Path, help="Folder containing LCP *.TIF and *.DAT files.")
    parser.add_argument("--rcp", required=True, type=Path, help="Folder containing RCP *.TIF and *.DAT files.")
    parser.add_argument("--save", type=Path, help="Optional path to save the resulting plot instead of showing it.")
    parser.add_argument(
        "--slice-format",
        choices=["csv", "txt"],
        default="csv",
        help="File format used when exporting individual slices (default: csv).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity (DEBUG, INFO, ...).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s | %(message)s")

    dataset_lcp = load_dataset(args.lcp)
    dataset_rcp = load_dataset(args.rcp)

    LOGGER.info("Loaded LCP stack: %s slices | FoV %.2f µm", dataset_lcp.stack.shape[0], dataset_lcp.fov)
    LOGGER.info("Loaded RCP stack: %s slices | FoV %.2f µm", dataset_rcp.stack.shape[0], dataset_rcp.fov)

    plot_results(
        dataset_rcp,
        dataset_lcp,
        save_path=args.save,
        slice_format=args.slice_format,
    )


if __name__ == "__main__":
    main()
