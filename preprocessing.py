#!/usr/bin/env python3
"""
Read left/right circular polarization image stacks together with the
associated spectroscopy metadata and reproduce the magneto-optical
contrast plot from the MATLAB reference scripts by Dr. Thomas Jauk.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


@dataclass
class _DatEntry:
    header: Dict[str, str]
    energy: float
    fov: float
    consumed: bool = False
    source_name: str = ""


@dataclass
class FrameMetadata:
    """Metadata describing a single TIFF slice and its acquisition settings."""

    filename: str
    timestamp: str
    institution: str
    sample: str
    photon_energy: float
    magnification_fov: float
    contrast_aperture: float
    energy: float
    fov: float
    ef: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "filename": self.filename,
            "timestamp": self.timestamp,
            "institution": self.institution,
            "sample": self.sample,
            "photon_energy": self.photon_energy,
            "magnification_fov": self.magnification_fov,
            "contrast_aperture": self.contrast_aperture,
            "energy": self.energy,
            "fov": self.fov,
            "ef": self.ef,
        }


@dataclass
class Dataset:
    """Container bundling the image stack and auxiliary metadata."""

    stack: torch.Tensor  # shape: (num_slices, height, width)
    frame_metadata: List[FrameMetadata]
    ef: float  # Fermi level (arb.)

    @property
    def energy(self) -> torch.Tensor:
        values = [meta.energy for meta in self.frame_metadata]
        if not values:
            return torch.empty(0, dtype=torch.float32)
        return torch.tensor(values, dtype=torch.float32)

    @property
    def fov(self) -> float:
        values = [meta.fov for meta in self.frame_metadata if not math.isnan(meta.fov)]
        if not values:
            return float("nan")
        return float(np.mean(values))

    def metadata_as_columns(self) -> Dict[str, Any]:
        columns: Dict[str, Any] = {
            "FILENAME": [],
            "TIMESTAMP": [],
            "INSTITUTION": [],
            "SAMPLE": [],
            "E_PH": [],
            "M_FOV": [],
            "CONTRAST_APERTURE": [],
            "ENERGY": [],
            "FOV": [],
        }

        for meta in self.frame_metadata:
            columns["FILENAME"].append(meta.filename)
            columns["TIMESTAMP"].append(meta.timestamp)
            columns["INSTITUTION"].append(meta.institution)
            columns["SAMPLE"].append(meta.sample)
            columns["E_PH"].append(meta.photon_energy)
            columns["M_FOV"].append(meta.magnification_fov)
            columns["CONTRAST_APERTURE"].append(meta.contrast_aperture)
            columns["ENERGY"].append(meta.energy)
            columns["FOV"].append(meta.fov)

        columns["EF"] = self.ef
        return columns

    def save_metadata_json(self, path: Path | str) -> None:
        payload = self.metadata_as_columns()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


_TOKEN_PATTERN = re.compile(r'"[^"]*"|\S+')


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    stem = Path(value).stem if "." in value else value
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def _parse_datasum_line(line: str) -> Tuple[str, float, float] | None:
    tokens = _TOKEN_PATTERN.findall(line.strip())
    if len(tokens) < 12:
        return None
    try:
        energy = float(tokens[0])
        fov = float(tokens[8])
    except ValueError:
        return None
    filename = tokens[11].strip('"')
    return filename, energy, fov


def _to_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _read_dat_metadata(dat_files: Iterable[Path], tif_files: List[Path]) -> Tuple[List[FrameMetadata], float]:
    entries: List[_DatEntry] = []
    ordered_indices: List[int] = []
    aggregated: Dict[str, List[int]] = {}
    ef_value = float("nan")

    for dat_path in sorted(dat_files):
        header: Dict[str, str] = {}
        section = None
        file_indices: List[int] = []

        with dat_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line.strip("[]").upper()
                    continue

                if section == "HEAD" and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    cleaned_value = value.strip().strip('"')
                    header[key.upper()] = cleaned_value
                    normalized_key = re.sub(r"[\s_\-]", "", key).upper()
                    if normalized_key.startswith("EF") and math.isnan(ef_value):
                        parsed = _to_float(cleaned_value.split()[0] if cleaned_value else "")
                        if not math.isnan(parsed):
                            ef_value = parsed
                    continue

                if section == "DATASUM":
                    parsed = _parse_datasum_line(line)
                    if parsed is None:
                        continue
                    filename, energy, fov = parsed
                    idx = len(entries)
                    normalized = _normalize_name(filename)
                    entries.append(
                        _DatEntry(
                            header=header.copy(),
                            energy=energy,
                            fov=fov,
                            source_name=normalized or filename,
                        )
                    )
                    file_indices.append(idx)
                    if normalized:
                        aggregated.setdefault(normalized, []).append(idx)

        if not file_indices:
            idx = len(entries)
            entries.append(
                _DatEntry(
                    header=header.copy(),
                    energy=float("nan"),
                    fov=float("nan"),
                    source_name=_normalize_name(dat_path.stem),
                )
            )
            file_indices.append(idx)

        ordered_indices.extend(file_indices)

    if math.isnan(ef_value):
        LOGGER.warning("Fermi level (Ef) not found in DAT headers; defaulting to NaN.")

    tif_sorted = sorted(tif_files)
    fallback_pos = 0
    metadata: List[FrameMetadata] = []

    def _consume_from_indices(indices: List[int]) -> _DatEntry | None:
        while indices:
            idx = indices.pop(0)
            entry = entries[idx]
            if entry.consumed:
                continue
            entry.consumed = True
            return entry
        return None

    while fallback_pos < len(ordered_indices) and entries[ordered_indices[fallback_pos]].consumed:
        fallback_pos += 1

    for tif_path in tif_sorted:
        normalized = _normalize_name(tif_path.name)
        entry: _DatEntry | None = None

        if normalized and normalized in aggregated:
            bucket = aggregated[normalized]
            entry = _consume_from_indices(bucket)
            if not bucket or all(entries[idx].consumed for idx in bucket):
                aggregated.pop(normalized, None)
        if entry is None:
            while fallback_pos < len(ordered_indices):
                idx = ordered_indices[fallback_pos]
                fallback_pos += 1
                candidate = entries[idx]
                if candidate.consumed:
                    continue
                candidate.consumed = True
                entry = candidate
                break

        if entry is None:
            entry = _DatEntry(header={}, energy=float("nan"), fov=float("nan"))

        metadata.append(
            FrameMetadata(
                filename=tif_path.name,
                timestamp=entry.header.get("TIMESTAMP", ""),
                institution=entry.header.get("INSTITUTION", ""),
                sample=entry.header.get("SAMPLE", ""),
                photon_energy=_to_float(entry.header.get("E_PH")),
                magnification_fov=_to_float(entry.header.get("M_FOV")),
                contrast_aperture=_to_float(entry.header.get("CONTRAST_APERTURE")),
                energy=entry.energy,
                fov=entry.fov,
                ef=ef_value,
            )
        )

    return metadata, ef_value


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


def _read_tiff_stack(tif_files: List[Path]) -> torch.Tensor:
    slices = [_pil_image_to_tensor(path) for path in tif_files]
    stack = torch.stack(slices, dim=0)
    return stack


def load_dataset(folder: Path | str) -> Dataset:
    folder = Path(folder).expanduser()
    dat_files = _find_files(folder, "*.DAT")
    tif_files = _find_files(folder, "*.TIF")
    stack = _read_tiff_stack(tif_files)
    metadata, ef = _read_dat_metadata(dat_files, tif_files)
    print("Stack shape: ", stack.shape)

    return Dataset(stack=stack, frame_metadata=metadata, ef=ef)


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
    """
    Implemented as same function method as From Dr. Jauk
    
    
    _average_filter(stack, kernel_size) pads each slice with a reflection border, then convolves with a kernel filled with ones normalized by its area. 
    Effectively, it replaces every pixel with the local mean over a kernel_size × kernel_size neighbourhood, producing a smooth “flat-field” version of the stack.

    builds a binary mask from the first RCP slice where intensities exceed mask_threshold
    
    _average_filter to estimate the local background for each, and divides by those backgrounds to flatten illumination.

    magneto-optical contrast (R − L) / (R + L) slice by slice, yielding the magneto-circular dichroism tensor used later for plotting/export.
    
    outptus MCD ... Magneto Circular dichroism tensor
    """
    
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
    mcd = (norm_rcp - norm_lcp) / (norm_rcp + norm_lcp) #+ 1e-6
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
        filename = output_path / f"{prefix}_{idx+1:03d}{suffix}"
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

    print("lcp", dataset_lcp.fov)
    print("rcp", dataset_rcp.fov)

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
        ax.set_title(f"MCD slice {idx+1}")
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

    dataset_lcp.save_metadata_json(str( args.save / "metadata.csv" ))
    LOGGER.info(f"Succesfully saved metadata to {str( args.save / "metadata.csv" )}.")

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
