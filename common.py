"""Shared image, NIfTI, crop and subject-discovery helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from skimage import exposure

MODALITIES = ("flair", "t1ce", "t2")
MODEL_TO_BRATS = np.array([0, 1, 2, 4], dtype=np.uint8)


def nii_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def find_nifti(case_dir: Path, suffix: str) -> Path | None:
    matches = sorted(p for p in case_dir.iterdir() if p.is_file() and
                     (p.name.endswith(".nii") or p.name.endswith(".nii.gz")) and
                     nii_stem(p).lower().endswith(f"_{suffix.lower()}"))
    return matches[0] if matches else None


def discover_cases(root: Path, require_mask: bool = False) -> list[dict]:
    cases = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = {m: find_nifti(case_dir, m) for m in MODALITIES}
        mask = find_nifti(case_dir, "seg")
        if all(paths.values()) and (mask is not None or not require_mask):
            cases.append({"id": case_dir.name, "dir": case_dir, "modalities": paths, "mask": mask})
    if not cases:
        raise FileNotFoundError(
            f"No valid cases under {root}. Each case needs flair, t1ce and t2 NIfTI files.")
    return cases


def load_volume(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D NIfTI at {path}, got shape {data.shape}")
    return data, image


def enhance_mri(volume: np.ndarray, mode: str = "clahe") -> np.ndarray:
    """Clip non-zero intensities, apply optional slice-wise CLAHE, normalize 0..1."""
    volume = np.asarray(volume, dtype=np.float32)
    foreground = volume > 0
    if not foreground.any():
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(volume[foreground], (0.5, 99.5))
    clipped = np.clip(volume, low, high)
    normalized = np.zeros_like(clipped, dtype=np.float32)
    normalized[foreground] = (clipped[foreground] - low) / max(high - low, 1e-6)
    normalized = np.clip(normalized, 0, 1)
    if mode == "none":
        return normalized
    if mode != "clahe":
        raise ValueError("enhancement mode must be 'clahe' or 'none'")
    enhanced = np.zeros_like(normalized)
    # 2-D CLAHE avoids excessive 3-D memory use and preserves anatomical slices.
    for z in range(normalized.shape[2]):
        enhanced[:, :, z] = exposure.equalize_adapthist(
            normalized[:, :, z], clip_limit=0.015, kernel_size=(32, 32))
    enhanced[~foreground] = 0
    return enhanced.astype(np.float32)


def crop_bounds(shape: Iterable[int], foreground: np.ndarray, crop_size: tuple[int, int, int]) -> tuple[list[int], list[int]]:
    shape = np.asarray(tuple(shape), dtype=int)
    crop = np.asarray(crop_size, dtype=int)
    if np.any(crop <= 0) or np.any(crop % 16):
        raise ValueError("Each crop dimension must be positive and divisible by 16.")
    coords = np.argwhere(foreground)
    center = (coords.min(axis=0) + coords.max(axis=0)) // 2 if len(coords) else shape // 2
    start = center - crop // 2
    start = np.maximum(start, 0)
    start = np.minimum(start, np.maximum(shape - crop, 0))
    end = np.minimum(start + crop, shape)
    return start.astype(int).tolist(), end.astype(int).tolist()


def crop_and_pad(volume: np.ndarray, start: list[int], crop_size: tuple[int, int, int]) -> np.ndarray:
    crop_size = tuple(crop_size)
    # `volume` can be X×Y×Z or X×Y×Z×channels.  Crop only in spatial axes.
    end = np.minimum(np.asarray(start) + crop_size, volume.shape[:3])
    cropped = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    padding = [(0, crop_size[i] - cropped.shape[i]) for i in range(3)]
    padding.extend([(0, 0)] * (cropped.ndim - 3))  # preserve channels/probabilities
    return np.pad(cropped, padding, mode="constant")


def restore_crop(crop: np.ndarray, original_shape: tuple[int, int, int], start: list[int]) -> np.ndarray:
    """Put a cropped 3-D label map or 4-D probability map back into source space."""
    original_shape = tuple(original_shape)
    spatial_shape = original_shape[:3]
    trailing_shape = original_shape[3:]
    restored = np.zeros(original_shape, dtype=crop.dtype)
    end = np.minimum(np.asarray(start) + np.asarray(crop.shape[:3]), spatial_shape)
    used = tuple(slice(0, int(end[i] - start[i])) for i in range(3)) + tuple(slice(None) for _ in trailing_shape)
    target = (slice(start[0], int(end[0])), slice(start[1], int(end[1])), slice(start[2], int(end[2]))) + tuple(slice(None) for _ in trailing_shape)
    restored[target] = crop[used]
    return restored


def affine_for_crop(affine: np.ndarray, start: list[int]) -> np.ndarray:
    """Return the NIfTI affine whose voxel (0, 0, 0) is `start` in source space."""
    translate = np.eye(4, dtype=float)
    translate[:3, 3] = start
    return np.asarray(affine, dtype=float) @ translate


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
