"""Enhance BraTS NIfTI volumes and create cropped NIfTI data for 3-D U-Net training."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from common import MODALITIES, affine_for_crop, crop_and_pad, crop_bounds, discover_cases, enhance_mri, load_volume


def process_case(case: dict, output: Path, crop_size: tuple[int, int, int], enhancement: str) -> dict:
    volumes, reference = [], None
    enhanced_dir = output / "enhanced" / case["id"]
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    for modality in MODALITIES:
        volume, image = load_volume(case["modalities"][modality])
        if reference is None:
            reference = image
            shape = volume.shape
        elif volume.shape != shape:
            raise ValueError(f"{case['id']}: modality grids differ; register/resample them before training.")
        enhanced = enhance_mri(volume, enhancement)
        nib.save(nib.Nifti1Image(enhanced, image.affine, image.header),
                 str(enhanced_dir / f"{case['id']}_{modality}_enhanced.nii.gz"))
        volumes.append(enhanced)

    image_tensor = np.stack(volumes, axis=-1)
    foreground = np.any(image_tensor > 0, axis=-1)
    start, _ = crop_bounds(image_tensor.shape[:3], foreground, crop_size)
    cropped_image = crop_and_pad(image_tensor, start, crop_size).astype(np.float32)
    cropped_affine = affine_for_crop(reference.affine, start)
    image_header = reference.header.copy()
    image_header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(cropped_image, cropped_affine, image_header),
             str(output / "processed" / "images" / f"{case['id']}.nii.gz"))
    item = {"id": case["id"], "split": "test"}
    if case["mask"]:
        mask, _ = load_volume(case["mask"])
        if mask.shape != image_tensor.shape[:3]:
            raise ValueError(f"{case['id']}: segmentation grid differs from modalities.")
        mask = mask.astype(np.uint8)
        mask[mask == 4] = 3
        cropped_mask = crop_and_pad(mask, start, crop_size).astype(np.uint8)
        mask_header = reference.header.copy()
        mask_header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(cropped_mask, cropped_affine, mask_header),
                 str(output / "processed" / "masks" / f"{case['id']}.nii.gz"))
        item["split"] = "unassigned"
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder containing BraTS case folders")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--enhancement", choices=("clahe", "none"), default="clahe")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "processed" / "images").mkdir(parents=True, exist_ok=True)
    (args.output / "processed" / "masks").mkdir(parents=True, exist_ok=True)
    cases = discover_cases(args.input)
    rows = [process_case(case, args.output, tuple(args.crop_size), args.enhancement) for case in cases]
    labelled = [r for r in rows if r["split"] == "unassigned"]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(labelled)
    val_count = max(1, round(len(labelled) * args.val_fraction)) if labelled else 0
    val_ids = {r["id"] for r in labelled[:val_count]}
    for row in rows:
        if row["split"] == "unassigned":
            row["split"] = "val" if row["id"] in val_ids else "train"
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "split"))
        writer.writeheader(); writer.writerows(rows)
    print(f"Prepared {len(rows)} cases at {args.output}; labelled train/val: {len(labelled) - val_count}/{val_count}")


if __name__ == "__main__":
    main()
