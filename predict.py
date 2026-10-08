"""Enhance one BraTS case, run inference, and write full-size NIfTI predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from skimage.measure import label

from common import MODALITIES, MODEL_TO_BRATS, crop_and_pad, crop_bounds, enhance_mri, find_nifti, load_volume, restore_crop, save_json


def remove_tiny_components(mask: np.ndarray, threshold: int) -> np.ndarray:
    if threshold <= 0:
        return mask
    cleaned = mask.copy()
    for class_id in (1, 2, 3):
        components = label(mask == class_id, connectivity=1)
        for component_id in range(1, components.max() + 1):
            if np.count_nonzero(components == component_id) < threshold:
                cleaned[components == component_id] = 0
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True, help="One folder containing flair, t1ce and t2 NIfTI volumes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enhancement", choices=("clahe", "none"), default="clahe")
    parser.add_argument("--save-probability", action="store_true")
    parser.add_argument("--min-component-size", type=int, default=20)
    args = parser.parse_args()
    if not args.case.is_dir():
        parser.error("--case must be a subject directory")
    paths = {m: find_nifti(args.case, m) for m in MODALITIES}
    if not all(paths.values()):
        parser.error("Case must contain files ending in _flair, _t1ce and _t2 (.nii or .nii.gz).")
    model = tf.keras.models.load_model(args.model, compile=False)
    crop_size = tuple(int(value) for value in model.input_shape[1:4])
    volumes, reference = [], None
    args.output.mkdir(parents=True, exist_ok=True)
    for modality in MODALITIES:
        volume, image = load_volume(paths[modality])
        if reference is None:
            reference, original_shape = image, volume.shape
        elif volume.shape != original_shape:
            raise ValueError("Input modality grids differ; register/resample them before inference.")
        enhanced = enhance_mri(volume, args.enhancement)
        volumes.append(enhanced)
        nib.save(nib.Nifti1Image(enhanced, image.affine, image.header),
                 str(args.output / f"{args.case.name}_{modality}_enhanced.nii.gz"))
    image_tensor = np.stack(volumes, axis=-1)
    start, _ = crop_bounds(original_shape, np.any(image_tensor > 0, axis=-1), crop_size)
    crop = crop_and_pad(image_tensor, start, crop_size)[None, ...]
    probability_crop = model.predict(crop, verbose=0)[0]
    model_mask_crop = np.argmax(probability_crop, axis=-1).astype(np.uint8)
    model_mask_crop = remove_tiny_components(model_mask_crop, args.min_component_size)
    model_mask = restore_crop(model_mask_crop, original_shape, start)
    brats_mask = MODEL_TO_BRATS[model_mask]
    seg_path = args.output / f"{args.case.name}_segmentation.nii.gz"
    nib.save(nib.Nifti1Image(brats_mask, reference.affine, reference.header), str(seg_path))
    if args.save_probability:
        probability = restore_crop(probability_crop, (*original_shape, 4), start)
        nib.save(nib.Nifti1Image(probability.astype(np.float32), reference.affine, reference.header),
                 str(args.output / f"{args.case.name}_probability.nii.gz"))
    z = int(np.argmax(np.count_nonzero(model_mask, axis=(0, 1)))) if np.any(model_mask) else original_shape[2] // 2
    plt.figure(figsize=(10, 4)); plt.subplot(1, 2, 1); plt.imshow(volumes[0][:, :, z], cmap="gray"); plt.title("Enhanced FLAIR"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(volumes[0][:, :, z], cmap="gray"); plt.imshow(model_mask[:, :, z], cmap="turbo", alpha=0.55, vmin=0, vmax=3); plt.title("Predicted segmentation"); plt.axis("off")
    plt.tight_layout(); plt.savefig(args.output / f"{args.case.name}_preview.png", dpi=180); plt.close()
    save_json(args.output / "inference.json", {"case": args.case.name, "model": str(args.model), "crop_size": crop_size, "crop_start": start, "enhancement": args.enhancement})
    print(f"Saved segmentation NIfTI: {seg_path}")


if __name__ == "__main__":
    main()
