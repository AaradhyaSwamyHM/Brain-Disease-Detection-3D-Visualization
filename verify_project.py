"""Quick self-test for dependencies and the NIfTI-only BraTS pipeline.

Run this after `pip install -r requirements.txt`.  It uses synthetic data only
and does not change your dataset or trained models.
"""
from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import tensorflow as tf

from common import affine_for_crop, crop_and_pad, crop_bounds, enhance_mri, restore_crop
from data import BraTSSequence, case_ids
from unet3d import simple_unet_model


def main() -> None:
    # Enhancement, 4-D crop/pad, and restoration keep the channel dimension.
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    volume[8:24, 8:24, 8:24] = np.linspace(1, 100, 16 ** 3, dtype=np.float32).reshape(16, 16, 16)
    enhanced = enhance_mri(volume, "clahe")
    assert enhanced.shape == volume.shape and 0 <= enhanced.min() <= enhanced.max() <= 1
    image = np.stack((enhanced, enhanced, enhanced), axis=-1)
    start, _ = crop_bounds(volume.shape, volume > 0, (32, 32, 32))
    crop = crop_and_pad(image, start, (32, 32, 32))
    assert crop.shape == (32, 32, 32, 3)
    assert restore_crop(crop, image.shape, start).shape == image.shape
    assert np.allclose(affine_for_crop(np.eye(4), [2, 3, 4])[:3, 3], [2, 3, 4])

    # Verify actual NIfTI output and the training data loader pattern.
    with TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "processed" / "images").mkdir(parents=True)
        (root / "processed" / "masks").mkdir(parents=True)
        nib.save(nib.Nifti1Image(crop, np.eye(4)), str(root / "processed" / "images" / "synthetic.nii.gz"))
        mask = np.zeros((32, 32, 32), dtype=np.uint8); mask[12:20, 12:20, 12:20] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), str(root / "processed" / "masks" / "synthetic.nii.gz"))
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("id", "split")); writer.writeheader()
            writer.writerow({"id": "synthetic", "split": "train"})
        ids = case_ids(root, "train")
        x, y = BraTSSequence(root, ids, batch_size=1, shuffle=False)[0]
        assert x.shape == (1, 32, 32, 32, 3) and y.shape == (1, 32, 32, 32)

    # A compact forward pass checks the U-Net is compatible with TensorFlow.
    model = simple_unet_model((32, 32, 32, 3), num_classes=4, base_filters=2)
    output = model(tf.zeros((1, 32, 32, 32, 3), dtype=tf.float32), training=False)
    assert tuple(output.shape) == (1, 32, 32, 32, 4)
    print("All checks passed: dependencies, enhancement, NIfTI I/O, loader, and 3-D U-Net.")


if __name__ == "__main__":
    main()
