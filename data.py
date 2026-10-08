"""Keras data loader for preprocessed NIfTI BraTS volumes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib
import tensorflow as tf


def case_ids(data_dir: Path, split: str) -> list[str]:
    manifest = np.genfromtxt(data_dir / "manifest.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    records = np.atleast_1d(manifest)
    ids = [str(row["id"]) for row in records if str(row["split"]) == split]
    # A manifest can survive an interrupted preprocessing run.  Train only on
    # complete NIfTI image/mask pairs, rather than failing later inside Keras.
    if split in {"train", "val"}:
        ready, missing = [], []
        for case_id in ids:
            image_path = data_dir / "processed" / "images" / f"{case_id}.nii.gz"
            mask_path = data_dir / "processed" / "masks" / f"{case_id}.nii.gz"
            if image_path.is_file() and mask_path.is_file():
                ready.append(case_id)
            else:
                missing.append(case_id)
        if missing:
            print(f"Skipping {len(missing)} incomplete {split} cases (missing processed image or mask NIfTI).")
        return ready
    return ids


class BraTSSequence(tf.keras.utils.Sequence):
    """Loads 4-D image NIfTIs and 3-D integer mask NIfTIs lazily."""
    def __init__(self, data_dir: Path, ids: list[str], batch_size: int = 1, shuffle: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.data_dir, self.ids, self.batch_size, self.shuffle = Path(data_dir), list(ids), batch_size, shuffle
        if not self.ids:
            raise ValueError("No cases supplied to data generator.")
        self.order = np.arange(len(self.ids))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.ids) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.order)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        indices = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = [], []
        for item_index in indices:
            case_id = self.ids[item_index]
            image_path = self.data_dir / "processed" / "images" / f"{case_id}.nii.gz"
            mask_path = self.data_dir / "processed" / "masks" / f"{case_id}.nii.gz"
            if not mask_path.exists():
                raise ValueError(f"{case_id} has no mask NIfTI and cannot be used for training.")
            image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
            mask = np.asarray(nib.load(str(mask_path)).dataobj, dtype=np.int32)
            if image.ndim != 4 or image.shape[-1] != 3 or mask.shape != image.shape[:3]:
                raise ValueError(f"Invalid processed NIfTI shape for {case_id}: image {image.shape}, mask {mask.shape}")
            images.append(image)
            masks.append(mask)
        return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.int32)
