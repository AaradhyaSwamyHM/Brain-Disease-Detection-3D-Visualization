# BraTS MRI enhancement + 3D tumour segmentation

This is a complete, TensorFlow project built from the supplied
BraTS preparation, custom generator and 3D U-Net code.  It keeps the original
three MRI inputs (**FLAIR, T1ce, T2**) and four label indices, while adding
safe paths, validation splitting, MRI enhancement, checkpointing, and NIfTI
exports.

> **Important:** This is a research/education pipeline, not a clinical device.
> Image enhancement improves display/model input contrast; it does not create
> diagnostic information or guarantee clinical image quality.

## What it creates

For each input case, `preprocess_brats.py` writes:

* `enhanced/<case>/*_enhanced.nii.gz` – full-resolution enhanced MRI volumes
  with the original spatial metadata.
* `processed/images/<case>.nii.gz` – normalized/cropped 4-D training images
  (X × Y × Z × 3 channels: FLAIR, T1ce, T2).
* `processed/masks/<case>.nii.gz` – normalized/cropped 3-D training masks.
* `manifest.csv` – deterministic train/validation split.


`predict.py` writes a full-size `*_segmentation.nii.gz` (BraTS labels
`0, 1, 2, 4`), an optional probability NIfTI, and a PNG preview.

## 1. Install

Use Python 3.10 or 3.11 in a virtual environment. From this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_setup.py
python verify_project.py
```

`verify_project.py` runs a synthetic, non-destructive check of enhancement,
NIfTI I/O, the training loader, and a 3-D U-Net forward pass.

For NVIDIA GPU training, install a TensorFlow build compatible with your
Windows/WSL and CUDA setup before running the project. CPU works, but a
128³ 3D U-Net is slow and memory intensive. Start with `--batch-size 1` and
`--base-filters 8` if GPU memory is limited.

## 2. Expected BraTS directory layout


Both `.nii` and `.nii.gz` are accepted.  Training cases need `seg`; test cases
may omit it. The modalities must be in the same voxel grid per case.

## 3. Prepare and enhance the dataset

```powershell
python preprocess_brats.py `
  --input "D:\datasets\BraTS2020\MICCAI_BraTS2020_TrainingData" `
  --output "data\brats_ready_nii" `
  --enhancement clahe `
  --crop-size 128 128 128
```

Enhancement is non-zero percentile clipping followed by slice-wise CLAHE and
min-max normalization. Use `--enhancement none` for a baseline.  The standard
BraTS label `4` is changed to model label `3` during training, then restored
to label `4` in the exported prediction.

## 4. Train

```powershell
python train.py `
  --data "data\brats_ready_nii" `
  --output "runs\brats_v1" `
  --epochs 100 --batch-size 1 --base-filters 16
```

The best model is saved as `runs\brats_v1\best_model.keras`; the final model,
history CSV, and loss chart are also saved there. Resume with
`--resume runs\brats_v1\last_model.keras`.

## 5. Predict a case and save NIfTI output

```powershell
python predict.py `
  --model "runs\brats_v1\best_model.keras" `
  --case "D:\datasets\BraTS2020\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001" `
  --output "predictions\BraTS20_Training_001" `
  --enhancement clahe --save-probability
```

The key result is `predictions\BraTS20_Training_001\BraTS20_Training_001_segmentation.nii.gz`.
Open it over the source MRI in 3D Slicer, ITK-SNAP, or MRIcroGL.

## Files

| File | Role |
| --- | --- |
| `preprocess_brats.py` | discovers subjects, enhances images, saves processed NIfTI only |
| `data.py` | efficient Keras `Sequence` loader (replacement for `233_custom_datagen.py`) |
| `unet3d.py` | corrected reusable 3D U-Net (from `simple_3d_unet.py`) |
| `losses.py` | Dice + sparse cross-entropy loss and Dice metric |
| `train.py` | training, validation, checkpoints, plots |
| `predict.py` | enhancement + inference + full-size NIfTI/PNG export |
