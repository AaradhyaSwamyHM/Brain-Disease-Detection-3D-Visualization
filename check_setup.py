"""Print package versions and verify TensorFlow can see hardware."""
import importlib

for name in ("tensorflow", "numpy", "nibabel", "skimage", "matplotlib", "pandas"):
    module = importlib.import_module(name)
    print(f"{name}: {module.__version__}")

import tensorflow as tf
print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))
print("Setup check passed.")
