"""Reusable 3-D U-Net adapted from the supplied simple_3d_unet.py."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


def _block(x: tf.Tensor, filters: int, dropout: float) -> tf.Tensor:
    x = layers.Conv3D(filters, 3, activation="relu", kernel_initializer="he_uniform", padding="same")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Conv3D(filters, 3, activation="relu", kernel_initializer="he_uniform", padding="same")(x)


def simple_unet_model(input_shape: tuple[int, int, int, int], num_classes: int = 4, base_filters: int = 16) -> Model:
    """Four-level 3-D U-Net. Every spatial input dimension must divide by 16."""
    if any(size % 16 for size in input_shape[:3]):
        raise ValueError("Input spatial dimensions must be divisible by 16.")
    inputs = layers.Input(input_shape)
    c1 = _block(inputs, base_filters, 0.10); p1 = layers.MaxPooling3D()(c1)
    c2 = _block(p1, base_filters * 2, 0.10); p2 = layers.MaxPooling3D()(c2)
    c3 = _block(p2, base_filters * 4, 0.20); p3 = layers.MaxPooling3D()(c3)
    c4 = _block(p3, base_filters * 8, 0.20); p4 = layers.MaxPooling3D()(c4)
    c5 = _block(p4, base_filters * 16, 0.30)

    u6 = layers.Conv3DTranspose(base_filters * 8, 2, strides=2, padding="same")(c5)
    c6 = _block(layers.Concatenate()([u6, c4]), base_filters * 8, 0.20)
    u7 = layers.Conv3DTranspose(base_filters * 4, 2, strides=2, padding="same")(c6)
    c7 = _block(layers.Concatenate()([u7, c3]), base_filters * 4, 0.20)
    u8 = layers.Conv3DTranspose(base_filters * 2, 2, strides=2, padding="same")(c7)
    c8 = _block(layers.Concatenate()([u8, c2]), base_filters * 2, 0.10)
    u9 = layers.Conv3DTranspose(base_filters, 2, strides=2, padding="same")(c8)
    c9 = _block(layers.Concatenate()([u9, c1]), base_filters, 0.10)
    outputs = layers.Conv3D(num_classes, 1, activation="softmax", name="segmentation")(c9)
    return Model(inputs, outputs, name="brats_3d_unet")
