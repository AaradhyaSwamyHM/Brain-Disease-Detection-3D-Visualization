"""Serializable loss and metric functions without external segmentation packages."""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="brats")
def mean_foreground_dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.int32)
    one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1], dtype=y_pred.dtype)
    axes = (0, 1, 2, 3)
    intersection = tf.reduce_sum(one_hot * y_pred, axis=axes)
    denominator = tf.reduce_sum(one_hot + y_pred, axis=axes)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return tf.reduce_mean(dice[1:])  # background is deliberately excluded


@tf.keras.utils.register_keras_serializable(package="brats")
def dice_crossentropy_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return (1.0 - mean_foreground_dice(y_true, y_pred)) + tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
