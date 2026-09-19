"""Augmentations for the two-view pipeline: pad-and-crop, horizontal flip and color jitter."""

from __future__ import annotations

import tensorflow as tf


def color_jitter(x: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    """Random brightness, contrast and saturation with strength `s` in [0, 1], clipped back to [0, 1]."""
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    return tf.clip_by_value(x, 0.0, 1.0)


def random_augment(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """One random view as float32 in [0, 1]: pad-and-crop, horizontal flip, color jitter."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Pad to image_size + 8, then crop back: a small random translation.
    image = tf.image.resize_with_crop_or_pad(image, image_size + 8, image_size + 8)
    image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image


def two_view_map(image: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Two independent augmentations of the same image."""
    return random_augment(image, image_size), random_augment(image, image_size)
