"""
Augmentation utilities for VICReg-style pretraining.

This module provides simple spatial and color transforms that are fast on CPU
or GPU and safe to use inside tf.data maps.
"""

from __future__ import annotations
import tensorflow as tf


def color_jitter(x: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    """
    Apply light color jitter and clip back to [0, 1].

    Args:
        x: Image tensor in [0, 1] float32 or uint8 convertible to float32.
        s: Strength scalar in [0, 1]. Larger = more aggressive jitter.

    Returns:
        Transformed image tensor in [0, 1].
    """
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    return tf.clip_by_value(x, 0.0, 1.0)


def random_augment(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """
    Basic SSL-style crop + flip + jitter for a single view.

    Args:
        image: HWC image (uint8 or float) with channels last.
        image_size: Final square side after random crop.

    Returns:
        Augmented float32 image in [0, 1].
    """
    # Convert to float and standardize shape
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Pad by 8 pixels and then random crop back to the target size
    image = tf.image.resize_with_crop_or_pad(image, image_size + 8, image_size + 8)
    image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Color jitter
    image = color_jitter(image, s=0.5)
    return image


def two_view_map(image: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Produce two independent augmented views of the same input image.

    Args:
        image: Input image.
        image_size: Target crop size.

    Returns:
        Tuple (view1, view2) after independent augmentation.
    """
    return random_augment(image, image_size), random_augment(image, image_size)
