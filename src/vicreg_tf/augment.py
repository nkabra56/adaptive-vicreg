"""
SSL augmentations: random crop, horizontal flip, and mild color jitter. This
is the standard lightweight SSL trio, kept simple and tf.data-friendly.
"""

from __future__ import annotations
import tensorflow as tf


def color_jitter(x: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    """
    Randomly perturb brightness, contrast, and saturation, then clip back to
    [0, 1]. `s` in [0, 1] scales the jitter strength across all three.
    """
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    return tf.clip_by_value(x, 0.0, 1.0)


def random_augment(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """
    Single randomized view: pad-and-crop (8px border) to `image_size`, random
    horizontal flip, then color jitter. Output is float32 in [0, 1].
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, image_size + 8, image_size + 8)
    image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image


def two_view_map(image: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Return two independently augmented views of the same image, for the VICReg invariance objective."""
    return random_augment(image, image_size), random_augment(image, image_size)
