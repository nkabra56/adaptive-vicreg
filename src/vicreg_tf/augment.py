"""
Augmentation utilities for VICReg-style pretraining.

What this module does
---------------------
I keep my image augmentations lightweight and tf.data-friendly so they work on
CPU or GPU without breaking the input pipeline. These transforms are the
“standard SSL trio”: random crop, horizontal flip, and mild color jitter.
They're intentionally simple to keep the pretext task focused on invariances
that VICReg tries to learn.

How I use it
------------
- `random_augment` builds a *single* view (crop + flip + jitter) for an image.
- `two_view_map` builds *two independent* views, which I feed to the VICReg
  trainer as (x1, x2).
- `color_jitter` applies brightness/contrast/saturation changes and clamps
  results back to [0, 1].

Example
-------
>>> # Inside my tf.data pipeline:
>>> ds = images_ds.map(lambda img: two_view_map(img, image_size=32),
...                    num_parallel_calls=tf.data.AUTOTUNE)

Author: Nishant Kabra
Date: 11/17/2025
"""

from __future__ import annotations
import tensorflow as tf


def color_jitter(x: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    """
    Apply light color jitter and clip back to [0, 1].

    Why I wrote this
    ----------------
    I want a fast, no-surprises color augmentation that nudges brightness,
    contrast, and saturation but keeps pixel values valid for downstream ops.
    Strength `s` scales the effect across all components.

    Parameters
    ----------
    x : tf.Tensor
        Image tensor in [0, 1] float32, or uint8 convertible to float32.
        Shape is H x W x 3 (channels-last).
    s : float, default=0.5
        Jitter strength in [0, 1]. Larger means more aggressive color changes.

    Returns
    -------
    tf.Tensor
        Transformed image tensor in [0, 1] float32 with the same shape as input.

    Notes
    -----
    - I keep the deltas proportional to `s` so I can tune augmentation quickly.
    - I clamp at the end to avoid out-of-range values due to random ops.
    """
    # Randomly adjust brightness; max_delta scales with my strength knob.
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    # Randomly adjust contrast with symmetric lower/upper bounds around 1.0.
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    # Randomly adjust saturation in the same proportional range.
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    # Clamp to the valid range to keep the pipeline stable.
    return tf.clip_by_value(x, 0.0, 1.0)


def random_augment(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """
    Basic SSL-style crop + flip + jitter for a single view.

    Why I wrote this
    ----------------
    For VICReg-style pretraining I need a *single* randomized view that is
    cheap and consistent. I standardize the image to float32, do a padded crop
    to introduce spatial diversity, optionally flip, then apply color jitter.

    Parameters
    ----------
    image : tf.Tensor
        Input H x W x 3 image (uint8 or float). Channels-last is assumed.
    image_size : int
        Final square side after random crop (e.g., 32 for CIFAR-sized inputs).

    Returns
    -------
    tf.Tensor
        Augmented float32 image in [0, 1] with shape (image_size, image_size, 3).

    Notes
    -----
    - I use `resize_with_crop_or_pad` to add an 8px border before cropping.
      This mimics the standard CIFAR “pad-and-crop” trick used in many SSL
      baselines without resizing artifacts.
    """
    # Convert to float in [0, 1] and ensure dtype compatibility downstream.
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Symmetrically pad to (image_size+8) so the random crop can move around.
    image = tf.image.resize_with_crop_or_pad(image, image_size + 8, image_size + 8)
    # Take a random spatial crop back down to the target size.
    image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    # Apply a left-right flip with p=0.5 for additional spatial augmentation.
    image = tf.image.random_flip_left_right(image)
    # Nudge color statistics in a controlled way.
    image = color_jitter(image, s=0.5)
    return image


def two_view_map(image: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Produce two independent augmented views of the same input image.

    Why I wrote this
    ----------------
    The VICReg objective needs *two* different views of the same sample to
    encourage invariance. I simply call my `random_augment` twice so each view
    is independently randomized.

    Parameters
    ----------
    image : tf.Tensor
        Input H x W x 3 image (uint8 or float), channels-last.
    image_size : int
        Target crop size for both views.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        A pair `(view1, view2)`, each float32 in [0, 1] with shape
        (image_size, image_size, 3).

    Examples
    --------
    >>> v1, v2 = two_view_map(img, 32)  # feed (v1, v2) into my trainer
    """
    # Create view 1 and view 2 independently so they differ in crop/jitter/flip.
    return random_augment(image, image_size), random_augment(image, image_size)
