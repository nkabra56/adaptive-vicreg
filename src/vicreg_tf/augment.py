"""
VICReg TensorFlow augmentation utilities.

This module builds graph-safe image augmentations for tf.data pipelines.
It avoids Python-side iteration over Tensors and keeps all branching inside
TensorFlow control flow (tf.cond). It also standardizes channels using
ImageNet mean and std to stabilize optimization across backbones.

Author: Nishant Kabra
Date: 11/8/2025
"""
from typing import Tuple
import tensorflow as tf

# ImageNet normalization constants as float32 tensors for numerical stability
_IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _standardize_channels(x: tf.Tensor) -> tf.Tensor:
    """
    Standardize channels with ImageNet mean and std.

    Args:
        x: Float32 tensor with shape [H, W, C] in [0, 1].

    Returns:
        Float32 tensor with shape [H, W, C] standardized channel-wise.
    """
    # Ensure shape compatibility by broadcasting over [H, W, C]
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


def _color_jitter(x: tf.Tensor,
                  b: float = 0.4,
                  c: float = 0.4,
                  s: float = 0.4,
                  p: float = 0.8) -> tf.Tensor:
    """
    Apply brightness, contrast, and saturation jitter with probability p.

    All randomness happens in graph via tf.random.uniform and tf.cond.

    Args:
        x: Float32 tensor [H, W, C] in [0, 1]
        b: Brightness max delta
        c: Contrast factor range
        s: Saturation factor range
        p: Probability to apply jitter

    Returns:
        Augmented image in [0, 1].
    """
    x = tf.clip_by_value(x, 0.0, 1.0)
    u = tf.random.uniform([], 0.0, 1.0)

    def _apply():
        y = tf.image.random_brightness(x, max_delta=b)
        y = tf.image.random_contrast(y, lower=1.0 - c, upper=1.0 + c)
        y = tf.image.random_saturation(y, lower=1.0 - s, upper=1.0 + s)
        y = tf.clip_by_value(y, 0.0, 1.0)
        return y

    return tf.cond(u < p, _apply, lambda: x)


def _maybe_grayscale(x: tf.Tensor, p: float = 0.2) -> tf.Tensor:
    """
    Convert to grayscale with probability p and back to 3 channels.

    Args:
        x: Float32 tensor [H, W, C] in [0, 1]
        p: Probability to apply grayscale

    Returns:
        Float32 [H, W, C] in [0, 1]
    """
    u = tf.random.uniform([], 0.0, 1.0)

    def _do_gray():
        g = tf.image.rgb_to_grayscale(x)               # [H, W, 1]
        return tf.image.grayscale_to_rgb(g)            # [H, W, 3]

    return tf.cond(u < p, _do_gray, lambda: x)


def _maybe_blur_avgpool(x: tf.Tensor, p: float = 0.2) -> tf.Tensor:
    """
    Cheap blur using average pooling, applied with probability p.

    Works inside tf.data graph mode by avoiding Python loops and by operating
    on a rank-4 tensor as required by tf.nn.avg_pool2d.

    Args:
        x: Float32 tensor [H, W, C]
        p: Probability to apply blur

    Returns:
        Float32 tensor [H, W, C]
    """
    x = tf.cast(x, tf.float32)
    x4 = tf.expand_dims(x, axis=0)  # [1, H, W, C]

    u = tf.random.uniform([], 0.0, 1.0)
    v = tf.random.uniform([], 0.0, 1.0)

    # Precompute both kernel sizes and choose in-graph
    blur3 = tf.nn.avg_pool2d(x4, ksize=3, strides=1, padding='SAME')
    blur5 = tf.nn.avg_pool2d(x4, ksize=5, strides=1, padding='SAME')
    blurred4 = tf.cond(v < 0.5, lambda: blur3, lambda: blur5)
    blurred = tf.squeeze(blurred4, axis=0)  # [H, W, C]

    return tf.cond(u < p, lambda: blurred, lambda: x)


def augment_one_view(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """
    Build one augmented view of the input image in a graph-safe way.

    Pipeline:
      1) Convert to float32 and resize to [image_size, image_size]
      2) Random horizontal flip
      3) Color jitter with probability
      4) Optional grayscale
      5) Optional cheap blur via avg_pool2d
      6) Channel standardization (ImageNet mean/std)

    Args:
        image: Tensor [H, W, 3] or uint8
        image_size: Target size for both height and width

    Returns:
        Float32 tensor [image_size, image_size, 3] standardized.
    """
    # Convert to float [0, 1] early for augmentations
    img = tf.image.convert_image_dtype(image, dtype=tf.float32)
    img = tf.image.resize(img, [image_size, image_size], antialias=True)
    img = tf.image.random_flip_left_right(img)
    img = _color_jitter(img, b=0.4, c=0.4, s=0.4, p=0.8)
    img = _maybe_grayscale(img, p=0.2)
    img = _maybe_blur_avgpool(img, p=0.2)
    img = _standardize_channels(img)
    return img


def make_two_views(image: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Produce two independently augmented views of the same image.

    Args:
        image: Tensor [H, W, 3] or uint8
        image_size: Target side length

    Returns:
        A pair (view1, view2), each Float32 [image_size, image_size, 3]
    """
    v1 = augment_one_view(image, image_size)
    v2 = augment_one_view(image, image_size)
    return v1, v2
