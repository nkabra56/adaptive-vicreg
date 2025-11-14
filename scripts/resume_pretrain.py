"""
Resume / continue pretraining VICReg or Adaptive-VICReg from a checkpoint.

Intent
------
If your pretraining stopped (e.g., Colab timeout), this script reconstructs
the same encoder+projector modules with identical layer names, reloads your
HDF5 weights, and continues training with the VICReg (or Adaptive-VICReg)
loss. It uses a simple two-view CIFAR augmentation pipeline.

Design notes
------------
- Device selection before TF import (cpu|gpu|auto).
- Loss components computed in float32 to avoid mixed-precision surprises.
- Schedules (cosine) for both adaptive targets (γ, ν) and loss weights.
- Uses a small CNN encoder; align with your trainer if you customized it.

Author: Nishant Kabra
Date: 11/10/25
"""
from __future__ import annotations
import os
import math
import argparse

# -------------------- device pre-parse (before TF import!) --------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    # Hide GPUs so TF won't even enumerate them.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# -------------------------------- TensorFlow ----------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------ device helpers --------------------------------
def _enable_mem_growth():
    """
    Enable GPU memory growth (safe on CPU). Helpful on laptops and shared GPUs.
    """
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("[resume_pretrain] set_memory_growth warning:", repr(e))


def _gpu_probe() -> bool:
    """
    Validate that a tiny kernel can run on the GPU (detect invalid PTX, etc.).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform([1, 8, 8, 3])
            y = layers.Conv2D(4, 3, padding="same")(x)
            _ = tf.reduce_sum(y).numpy()
        print("[resume_pretrain] GPU probe OK.")
        return True
    except Exception as e:
        print("[resume_pretrain] GPU probe FAILED:", repr(e))
        return False


def decide_device(policy: str) -> str:
    """
    Decide TF device from policy ('cpu'|'gpu'|'auto' with probing).
    """
    if policy == "cpu":
        print("[resume_pretrain] --device cpu -> /CPU:0")
        return "/CPU:0"
    _enable_mem_growth()
    if policy == "gpu":
        print("[resume_pretrain] --device gpu requested; /GPU:0 (may error).")
        return "/GPU:0"
    return "/GPU:0" if _gpu_probe() else "/CPU:0"


# --------------------------- data (two views) ---------------------------------
def color_jitter(x: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    """
    Light color jitter in TF: brightness, contrast, saturation.

    Parameters
    ----------
    x : tf.Tensor [H,W,3], float32 in [0,1]
    s : float
        Strength multiplier (0..1). 0.5 is modest.

    Returns
    -------
    tf.Tensor
        Jittered image clipped back to [0,1].
    """
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    return tf.clip_by_value(x, 0.0, 1.0)


def random_augment(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """
    Basic SSL-style spatial + color augmentation.

    Steps
    -----
    - Convert to float32 [0,1]
    - Random crop from padded image (resize_with_crop_or_pad -> random_crop)
    - Random horizontal flip
    - Color jitter (light)

    Returns
    -------
    tf.Tensor [image_size, image_size, 3]
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, image_size + 8, image_size + 8)
    image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image


def two_view_map(image: tf.Tensor, image_size: int):
    """
    Create two independently augmented views of the same image.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        (view1, view2) each [H,W,3] float32
    """
    return random_augment(image, image_size), random_augment(image, image_size)


def build_cifar10(image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Two-view pipeline over CIFAR-10 training set (50k images).

    Returns
    -------
    tf.data.Dataset
        Batches of (x1, x2) augmentations, drop_remainder=True for static shapes.
    """
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_cifar100(image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Two-view pipeline over CIFAR-100 training set (50k images).
    """
    (x_train, _), _ = keras.datasets.cifar100.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


# ------------------- encoder / projector (match trainer) ----------------------
def conv_block(x: tf.Tensor, filters: int, k: int = 3, s: int = 1) -> tf.Tensor:
    """
    Conv2D -> BatchNorm -> ReLU block (kept inside Keras layers for graph safety).
    """
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_encoder(image_size: int) -> keras.Model:
    """
    Tiny CNN encoder with final Dense(2048) named 'feat'.

    Keep names identical to pretraining so weight loading matches by name.
    """
    inp = keras.Input(shape=(image_size, image_size, 3))
    x = conv_block(inp, 64)
    x = conv_block(x, 64, s=2)
    x = conv_block(x, 128)
    x = conv_block(x, 128, s=2)
    x = conv_block(x, 256)
    x = conv_block(x, 256, s=2)
    x = conv_block(x, 512)
    x = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(2048, name="feat")(x)
    return keras.Model(inp, feat, name="encoder")


def build_projector(in_dim: int, out_dim: int, num_layers: int) -> keras.Model:
    """
    MLP projector for VICReg pretraining (rebuilt for weight loading & training).

    Parameters
    ----------
    in_dim : int
    out_dim : int
    num_layers : int (>=1)

    Returns
    -------
    keras.Model named 'projector'
    """
    assert num_layers >= 1
    inp = keras.Input(shape=(in_dim,))
    x = inp
    hidden = max(2048, out_dim)
    for i in range(num_layers - 1):
        x = layers.Dense(hidden, use_bias=False, name=f"proj_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"proj_bn_{i}")(x)
        x = layers.ReLU(name=f"proj_relu_{i}")(x)
    out = layers.Dense(out_dim, name="proj_out")(x)
    return keras.Model(inp, out, name="projector")


# ------------------------------ VICReg losses ---------------------------------
class VICRegWeights:
    """
    Container for weighting the three VICReg terms.

    Attributes
    ----------
    sim : float
        Weight for invariance (MSE between paired projections).
    var : float
        Weight for variance (keep per-dim std above gamma).
    cov : float
        Weight for covariance (reduce off-diagonal covariance).
    """
    def __init__(self, sim=25.0, var=25.0, cov=1.0):
        self.sim, self.var, self.cov = float(sim), float(var), float(cov)


def invariance_loss(z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
    """
    Invariance: Encourage z1 ~ z2 (same image, different views).

    Returns
    -------
    tf.Tensor (scalar)
        Mean squared error between projections.
    """
    z1 = tf.cast(z1, tf.float32); z2 = tf.cast(z2, tf.float32)
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z: tf.Tensor, gamma: float = 1.0) -> tf.Tensor:
    """
    Variance: Push per-dimension stddev >= gamma (prevents collapse).

    Parameters
    ----------
    z : tf.Tensor [B, D]
    gamma : float
        Minimum desired std per dimension.

    Returns
    -------
    tf.Tensor (scalar)
        Mean ReLU(gamma - std) across dimensions.
    """
    z = tf.cast(z, tf.float32)
    std = tf.math.reduce_std(z, axis=0)
    return tf.reduce_mean(tf.nn.relu(float(gamma) - std))


def covariance_loss(z: tf.Tensor, nu: float = 0.0) -> tf.Tensor:
    """
    Covariance: Reduce off-diagonal covariance (promotes decorrelation).

    Parameters
    ----------
    z : tf.Tensor [B, D]
    nu : float
        Optional diagonal target; usually 0.0.

    Returns
    -------
    tf.Tensor (scalar)
        Mean squared off-diagonal covariance (centered).
    """
    z = tf.cast(z, tf.float32)
    z = z - tf.reduce_mean(z, axis=0, keepdims=True)
    n = tf.cast(tf.shape(z)[0], tf.float32)
    cov = (tf.transpose(z) @ z) / (n - 1.0)
    d = tf.linalg.tensor_diag_part(cov)
    off = cov - tf.linalg.diag(d)
    return tf.reduce_mean(tf.square(off - float(nu)))


def vicreg_total(z1: tf.Tensor, z2: tf.Tensor, w: VICRegWeights, gamma: float, nu: float):
    """
    Combine the three VICReg terms into a total loss.

    Returns
    -------
    total : tf.Tensor (scalar)
    logs : dict[str, tf.Tensor]
        Individual components for logging.
    """
    inv = invariance_loss(z1, z2)
    var = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov = covariance_loss(z1, nu) + covariance_loss(z2, nu)
    total = w.sim * inv + w.var * var + w.cov * cov
    return total, {"inv": inv, "var": var, "cov": cov, "total": total}


# --------------------------- schedules (pure Python) --------------------------
def cosine_schedule(start: float, end: float, t: float) -> float:
    """
    Cosine interpolation between `start` and `end` for t in [0,1].

    Returns
    -------
    float
        Interpolated value. Clamps t outside [0,1].
    """
    t = float(max(0.0, min(1.0, t)))
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t)))


class AdaptiveTargets:
    """
    Produce adaptive targets gamma_t and nu_t over training progress
