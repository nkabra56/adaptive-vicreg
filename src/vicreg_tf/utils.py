"""
Shared helpers for the training and evaluation scripts: seeding, mixed
precision, GPU visibility/memory growth, a GPU kernel probe, forcing variable
creation on subclassed models, and a resilient checkpoint loader.
"""

from __future__ import annotations

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras


def set_global_seed(seed: int) -> None:
    """
    Seed Python's `random`, NumPy, and TF so a run is reproducible.

    `tf.data` shuffle ops with `reshuffle_each_iteration=True` (used in
    `data.py`) draw from the global TF seed, so this needs to run before the
    dataset/model are built.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[utils] Global seed set to {seed}")


def set_mixed_precision(enable: bool) -> None:
    """
    Set the global Keras precision policy: "mixed_bfloat16" if `enable`,
    else "float32". Defaults to float32 for reproducibility; mixed precision
    can introduce numerical differences on some models or environments.
    """
    try:
        from tensorflow.keras import mixed_precision as mp
    except Exception:
        mp = None

    if mp is None:
        print("[utils] Mixed precision is not available in this TF build.")
        return

    mp.set_global_policy("mixed_bfloat16" if enable else "float32")
    print(f"[utils] Global policy set to: {mp.global_policy()}")


def print_devices() -> None:
    """Print visible physical GPUs, for a quick sanity check in a new environment."""
    print("[utils] Visible GPUs:", tf.config.list_physical_devices("GPU"))


def enable_memory_growth() -> None:
    """
    Enable per-GPU memory growth so TF allocates VRAM on demand instead of
    grabbing most of it at startup. Failures (driver quirks, permissions) are
    logged as warnings, not raised.
    """
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("[utils] set_memory_growth warning:", repr(e))


def gpu_probe_ok() -> bool:
    """
    Run a tiny Conv2D on /GPU:0 to confirm CUDA/cuDNN actually work, not just
    that TF can see a GPU device. Returns False (and logs the error) on any
    failure.
    """
    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform([1, 16, 16, 3])
            y = tf.keras.layers.Conv2D(4, 3, padding="same")(x)
            _ = tf.reduce_sum(y).numpy()

        print("[utils] GPU probe OK.")
        return True
    except Exception as e:
        print("[utils] GPU probe FAILED:", repr(e))
        return False


def force_build_for_saving(
    trainer: keras.Model, encoder: keras.Model, projector: keras.Model, image_size: int
) -> None:
    """
    Run the encoder and projector on dummy inputs to materialize their
    variables, then mark `trainer.built = True`. Subclassed Keras models
    don't create variables until first called, so this is needed before
    `save_weights` can be used right after construction.
    """
    _ = encoder(tf.zeros([1, image_size, image_size, 3]), training=False)
    _ = projector(tf.zeros([1, encoder.output_shape[-1]]), training=False)

    trainer.built = True
    print("[utils] Forced variable creation; trainer.built = True")


def safe_load_trainer_weights(trainer: keras.Model, ckpt_path: str) -> None:
    """
    Load weights into `trainer`, trying an exact structural match first and
    falling back to `by_name=True, skip_mismatch=True` if that fails (e.g.
    after a small architecture change), so compatible variables still load.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[utils] Loading weights: {ckpt_path}")
    try:
        trainer.load_weights(ckpt_path)
        print("[utils] Weights loaded (full structural match).")
        return
    except Exception as e:
        print("[utils] Exact load failed; trying partial by_name:", repr(e))

    trainer.load_weights(ckpt_path, by_name=True, skip_mismatch=True)
    print("[utils] Weights restored by_name with skip_mismatch=True.")
