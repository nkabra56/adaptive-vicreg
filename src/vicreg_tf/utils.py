"""Seeding, precision, device selection and checkpoint helpers shared by the scripts."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

DEVICE_CHOICES = ("auto", "gpu", "cpu")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TF. Call before building the dataset: `tf.data` shuffling uses the global TF seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[utils] Global seed set to {seed}")


def set_mixed_precision(enable: bool) -> None:
    """Set the global precision policy to mixed_bfloat16 or float32."""
    keras.mixed_precision.set_global_policy("mixed_bfloat16" if enable else "float32")
    print(f"[utils] Global policy set to: {keras.mixed_precision.global_policy()}")


def print_devices() -> None:
    print("[utils] Visible GPUs:", tf.config.list_physical_devices("GPU"))


def enable_memory_growth() -> None:
    """Let TF allocate GPU memory on demand. Failures are logged, not raised."""
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("[utils] set_memory_growth warning:", repr(e))


def gpu_probe_ok() -> bool:
    """Run a small Conv2D on /GPU:0 to check that CUDA and cuDNN work, not just that a GPU is visible."""
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


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="'auto' probes the GPU and falls back to CPU; 'gpu' and 'cpu' force a device.",
    )


def preparse_device() -> str:
    """Read --device from sys.argv and hide the GPUs for `cpu`.

    Scripts call this at import time, before anything initializes CUDA.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_device_arg(parser)
    args, _ = parser.parse_known_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return args.device


def select_device(flag: str, tag: str) -> str:
    """Turn a --device value into a TF device string, probing the GPU when `flag` is 'auto'."""
    if flag == "cpu":
        print(f"[{tag}] Forcing CPU mode per flag.")
        return "/CPU:0"

    print_devices()
    enable_memory_growth()

    if flag == "gpu":
        print(f"[{tag}] Requested GPU; will not fall back.")
        return "/GPU:0"

    if gpu_probe_ok():
        return "/GPU:0"
    print(f"[{tag}] GPU probe failed; falling back to CPU.")
    return "/CPU:0"


def force_build_for_saving(
    trainer: keras.Model, encoder: keras.Model, projector: keras.Model, image_size: int
) -> None:
    """Create the encoder and projector variables with dummy inputs so `save_weights` works right after construction."""
    _ = encoder(tf.zeros([1, image_size, image_size, 3]), training=False)
    _ = projector(tf.zeros([1, encoder.output_shape[-1]]), training=False)

    trainer.built = True
    print("[utils] Forced variable creation; trainer.built = True")


def safe_load_trainer_weights(trainer: keras.Model, ckpt_path: str) -> None:
    """Load `ckpt_path` into `trainer`, falling back to `by_name` with `skip_mismatch` if the structure differs."""
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
