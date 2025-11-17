"""
General helpers shared by training and evaluation scripts.
"""

from __future__ import annotations
import os
import tensorflow as tf
from tensorflow import keras


def set_mixed_precision(enable: bool) -> None:
    """
    Optionally enable mixed_bfloat16. Default is float32 for stability.

    Args:
        enable: True to set mixed_bfloat16, else float32.
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
    """Print visible physical GPU devices for quick sanity checking."""
    print("[utils] Visible GPUs:", tf.config.list_physical_devices("GPU"))


def enable_memory_growth() -> None:
    """Enable per-GPU memory growth to avoid grabbing all VRAM up front."""
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("[utils] set_memory_growth warning:", repr(e))


def gpu_probe_ok() -> bool:
    """
    Run a tiny Conv2D on /GPU:0 to validate kernels and drivers.

    Returns:
        True if a kernel executed on GPU, False otherwise.
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
    Make sure variables exist so `save_weights` works on subclassed models.

    Args:
        trainer: VICRegTrainer instance.
        encoder: Encoder model.
        projector: Projector model.
        image_size: Input image size used during build.
    """
    _ = encoder(tf.zeros([1, image_size, image_size, 3]), training=False)
    _ = projector(tf.zeros([1, encoder.output_shape[-1]]), training=False)
    trainer.built = True  # mark as built for Keras Checkpoint
    print("[utils] Forced variable creation; trainer.built = True")


def safe_load_trainer_weights(trainer: keras.Model, ckpt_path: str) -> None:
    """
    Robust loader that first tries full structural match, then falls back.

    Args:
        trainer: Model wrapper that holds encoder and projector.
        ckpt_path: Path to `.weights.h5`.

    Behavior:
        - Exact load first.
        - If shape/name mismatches occur, try by_name with skip_mismatch=True.
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
