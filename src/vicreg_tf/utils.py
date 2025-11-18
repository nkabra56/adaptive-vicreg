"""
Module: utils.py — general helpers I share across training and evaluation

Purpose
-------
I keep small, reusable utilities here that make my training and evaluation
scripts cleaner and more robust:
  • Mixed precision toggling (bfloat16 vs float32)
  • Quick visibility of available GPUs
  • Safe, per-GPU memory growth so TF does not pre-allocate all VRAM
  • A tiny GPU kernel "probe" to verify CUDA/cuDNN are actually working
  • A helper to force variable creation so `save_weights` works for subclassed models
  • A resilient weight loader that gracefully falls back to name-based loading

Example usage
-------------
from vicreg_tf.utils import (
    set_mixed_precision, print_devices, enable_memory_growth, gpu_probe_ok,
    force_build_for_saving, safe_load_trainer_weights
)

# I prefer to keep training stable in float32 unless I know bfloat16 helps.
set_mixed_precision(False)

# Sanity check what TF sees:
print_devices()
enable_memory_growth()
_ = gpu_probe_ok()  # prints result and returns True/False

# Before saving a subclassed model, I make sure variables exist:
force_build_for_saving(trainer, encoder, projector, image_size=32)

# Loading weights with a friendly fallback:
safe_load_trainer_weights(trainer, "checkpoints_tf/run_XXX/vicreg_full.weights.h5")

Author: Nishant Kabra
Date: 11/17/2025
"""

from __future__ import annotations

import os
import tensorflow as tf
from tensorflow import keras


def set_mixed_precision(enable: bool) -> None:
    """
    Toggle global mixed precision (bfloat16) for the current TF process.

    Why I need this
    ---------------
    Mixed precision can speed up training on modern accelerators, but for some
    models or environments it can introduce numerical differences. I default to
    float32 for reproducibility and only enable mixed precision when I want it.

    Args:
        enable: If True, I set the global policy to "mixed_bfloat16".
                If False, I force the safer "float32" policy.

    Returns:
        None. I print the active policy so I can see what happened.

    Notes
    -----
    • I import `tensorflow.keras.mixed_precision` inside the function to keep
      imports lightweight. If that module is unavailable (older TF builds),
      I print a friendly message and do nothing.
    """
    try:
        # Import inside the function so failures here don't break imports elsewhere.
        from tensorflow.keras import mixed_precision as mp
    except Exception:
        mp = None

    if mp is None:
        # Mixed precision is optional; I do not treat it as fatal.
        print("[utils] Mixed precision is not available in this TF build.")
        return

    # Set the policy to bfloat16 or float32 depending on the flag.
    mp.set_global_policy("mixed_bfloat16" if enable else "float32")
    print(f"[utils] Global policy set to: {mp.global_policy()}")


def print_devices() -> None:
    """
    Print visible physical GPUs for a quick sanity check.

    Why I call this
    ---------------
    When I jump into a new container or VM, I want to confirm TF can actually
    see the GPU devices before I start training.

    Returns:
        None. I simply print the list of physical GPU devices.
    """
    # Query TF for visible GPU devices (this does not run a kernel).
    print("[utils] Visible GPUs:", tf.config.list_physical_devices("GPU"))


def enable_memory_growth() -> None:
    """
    Enable per-GPU memory growth to avoid grabbing all VRAM up front.

    Why I do this
    -------------
    By default, TF tries to allocate most of the GPU memory on startup. That
    can cause OOMs or starve other processes. Setting memory growth makes TF
    allocate memory on demand.

    Behavior
    --------
    • I iterate over all visible GPUs and enable memory growth on each.
    • If anything fails (driver quirk, permission, etc.), I print a warning
      and keep going.

    Returns:
        None.
    """
    try:
        # Iterate over every visible physical GPU device.
        for gpu in tf.config.list_physical_devices("GPU"):
            # Use experimental API for per-device memory growth.
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        # I don't want this to be fatal; just warn and continue.
        print("[utils] set_memory_growth warning:", repr(e))


def gpu_probe_ok() -> bool:
    """
    Run a tiny Conv2D on /GPU:0 to validate kernels and drivers.

    What this catches
    -----------------
    • Misconfigured CUDA/cuDNN or incompatible TF builds
    • Cases where TF can "see" a GPU but fails to execute any kernel on it

    Returns:
        True  — if I successfully execute a small Conv2D on the GPU
        False — if any exception occurs; I also print the error for context
    """
    try:
        # Explicitly place a tiny operation on the first GPU device.
        with tf.device("/GPU:0"):
            # Create a small random input tensor.
            x = tf.random.uniform([1, 16, 16, 3])
            # Run a lightweight conv to force cuDNN/BLAS initialization.
            y = tf.keras.layers.Conv2D(4, 3, padding="same")(x)
            # Reduce to a scalar and move back to host to ensure execution.
            _ = tf.reduce_sum(y).numpy()

        print("[utils] GPU probe OK.")
        return True
    except Exception as e:
        # Any failure indicates kernels did not execute on the GPU.
        print("[utils] GPU probe FAILED:", repr(e))
        return False


def force_build_for_saving(
    trainer: keras.Model, encoder: keras.Model, projector: keras.Model, image_size: int
) -> None:
    """
    Force variable creation so `save_weights` works for subclassed models.

    Why I need this
    ---------------
    When using subclassed Keras models, variables aren't created until the model
    is actually called at least once. I call the models with dummy inputs so I
    can save weights immediately after constructing the training graph.

    Args:
        trainer: The VICRegTrainer (subclassed Model) that wraps encoder+projector.
        encoder: The encoder network.
        projector: The projection head stacked on top of the encoder.
        image_size: The square input size I use to create the dummy input.

    Returns:
        None. I set `trainer.built = True` and print a small status line.
    """
    # Call encoder once with a dummy image to materialize its variables.
    _ = encoder(tf.zeros([1, image_size, image_size, 3]), training=False)
    # Call projector once with a dummy feature vector to create its variables.
    _ = projector(tf.zeros([1, encoder.output_shape[-1]]), training=False)

    # Mark the trainer as "built" so Keras checkpointing APIs are happy.
    trainer.built = True
    print("[utils] Forced variable creation; trainer.built = True")


def safe_load_trainer_weights(trainer: keras.Model, ckpt_path: str) -> None:
    """
    Load weights robustly, falling back to name-based loading if needed.

    Behavior
    --------
    1) I first try an exact structural load (strict shape and name matches).
    2) If that fails due to small architecture changes or shape drift, I retry
       with `by_name=True` and `skip_mismatch=True` so compatible variables
       are restored while incompatible ones are left at their initial values.

    Args:
        trainer: The top-level Keras model that contains submodules to restore.
        ckpt_path: Path to a `.weights.h5` file on disk.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.

    Returns:
        None. I print which strategy succeeded.
    """
    if not os.path.exists(ckpt_path):
        # Fast fail: there's nothing to load.
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[utils] Loading weights: {ckpt_path}")
    try:
        # Try a full structural load first (the safest when it works).
        trainer.load_weights(ckpt_path)
        print("[utils] Weights loaded (full structural match).")
        return
    except Exception as e:
        # If shapes or names don't match, I fall back to a more permissive mode.
        print("[utils] Exact load failed; trying partial by_name:", repr(e))

    # Name-based, skip mismatched shapes — useful after small refactors.
    trainer.load_weights(ckpt_path, by_name=True, skip_mismatch=True)
    print("[utils] Weights restored by_name with skip_mismatch=True.")
