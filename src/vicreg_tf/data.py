"""
Dataset builders for CIFAR-10/100 with two-view augmentation.

What this module does
---------------------
I build tf.data input pipelines that yield *pairs* of augmented views for each
image, exactly what VICReg-style training expects. The pipelines are:
- repeat()'ed to be effectively infinite,
- shuffled each epoch,
- mapped with my `two_view_map` augmentation,
- batched, and
- prefetched for overlap with model compute.

How I use it
------------
- `build_dataset(name, image_size, batch_size)` is my single entry point.
- For CIFAR-10 I call `build_cifar10`, and for CIFAR-100 I call `build_cifar100`.
- `steps_for_dataset` tells me the dataset size and steps/epoch for a given
  batch size (I keep the math centralized to avoid mistakes elsewhere).

Example
-------
>>> # Pretraining: two-view pipeline for VICReg
>>> ds = build_dataset("cifar10", image_size=32, batch_size=256)
>>> # I usually couple this with steps_for_dataset to cap each epoch
>>> num_samples, steps = steps_for_dataset("cifar10", 256)
>>> model.fit(ds, steps_per_epoch=steps, epochs=100)

Author: Nishant Kabra
Date: 11/17/2025
"""

from __future__ import annotations
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from .augment import two_view_map

# I use AUTOTUNE so the tf runtime can pick good parallelism/prefetch values.
AUTOTUNE = tf.data.AUTOTUNE


def steps_for_dataset(name: str, batch_size: int) -> Tuple[int, int]:
    """
    Compute number of samples and steps per epoch for a dataset.

    Why I wrote this
    ----------------
    I prefer one authoritative place to compute epoch length so training runs
    stay consistent. For CIFAR I hardcode the train size (50k) and derive steps.

    Parameters
    ----------
    name : str
        Dataset identifier. I accept 'cifar10' or 'cifar100' (case-insensitive).
    batch_size : int
        Global batch size for training.

    Returns
    -------
    (int, int)
        A tuple `(num_samples, steps_per_epoch)` for the requested dataset.

    Raises
    ------
    ValueError
        If `name` is not a supported CIFAR dataset.

    Notes
    -----
    - I integer-divide to get steps/epoch and guard against zero by using max(1, ...).
    - If I later add datasets with different sizes, I can extend this function.
    """
    # CIFAR-10 and CIFAR-100 both have 50,000 training images; support aliases.
    n = 50_000 if name.lower() in {"cifar10", "cifar-10", "cifar100", "cifar-100"} else None
    if n is None:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    # Compute steps per epoch by integer division; ensure at least one step.
    return n, max(1, n // batch_size)


def build_cifar10(image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Two-view pipeline over CIFAR-10 train split.

    Why I wrote this
    ----------------
    For VICReg I need a stream of pairs (x1, x2) where each is a differently
    augmented view of the same image. I construct an efficient tf.data graph
    with shuffle→map(augment)→batch→prefetch→repeat.

    Parameters
    ----------
    image_size : int
        Target crop size for each augmented view (e.g., 32 for CIFAR).
    batch_size : int
        Batch size for training.

    Returns
    -------
    tf.data.Dataset
        An *infinite* dataset that yields `(view1, view2)` float32 batches in [0, 1],
        each of shape `(batch_size, image_size, image_size, 3)`.

    Notes
    -----
    - I only use the train split here (50k images). Eval pipelines live elsewhere.
    - `reshuffle_each_iteration=True` gives me a fresh shuffle every epoch.
    """
    # Load CIFAR-10; I only need the training images for pretraining.
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    # Create a source dataset of images.
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    # Keep a large shuffle buffer to decouple order (10k is a common sweet spot for CIFAR).
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    # Map each image to a pair of independently augmented views.
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    # Batch into (view1, view2) tensors and overlap input with host/device work.
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    # Repeat to provide an effectively infinite stream for model.fit.
    return ds.repeat()


def build_cifar100(image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Two-view pipeline over CIFAR-100 train split.

    Why I wrote this
    ----------------
    Same structure as CIFAR-10, but I point to the CIFAR-100 loader. Keeping
    them separate makes it crystal clear which split I am using.

    Parameters
    ----------
    image_size : int
        Target crop size for each augmented view (e.g., 32 for CIFAR).
    batch_size : int
        Batch size for training.

    Returns
    -------
    tf.data.Dataset
        An *infinite* dataset that yields `(view1, view2)` float32 batches in [0, 1],
        each of shape `(batch_size, image_size, image_size, 3).
    """
    # Load CIFAR-100 training images.
    (x_train, _), _ = keras.datasets.cifar100.load_data()
    # Turn the numpy array into a dataset of images.
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    # Shuffle per epoch to avoid learning order artifacts.
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    # Produce two augmented views per image using my augmentation pipeline.
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    # Batch + prefetch for throughput, then repeat to make it infinite.
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds.repeat()


def build_dataset(name: str, image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Unified factory for CIFAR datasets.

    Why I wrote this
    ----------------
    I like a single entry point in training code. This function routes to the
    right builder based on the dataset name while keeping argument names uniform.

    Parameters
    ----------
    name : str
        'cifar10' or 'cifar100' (I also accept 'cifar-10' and 'cifar-100').
    image_size : int
        Final crop size for each view that my augmentations should produce.
    batch_size : int
        Batch size for the dataset pipeline.

    Returns
    -------
    tf.data.Dataset
        Infinite dataset of `(view1, view2)` float32 batches.

    Raises
    ------
    ValueError
        If an unsupported dataset name is provided.
    """
    # Normalize the name and route to the corresponding dataset builder.
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        return build_cifar10(image_size, batch_size)
    if name in {"cifar100", "cifar-100"}:
        return build_cifar100(image_size, batch_size)
    # If we reach here, the dataset is not supported.
    raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
