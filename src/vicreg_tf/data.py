"""tf.data pipelines that yield two augmented views of each CIFAR image."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras

from .augment import two_view_map

AUTOTUNE = tf.data.AUTOTUNE

CIFAR_TRAIN_SIZE = 50_000
_CIFAR10_NAMES = {"cifar10", "cifar-10"}
_CIFAR100_NAMES = {"cifar100", "cifar-100"}


def steps_for_dataset(name: str, batch_size: int) -> tuple[int, int]:
    """Return (num_samples, steps_per_epoch). CIFAR-10 and CIFAR-100 both have 50,000 training images."""
    if name.lower() not in _CIFAR10_NAMES | _CIFAR100_NAMES:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    return CIFAR_TRAIN_SIZE, max(1, CIFAR_TRAIN_SIZE // batch_size)


def _two_view_pipeline(images, image_size: int, batch_size: int) -> tf.data.Dataset:
    """Infinite pipeline of (view1, view2) float32 batches in [0, 1]."""
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds.repeat()


def build_cifar10(image_size: int, batch_size: int) -> tf.data.Dataset:
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    return _two_view_pipeline(x_train, image_size, batch_size)


def build_cifar100(image_size: int, batch_size: int) -> tf.data.Dataset:
    (x_train, _), _ = keras.datasets.cifar100.load_data()
    return _two_view_pipeline(x_train, image_size, batch_size)


def take_probe_batch(ds: tf.data.Dataset, size: Optional[int] = None) -> Optional[tf.Tensor]:
    """First view of one batch from a two-view dataset, for embedding statistics only.

    Returns None if the dataset yields nothing.
    """
    try:
        batch = next(iter(ds))
    except Exception:
        return None
    x = batch[0] if isinstance(batch, (tuple, list)) and len(batch) >= 1 else batch
    return x if size is None else x[: int(size)]


def build_dataset(name: str, image_size: int, batch_size: int) -> tf.data.Dataset:
    """Build the pipeline for 'cifar10' or 'cifar100' (a hyphen is also accepted, e.g. 'cifar-10')."""
    name = name.lower()
    if name in _CIFAR10_NAMES:
        return build_cifar10(image_size, batch_size)
    if name in _CIFAR100_NAMES:
        return build_cifar100(image_size, batch_size)
    raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
