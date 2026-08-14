"""
CIFAR-10/100 tf.data pipelines that yield two augmented views per image, for
VICReg-style self-supervised pretraining.
"""

from __future__ import annotations
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from .augment import two_view_map

AUTOTUNE = tf.data.AUTOTUNE


def steps_for_dataset(name: str, batch_size: int) -> Tuple[int, int]:
    """
    Return (num_samples, steps_per_epoch) for a CIFAR dataset. Both CIFAR-10
    and CIFAR-100 have 50,000 training images.
    """
    n = 50_000 if name.lower() in {"cifar10", "cifar-10", "cifar100", "cifar-100"} else None
    if n is None:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    return n, max(1, n // batch_size)


def build_cifar10(image_size: int, batch_size: int) -> tf.data.Dataset:
    """Infinite two-view pipeline over the CIFAR-10 train split, yielding (view1, view2) float32 batches in [0, 1]."""
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds.repeat()


def build_cifar100(image_size: int, batch_size: int) -> tf.data.Dataset:
    """Infinite two-view pipeline over the CIFAR-100 train split, yielding (view1, view2) float32 batches in [0, 1]."""
    (x_train, _), _ = keras.datasets.cifar100.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds.repeat()


def build_dataset(name: str, image_size: int, batch_size: int) -> tf.data.Dataset:
    """Route to `build_cifar10`/`build_cifar100` by name ('cifar10'/'cifar-10' or 'cifar100'/'cifar-100')."""
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        return build_cifar10(image_size, batch_size)
    if name in {"cifar100", "cifar-100"}:
        return build_cifar100(image_size, batch_size)
    raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
