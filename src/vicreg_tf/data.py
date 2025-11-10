"""
Dataset builders for self-supervised and supervised pipelines.

Provides:
  - TwoViewsDataset: wraps an unlabeled stream and emits ((view1, view2), 0)
  - CIFAR-10 / CIFAR-100 self-supervised and supervised builders
  - STL-10 builders via TFDS
  - Folder-based builders for custom datasets

Author: Nishant Kabra
Date: 11/8/2025
"""
from typing import Tuple, Optional
import tensorflow as tf

# Import graph-safe augmentations and channel standardization
from .augment import make_two_views, _standardize_channels as standardize

AUTOTUNE = tf.data.AUTOTUNE


class TwoViewsDataset:
    """
    Wrap an unlabeled image dataset and produce paired augmentations per example.

    Each output sample has the form:
        ((view1, view2), 0)
    where the label is a dummy zero to satisfy Keras' (inputs, labels) signature.
    """
    def __init__(self, ds_images: tf.data.Dataset, image_size: int):
        """
        Args:
            ds_images: dataset of single images with shape [H, W, 3], dtype uint8 or float
            image_size: target resize dimension (square)
        """
        self.ds_images = ds_images
        self.image_size = image_size

    def build(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create a batched, prefetched pipeline of two-view augmentations.

        Args:
            batch_size: global batch size
            shuffle: whether to shuffle the dataset

        Returns:
            tf.data.Dataset yielding ((view1, view2), 0) batched tensors.
        """
        def _map(img):
            v1, v2 = make_two_views(img, self.image_size)  # graph-safe
            return (v1, v2), tf.constant(0, dtype=tf.int32)

        ds = self.ds_images
        if shuffle:
            ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)
        return ds


# ---------------------------
# CIFAR-10 / CIFAR-100 (Keras)
# ---------------------------

def _cifar_source(name: str) -> tf.data.Dataset:
    if name == "cifar10":
        (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    elif name == "cifar100":
        (x_train, _), _ = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    else:
        raise ValueError(f"Unknown CIFAR name: {name}")
    # x_train: uint8 [N, 32, 32, 3]
    return tf.data.Dataset.from_tensor_slices(x_train)


def build_cifar10_selfsup(image_size: int, batch_size: int) -> tf.data.Dataset:
    ds = _cifar_source("cifar10")
    return TwoViewsDataset(ds, image_size).build(batch_size, shuffle=True)


def build_cifar100_selfsup(image_size: int, batch_size: int) -> tf.data.Dataset:
    ds = _cifar_source("cifar100")
    return TwoViewsDataset(ds, image_size).build(batch_size, shuffle=True)


def build_cifar10_supervised(image_size: int, batch_size: int, split: str = "train") -> tf.data.Dataset:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x, y = (x_train, y_train) if split == "train" else (x_test, y_test)

    def _map(img, label):
        img = tf.image.convert_image_dtype(img, tf.float32)       # [0,1]
        img = tf.image.resize(img, [image_size, image_size], antialias=True)
        img = standardize(img)                                     # channel-wise
        return img, tf.cast(tf.squeeze(label, axis=-1), tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if split == "train":
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_cifar100_supervised(image_size: int, batch_size: int, split: str = "train") -> tf.data.Dataset:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    x, y = (x_train, y_train) if split == "train" else (x_test, y_test)

    def _map(img, label):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [image_size, image_size], antialias=True)
        img = standardize(img)
        return img, tf.cast(tf.squeeze(label, axis=-1), tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if split == "train":
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return ds


# -------------
# STL-10 (TFDS)
# -------------

def build_stl10_selfsup(image_size: int, batch_size: int,
                        split: str = "unlabeled",
                        tfds_data_dir: Optional[str] = None) -> tf.data.Dataset:
    """
    Self-supervised STL-10 builder using TFDS.

    split: 'train', 'test', or 'unlabeled'. For self-supervised training we
           typically use 'unlabeled'.
    """
    import tensorflow_datasets as tfds
    ds = tfds.load("stl10", split=split, data_dir=tfds_data_dir, as_supervised=False)
    # ds elements are dicts with key 'image' (and 'label' for train/test)
    ds_img = ds.map(lambda el: el["image"], num_parallel_calls=AUTOTUNE)
    return TwoViewsDataset(ds_img, image_size).build(batch_size, shuffle=True)


def build_stl10_supervised(image_size: int, batch_size: int, split: str,
                           tfds_data_dir: Optional[str] = None) -> tf.data.Dataset:
    import tensorflow_datasets as tfds
    if split not in ("train", "test"):
        raise ValueError("STL-10 supervised split must be 'train' or 'test'")
    ds = tfds.load("stl10", split=split, data_dir=tfds_data_dir, as_supervised=True)

    def _map(img, label):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [image_size, image_size], antialias=True)
        img = standardize(img)
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ----------------
# Folder datasets
# ----------------

def build_folder_selfsup(data_root: str, image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Build a self-supervised pipeline from a folder-structured dataset.

    Uses Keras directory loader, then drops labels and wraps in TwoViewsDataset.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=None,                # return single images
        shuffle=True,
    )
    ds_img = ds.map(lambda img, lbl: img, num_parallel_calls=AUTOTUNE)
    return TwoViewsDataset(ds_img, image_size).build(batch_size, shuffle=True)


def build_folder_supervised(data_root: str, image_size: int, batch_size: int, split: str = "train") -> tf.data.Dataset:
    """
    Supervised folder dataset with ImageNet-style standardization.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=None,                # single images for consistent mapping
        shuffle=(split == "train"),
    )

    def _map(img, label):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = standardize(img)
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return ds
