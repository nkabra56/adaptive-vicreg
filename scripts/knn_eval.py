"""
kNN evaluation for a self-supervised encoder on CIFAR.
-------------------------------------------------------------------------------
This script builds a **frozen** feature extractor from a Keras Applications
backbone (e.g., ResNet50V2), optionally loads a checkpoint with `skip_mismatch=True`,
extracts L2-normalized features for the train/test splits, then evaluates
with a non-parametric k-Nearest Neighbors classifier (cosine similarity)
popularized by Wu et al. (2018), SimCLR, and many follow-ups.

Author: Nishant Kabra
Date: 11/10/25
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf
import keras
from keras import mixed_precision
from tensorflow.keras import layers


# ----------------------------- Global configuration -----------------------------
# Enable mixed bfloat16 policy when requested. This is safe on CPU and can
# speed up matmul/conv kernels on newer Intel/AMD CPUs with AVX512 BFloat16.
if os.getenv("MIXED_BF16", "0") == "1":
    mixed_precision.set_global_policy("mixed_bfloat16")
    print("[knn_eval] mixed_bfloat16 enabled")

# Allow overriding the backbone via environment variable for quick experiments.
DEFAULT_BACKBONE = os.getenv("BACKBONE", "resnet50v2").lower()


# ----------------------------- Utility / Data loading ----------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set Python/TensorFlow random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value used by TF and NumPy.
    """
    tf.keras.utils.set_random_seed(seed)


def load_cifar(dataset: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """
    Load CIFAR-10 or CIFAR-100 from Keras datasets.

    Parameters
    ----------
    dataset : {"cifar10", "cifar100"}
        Which dataset to load.

    Returns
    -------
    (x_train, y_train), (x_test, y_test), num_classes : tuple
        Numpy arrays of images/labels and the number of classes.
    """
    dataset = dataset.lower()
    if dataset == "cifar10":
        (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
        ytr, yte = ytr.squeeze(), yte.squeeze()
        num_classes = 10
    elif dataset == "cifar100":
        (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar100.load_data()
        ytr, yte = ytr.squeeze(), yte.squeeze()
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return (xtr, ytr), (xte, yte), num_classes


def _resize_only(x: tf.Tensor, y: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Pure-resize preprocessing used for kNN feature extraction.

    Notes
    -----
    We intentionally avoid heavy augmentation here because the goal is to
    evaluate the representation as-is. You may layer on stronger augments
    if you want to study invariances in kNN space.

    Returns
    -------
    (x, y) with x in [0, 255] float32 resized to (image_size, image_size, 3).
    """
    # Cast is done by the caller, so here we only resize. Using bilinear for speed.
    x = tf.image.resize(x, (image_size, image_size), method="bilinear")
    return x, y


def make_datasets(dataset: str, image_size: int, batch_size: int):
    """
    Create tf.data pipelines for train and test splits.

    We keep preprocessing minimal and leave normalization to the model
    (via the keras.applications preprocess function in a Lambda layer).

    Returns
    -------
    ds_train, ds_test : tf.data.Dataset
    """
    (xtr, ytr), (xte, yte), _ = load_cifar(dataset)
    # Keep uint8 on host memory; cast to float32 on-the-fly to reduce peak RAM.
    xtr = tf.convert_to_tensor(xtr, dtype=tf.uint8)
    xte = tf.convert_to_tensor(xte, dtype=tf.uint8)
    ytr = tf.convert_to_tensor(ytr, dtype=tf.int32)
    yte = tf.convert_to_tensor(yte, dtype=tf.int32)

    ds_train = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    ds_train = ds_train.map(lambda a, b: _resize_only(tf.cast(a, tf.float32), b, image_size),
                            num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((xte, yte))
    ds_test = ds_test.map(lambda a, b: _resize_only(tf.cast(a, tf.float32), b, image_size),
                          num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


# ----------------------------- Model / Feature extractor -------------------------
def build_encoder(backbone: str, image_size: int) -> keras.Model:
    """
    Build a convolutional image encoder (no classification head).

    Parameters
    ----------
    backbone : str
        Name of the Keras Applications backbone. Recommended: "resnet50v2".
    image_size : int
        Input resolution for the encoder.

    Returns
    -------
    keras.Model
        Model mapping (None, image_size, image_size, 3) -> pooled features.
    """
    # Choose a backbone from Keras applications. Add more branches as desired.
    backbone = backbone.lower()
    if backbone == "resnet50v2":
        # We use include_top=False to get convolutional features.
        base = keras.applications.ResNet50V2(include_top=False,
                                             weights=None,  # external checkpoint will be loaded
                                             input_shape=(image_size, image_size, 3))
        preprocess = keras.applications.resnet_v2.preprocess_input
    elif backbone == "resnet50":
        base = keras.applications.ResNet50(include_top=False,
                                           weights=None,
                                           input_shape=(image_size, image_size, 3))
        preprocess = keras.applications.resnet.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: '{backbone}'. Try 'resnet50v2'.")

    # Functional graph: Input -> preprocess -> base -> GAP -> features
    inputs = keras.Input(shape=(image_size, image_size, 3))
    # Preprocess with a Lambda so we don't call TF ops directly on a KerasTensor
    x = layers.Lambda(lambda z: preprocess(z))(inputs)
    x = base(x, training=False)  # features with spatial dims
    x = layers.GlobalAveragePooling2D()(x)  # pooled features [B, C]
    model = keras.Model(inputs, x, name=f"{backbone}_encoder")
    return model


def load_weights_safely(model: keras.Model, ckpt_path: str) -> None:
    """
    Load weights into `model` with `skip_mismatch=True`.

    This survives shape/name differences between the backbone here and a
    self-supervised checkpoint that may contain extra heads (e.g., VICReg MLP).

    Parameters
    ----------
    model : keras.Model
        The model whose weights should be loaded.
    ckpt_path : str
        Path to a `*.h5` / `*.weights.h5` file produced by `model.save_weights`.
    """
    if not ckpt_path:
        print("[knn_eval] No checkpoint provided; using randomly initialized encoder.")
        return
    if not os.path.exists(ckpt_path):
        print(f"[knn_eval] WARNING: checkpoint not found: {ckpt_path}")
        return
    try:
        model.load_weights(ckpt_path, skip_mismatch=True)
        print(f"[knn_eval] loaded weights with skip_mismatch=True from {ckpt_path}")
    except Exception as e:
        print(f"[knn_eval] WARNING: failed to load weights from {ckpt_path}: {e}")


def l2_normalize_layer() -> layers.Layer:
    """
    Return a layer that L2-normalizes features along the last axis.

    Normalization is standard for cosine kNN.
    """
    return layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="l2_norm")


def build_feature_extractor(ckpt_path: str, image_size: int, backbone: str) -> keras.Model:
    """
    Construct the frozen feature extractor, load checkpoint, and append L2 norm.

    Returns
    -------
    keras.Model
        Maps images to L2-normalized features.
    """
    enc = build_encoder(backbone, image_size)
    load_weights_safely(enc, ckpt_path)
    enc.trainable = False  # kNN uses frozen features

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = enc(inputs, training=False)
    x = l2_normalize_layer()(x)
    return keras.Model(inputs, x, name="feature_extractor")


# ----------------------------- kNN evaluation ------------------------------------
def extract_features(model: keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the dataset through `model` and collect features/labels as NumPy.

    Parameters
    ----------
    model : keras.Model
        The feature extractor (expects float32 images in [0,255] or preprocessed).
    ds : tf.data.Dataset
        Batched dataset yielding (images, labels).

    Returns
    -------
    feats : np.ndarray of shape [N, D]
    labels : np.ndarray of shape [N]
    """
    feats_list, labels_list = [], []
    for batch_x, batch_y in ds:
        # Ensure float32 dtype to match the model's expected input
        batch_x = tf.cast(batch_x, tf.float32)
        f = model(batch_x, training=False)
        # If mixed precision, cast back to float32 before moving to host memory
        f = tf.cast(f, tf.float32)
        feats_list.append(f.numpy())
        labels_list.append(batch_y.numpy())
    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return feats, labels


def knn_predict(train_feats: np.ndarray,
                train_labels: np.ndarray,
                test_feats: np.ndarray,
                k: int = 200,
                T: float = 0.07) -> np.ndarray:
    """
    Perform kNN classification with cosine similarity and temperature scaling.

    Parameters
    ----------
    train_feats : array [N_train, D] (L2-normalized)
    train_labels : array [N_train]
    test_feats : array [N_test, D] (L2-normalized)
    k : int
        Number of nearest neighbors.
    T : float
        Softmax temperature used to weight neighbors.

    Returns
    -------
    pred : array [N_test] of predicted integer labels.
    """
    # Cosine similarity reduces to dot product because of L2 normalization.
    sims = test_feats @ train_feats.T  # [N_test, N_train]

    # Get top-k indices for each test sample
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]  # fast partial sort
    # Gather the top-k similarities and labels
    topk_sims = np.take_along_axis(sims, idx, axis=1)     # [N_test, k]
    topk_labels = train_labels[idx]                       # [N_test, k]

    # Convert similarities into weights via temperature-scaled softmax
    # subtract max for numerical stability
    topk_sims = topk_sims - topk_sims.max(axis=1, keepdims=True)
    weights = np.exp(topk_sims / max(T, 1e-6))
    # Aggregate votes per class using the weights
    num_classes = int(train_labels.max()) + 1
    votes = np.zeros((test_feats.shape[0], num_classes), dtype=np.float32)
    for i in range(k):
        lab = topk_labels[:, i]
        np.add.at(votes, (np.arange(votes.shape[0]), lab), weights[:, i])

    # Final prediction is the class with the highest weighted vote
    return votes.argmax(axis=1)


# ----------------------------- Main ----------------------------------------------
def main() -> None:
    """
    CLI entry point.

    Examples
    --------
    Python:
        python scripts/knn_eval.py --dataset cifar10 --image-size 224 --batch-size 512 --k 200 \
            --ckpt checkpoints_tf/vicreg_tf.weights.h5 --backbone resnet50v2
    """
    parser = argparse.ArgumentParser(description="kNN eval for SSL features (cosine kNN).")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"], help="Dataset to evaluate on.")
    parser.add_argument("--image-size", type=int, default=32, help="Input resolution.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for feature extraction.")
    parser.add_argument("--k", type=int, default=200, help="k for kNN.")
    parser.add_argument("--T", type=float, default=0.07, help="Temperature for soft voting.")
    parser.add_argument("--ckpt", type=str, default="", help="Path to encoder weights (save_weights format).")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        help="Backbone name (e.g., resnet50v2).")
    args = parser.parse_args()

    set_seed(42)

    print(f"[knn_eval] dataset={args.dataset} img={args.image_size} bs={args.batch_size} "
          f"k={args.k} T={args.T} ckpt={args.ckpt}")
    ds_train, ds_test = make_datasets(args.dataset, args.image_size, args.batch_size)

    # Build extractor and load checkpoint
    feat_extractor = build_feature_extractor(args.ckpt, args.image_size, args.backbone)

    # Extract features as NumPy arrays
    train_feats, train_labels = extract_features(feat_extractor, ds_train)
    test_feats, test_labels = extract_features(feat_extractor, ds_test)

    # Predict with kNN
    pred = knn_predict(train_feats, train_labels, test_feats, k=args.k, T=args.T)

    # Report accuracy
    acc = (pred == test_labels).mean().item()
    print(f"Embedded: train {len(train_labels)} / test {len(test_labels)}")
    print(f"kNN@{args.k} accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
