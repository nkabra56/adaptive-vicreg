#!/usr/bin/env python3
"""
k-NN evaluation (cosine similarity) on frozen features.

Steps:
 1) Load supervised splits (train/test).
 2) Load exported encoder; extract pooled features (or projector if needed).
 3) L2-normalize features; compute cosine similarities test-vs-train.
 4) Majority vote among top-k neighbors; report accuracy.

Author: Nishant Kabra
Date: 11/8/2025
"""

from __future__ import annotations
import os, argparse
import numpy as np
import tensorflow as tf
from src.vicreg_tf import (
    set_seed, enable_mixed_precision,
    build_cifar10_supervised, build_cifar100_supervised,
    build_stl10_supervised, build_folder_supervised
)

def _extract_features(model: tf.keras.Model, ds: tf.data.Dataset):
    """Run model on ds and collect features/labels into NumPy arrays."""
    feats, labels = [], []
    for x, y in ds:
        z = model(x, training=False).numpy()  # [B, D]
        feats.append(z)
        labels.append(y.numpy())
    return np.concatenate(feats, 0), np.concatenate(labels, 0)

def _majority_vote(labels_row: np.ndarray) -> int:
    """Majority class id among integers in `labels_row`."""
    max_label = int(labels_row.max()) if labels_row.size else 0
    counts = np.bincount(labels_row.astype(int), minlength=max_label + 1)
    return int(counts.argmax())

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="k-NN evaluation (cosine)")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10", "folder"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--encoder-path", type=str, default=os.path.join("artifacts", "encoder_savedmodel"))
    p.add_argument("--tfds-data-dir", type=str, default=None)
    p.add_argument("--mixed", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    enable_mixed_precision(args.mixed)

    # ---------------- Data ----------------
    if args.dataset == "cifar10":
        img_size = 32 if args.image_size == 224 else args.image_size
        train_ds, test_ds, _ = build_cifar10_supervised(img_size, args.batch_size)
        image_size = img_size
    elif args.dataset == "cifar100":
        img_size = 32 if args.image_size == 224 else args.image_size
        train_ds, test_ds, _ = build_cifar100_supervised(img_size, args.batch_size)
        image_size = img_size
    elif args.dataset == "stl10":
        img_size = 96 if args.image_size == 224 else args.image_size
        train_ds, test_ds, _ = build_stl10_supervised(img_size, args.batch_size, data_dir=args.tfds_data_dir)
        image_size = img_size
    else:
        train_ds, test_ds, _ = build_folder_supervised(args.data_root, args.image_size, args.batch_size)
        image_size = args.image_size

    # ---------------- Encoder ----------------
    encoder = tf.keras.models.load_model(args.encoder_path, compile=False)
    encoder.trainable = False

    # Prefer pooled features if available (layer name 'feat_pool')
    try:
        feat_model = tf.keras.Model(encoder.input, encoder.get_layer("feat_pool").output)
    except Exception:
        feat_model = encoder

    # ---------------- Feature extraction ----------------
    Xtr, Ytr = _extract_features(feat_model, train_ds)  # train features/labels
    Xte, Yte = _extract_features(feat_model, test_ds)   # test features/labels

    # L2-normalize so dot products are cosines
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    Xte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)

    # Cosine similarity matrix [Nte, Ntr]
    sims = Xte @ Xtr.T

    # Partial sort to get indices of top-k similar train samples for each test sample
    topk_idx = np.argpartition(-sims, kth=args.k, axis=1)[:, :args.k]
    topk_labels = Ytr[topk_idx]                             # [Nte, k]

    # Majority vote over the k labels per test example
    preds = np.apply_along_axis(_majority_vote, 1, topk_labels)
    acc = (preds.flatten() == Yte.flatten()).mean()
    print(f"k-NN (k={args.k}) accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
