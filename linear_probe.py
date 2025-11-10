#!/usr/bin/env python3
"""
Linear probe: freeze the encoder, train a single Dense head on top.

Why:
- Standard, simple way to measure representation quality without full fine-tuning.

Author: Nishant Kabra
Date: 11/8/2025
"""

from __future__ import annotations
import os, argparse
import tensorflow as tf
from src.vicreg_tf import (
    set_seed, enable_mixed_precision,
    build_cifar10_supervised, build_cifar100_supervised,
    build_stl10_supervised, build_folder_supervised
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe evaluation")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10", "folder"])
    p.add_argument("--data-root", type=str, default="./data")   # used only for folder datasets
    p.add_argument("--image-size", type=int, default=224)       # 32 for CIFAR, 96 for STL-10, else 224
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--encoder-path", type=str, default=os.path.join("artifacts", "encoder_savedmodel"))
    p.add_argument("--tfds-data-dir", type=str, default=None)   # optional TFDS cache (STL-10)
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
        train_ds, test_ds, num_classes = build_cifar10_supervised(img_size, args.batch_size)
        image_size = img_size
    elif args.dataset == "cifar100":
        img_size = 32 if args.image_size == 224 else args.image_size
        train_ds, test_ds, num_classes = build_cifar100_supervised(img_size, args.batch_size)
        image_size = img_size
    elif args.dataset == "stl10":
        img_size = 96 if args.image_size == 224 else args.image_size
        train_ds, test_ds, num_classes = build_stl10_supervised(img_size, args.batch_size, data_dir=args.tfds_data_dir)
        image_size = img_size
    else:
        train_ds, test_ds, num_classes = build_folder_supervised(args.data_root, args.image_size, args.batch_size)
        image_size = args.image_size

    # ---------------- Encoder ----------------
    encoder = tf.keras.models.load_model(args.encoder_path, compile=False)  # SavedModel
    encoder.trainable = False                                              # freeze

    # Some users prefer probing pooled features (before projector) if available
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    try:
        feat_extractor = tf.keras.Model(encoder.input, encoder.get_layer("feat_pool").output)
        h = feat_extractor(inputs, training=False)
    except Exception:
        h = encoder(inputs, training=False)

    logits = tf.keras.layers.Dense(num_classes, name="linear_head")(h)     # linear classifier
    clf = tf.keras.Model(inputs, logits, name="linear_probe")

    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    clf.fit(train_ds, epochs=args.epochs, validation_data=test_ds)
    print("Done.")

if __name__ == "__main__":
    main()
