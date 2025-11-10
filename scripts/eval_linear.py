"""
Linear probe evaluation on CIFAR using a frozen encoder.
-------------------------------------------------------------------------------
This script builds a convolutional encoder (Keras Applications), optionally
loads a self-supervised checkpoint with `skip_mismatch=True`, runs optional
BatchNorm adaptation on unlabelled training images, then trains a **linear**
softmax classifier on top while keeping the encoder frozen. This approximates
the common "linear eval" protocol used in SimCLR, MoCo, BYOL, VICReg, etc.

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
if os.getenv("MIXED_BF16", "0") == "1":
    mixed_precision.set_global_policy("mixed_bfloat16")
    print("[eval_linear] mixed_bfloat16 enabled")

DEFAULT_BACKBONE = os.getenv("BACKBONE", "resnet50v2").lower()


# ----------------------------- Utils / data -------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Seed Python/TensorFlow RNG for reproducibility.
    """
    tf.keras.utils.set_random_seed(seed)


def load_cifar(dataset: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """
    Load CIFAR-10 or CIFAR-100 and return NumPy arrays plus class count.
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


def _preprocess_for_training(x: tf.Tensor, y: tf.Tensor, image_size: int, aug: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Resize + light augmentation (optional) for the linear probe.

    We keep augmentations mild to avoid leaking too much invariance from heavy
    SSL pipelines; the point is to measure the representation quality.
    """
    x = tf.image.resize(x, (image_size, image_size), method="bilinear")
    if aug:
        # Random flips are cheap and standard for CIFAR
        x = tf.image.random_flip_left_right(x)
    return x, y


def make_datasets(dataset: str, image_size: int, batch_size: int):
    """
    Return (train_ds, val_ds, test_ds) datasets for linear evaluation.

    We use 45k/5k split for train/val from CIFAR train by default.
    """
    (xtr, ytr), (xte, yte), _ = load_cifar(dataset)
    # Split 45k/5k for tuning LR & early stopping
    x_tr, y_tr = xtr[:45000], ytr[:45000]
    x_val, y_val = xtr[45000:], ytr[45000:]

    # Keep uint8 on host, cast per-map to float for smaller memory footprint
    x_tr = tf.convert_to_tensor(x_tr, dtype=tf.uint8)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.uint8)
    x_te = tf.convert_to_tensor(xte, dtype=tf.uint8)
    y_tr = tf.convert_to_tensor(y_tr, dtype=tf.int32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
    y_te = tf.convert_to_tensor(yte, dtype=tf.int32)

    # Training pipeline (with light aug)
    ds_train = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
    ds_train = ds_train.shuffle(10000, reshuffle_each_iteration=True)
    ds_train = ds_train.map(lambda a, b: _preprocess_for_training(tf.cast(a, tf.float32), b, image_size, True),
                            num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation/Test pipelines (no aug)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(lambda a, b: _preprocess_for_training(tf.cast(a, tf.float32), b, image_size, False),
                        num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((x_te, y_te))
    ds_test = ds_test.map(lambda a, b: _preprocess_for_training(tf.cast(a, tf.float32), b, image_size, False),
                          num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test


# ----------------------------- Encoder / Feature model ---------------------------
def _select_backbone(backbone: str, image_size: int):
    """
    Helper to build the Keras Applications backbone and corresponding preprocess fn.
    """
    backbone = backbone.lower()
    if backbone == "resnet50v2":
        base = keras.applications.ResNet50V2(include_top=False,
                                             weights=None,
                                             input_shape=(image_size, image_size, 3))
        preprocess = keras.applications.resnet_v2.preprocess_input
    elif backbone == "resnet50":
        base = keras.applications.ResNet50(include_top=False,
                                           weights=None,
                                           input_shape=(image_size, image_size, 3))
        preprocess = keras.applications.resnet.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: '{backbone}'. Try 'resnet50v2'.")
    return base, preprocess


def build_feature_extractor(ckpt_path: str, image_size: int, backbone: str) -> keras.Model:
    """
    Build encoder (+ GAP) wrapped with preprocessing and load weights if given.

    Returns
    -------
    keras.Model
        A frozen model that maps images -> pooled features (float32).
    """
    base, preprocess = _select_backbone(backbone, image_size)

    # Inputs and preprocessing live in the Keras graph (avoid raw TF ops on KerasTensor)
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Lambda(lambda z: preprocess(z))(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    feat_model = keras.Model(inputs, x, name=f"{backbone}_encoder")

    # Load checkpoint with mismatches skipped (e.g., extra MLP heads in SSL)
    if ckpt_path:
        if os.path.exists(ckpt_path):
            try:
                feat_model.load_weights(ckpt_path, skip_mismatch=True)
                print(f"[eval_linear] loaded weights from {ckpt_path} (skip_mismatch=True)")
            except Exception as e:
                print(f"[eval_linear] WARNING: failed weight load: {e}")
        else:
            print(f"[eval_linear] WARNING: checkpoint not found: {ckpt_path}")

    feat_model.trainable = False  # freeze for linear probe
    return feat_model


def bn_adapt(feature_extractor: keras.Model,
             ds: tf.data.Dataset,
             steps: int = 0) -> None:
    """
    Optional BatchNorm adaptation: run a few forward passes with `training=True`
    to update moving mean/variance on the target data distribution.

    Parameters
    ----------
    feature_extractor : keras.Model
        Encoder model that contains BN layers.
    ds : tf.data.Dataset
        Unlabeled training images (labels unused).
    steps : int
        Number of adaptation batches to run. 0 disables adaptation.
    """
    if steps <= 0:
        return

    print(f"[eval_linear] Running BN adaptation for {steps} steps...")
    it = iter(ds)  # reuse the same ds pipeline
    for i in range(steps):
        try:
            batch_x, _ = next(it)
        except StopIteration:
            # If we exhaust ds before completing, recreate iterator
            it = iter(ds)
            batch_x, _ = next(it)
        # Important: call with training=True so BN updates its moving stats
        _ = feature_extractor(batch_x, training=True)


def build_linear_probe(feature_extractor: keras.Model,
                       num_classes: int,
                       weight_decay: float = 1e-4) -> keras.Model:
    """
    Construct the linear classifier on top of the frozen encoder.

    We use a single Dense layer producing logits. The encoder stays frozen.
    """
    inputs = feature_extractor.inputs[0]
    x = feature_extractor(inputs, training=False)  # pooled features
    # Optionally normalize features to stabilize the head; not strictly required.
    # x = layers.LayerNormalization(epsilon=1e-6, name="feat_ln")(x)

    logits = layers.Dense(num_classes,
                          use_bias=True,
                          kernel_initializer="lecun_normal",
                          bias_initializer="zeros",
                          kernel_regularizer=keras.regularizers.l2(weight_decay),
                          name="linear_head")(x)
    model = keras.Model(inputs, logits, name="linear_probe")
    return model


# ----------------------------- Training / Evaluation -----------------------------
def linear_eval(dataset: str,
                image_size: int,
                batch_size: int,
                epochs: int,
                patience: int,
                base_lr: float,
                weight_decay: float,
                ckpt: str,
                backbone: str) -> None:
    """
    Perform linear probe training and report test accuracy.
    """
    ds_train, ds_val, ds_test = make_datasets(dataset, image_size, batch_size)
    _, _, num_classes = load_cifar(dataset)

    feat_extractor = build_feature_extractor(ckpt, image_size, backbone)

    # Optional BN adaptation before training (use the unaugmented validation stream)
    adapt_steps = int(os.getenv("BN_ADAPT_STEPS", "0"))
    if adapt_steps > 0:
        # ds_val has no augmentations; suitable for BN stats
        bn_adapt(feat_extractor, ds_val, steps=adapt_steps)

    # Build linear head and compile
    model = build_linear_probe(feat_extractor, num_classes, weight_decay)

    # Scale LR by batch size as commonly done in linear eval
    lr = base_lr * (batch_size / 256.0)
    try:
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    except TypeError:
        # Fallback for older Keras versions without `weight_decay` on the optimizer
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=False)

    # Use integer labels and from_logits=True to avoid one-hot issues
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    top1 = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    model.compile(optimizer=opt, loss=loss, metrics=[top1])

    # Callbacks: early stopping and LR reduction on plateau
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                      patience=patience,
                                      restore_best_weights=True,
                                      verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                          factor=0.2,
                                          patience=max(1, patience // 2),
                                          min_lr=1e-5,
                                          verbose=1),
    ]

    print(f"[eval_linear] dataset={dataset} img={image_size} bs={batch_size} epochs={epochs} ckpt={ckpt}")
    hist = model.fit(ds_train,
                     validation_data=ds_val,
                     epochs=epochs,
                     callbacks=callbacks,
                     verbose=2)

    # Final evaluation on the held-out test set
    test_loss, test_acc = model.evaluate(ds_test, verbose=2)
    print(f"Final eval:\naccuracy: {test_acc:.4f} - loss: {test_loss:.4f}")


# ----------------------------- CLI ------------------------------------------------
def main() -> None:
    """
    Script entry point.

    Examples
    --------
    python scripts/eval_linear.py --dataset cifar10 --image-size 224 --batch-size 32 \
        --epochs 50 --patience 20 --ckpt checkpoints_tf/vicreg_tf.weights.h5 --backbone resnet50v2
    """
    parser = argparse.ArgumentParser(description="Linear probe on CIFAR with a frozen encoder.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=0.1, help="Base LR scaled by (batch/256).")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay (L2 reg) for the head.")
    parser.add_argument("--ckpt", type=str, default="", help="Path to encoder weights (save_weights format).")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="Backbone (e.g., resnet50v2).")
    # Keep a flag for parity with your previous args; real BN-adapt is via env BN_ADAPT_STEPS
    parser.add_argument("--bn-adapt-steps", type=int, default=0,
                        help="Deprecated: use env BN_ADAPT_STEPS instead.")

    args = parser.parse_args()

    # If user still passes --bn-adapt-steps, mirror it into the env for this run
    if args.bn_adapt_steps and int(os.getenv("BN_ADAPT_STEPS", "0")) == 0:
        os.environ["BN_ADAPT_STEPS"] = str(args.bn_adapt_steps)

    set_seed(42)

    linear_eval(dataset=args.dataset,
                image_size=args.image_size,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                base_lr=args.base_lr,
                weight_decay=args.wd,
                ckpt=args.ckpt,
                backbone=args.backbone)


if __name__ == "__main__":
    main()
