"""
kNN evaluation for VICReg / Adaptive-VICReg checkpoints (cosine-sim kNN).

Overview
--------
Compute frozen encoder features for all train and test images, then classify
test features with a temperature-weighted kNN using cosine similarities.
This is a common non-parametric probe for SSL encoders.

Why a non-parametric probe?
---------------------------
It avoids training any classifier, so you can quickly test the quality of your
representations. If kNN accuracy is decent, the features likely carry useful
class information.

Implementation details
----------------------
- Device selection happens before TF import (cpu|gpu|auto).
- We rebuild a tiny loader model with the same encoder/projector names as during
  pretraining, call `load_weights(h5)`, then keep only the encoder.
- Features are L2 normalized before cosine similarity.
- We process test features in blocks (`--block`) to keep memory bounded.

Outputs
-------
- Prints Top-1 accuracy.
- Saves a JSON summary in `results/knn_eval_*.json`.

Author: Nishant Kabra
Date: 11/10/25
"""
from __future__ import annotations
import os
import json
import time
import argparse
from typing import Tuple

# -------------------- device pre-parse (before TF import!) --------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# -------------------------------- TensorFlow ----------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------- utilities -----------------------------------
def _enable_mem_growth():
    """
    Enable per-GPU memory growth to avoid grabbing all VRAM at start.
    """
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("[knn_eval] set_memory_growth warning:", repr(e))


def _gpu_probe() -> bool:
    """
    Validate basic GPU kernel launch (catches invalid PTX / driver mismatch).

    Returns
    -------
    bool
        True if a tiny Conv2D executes on GPU; False otherwise.
    """
    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform([1, 8, 8, 3])
            y = layers.Conv2D(4, 3, padding="same")(x)
            _ = tf.reduce_sum(y).numpy()
        print("[knn_eval] GPU probe OK.")
        return True
    except Exception as e:
        print("[knn_eval] GPU probe FAILED:", repr(e))
        return False


def decide_device(policy: str) -> str:
    """
    Map policy -> TF device string, probing GPU in 'auto' mode.
    """
    if policy == "cpu":
        print("[knn_eval] --device cpu -> using /CPU:0")
        return "/CPU:0"
    _enable_mem_growth()
    if policy == "gpu":
        print("[knn_eval] --device gpu requested; using /GPU:0 (may error).")
        return "/GPU:0"
    return "/GPU:0" if _gpu_probe() else "/CPU:0"


# ------------------------ data pipelines (CIFAR) ------------------------------
def _norm_img(x: tf.Tensor) -> tf.Tensor:
    """
    Convert uint8 image to float32 in [0,1].
    """
    return tf.image.convert_image_dtype(x, tf.float32)


def build_cifar10_sup(image_size: int, batch_size: int):
    """
    CIFAR-10 supervised pipes for feature extraction.

    Returns
    -------
    train_ds, test_ds, num_classes, n_train, n_test
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(lambda x, y: (tf.image.resize(_norm_img(x), [image_size, image_size]), tf.cast(y, tf.int32)),
             num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(lambda x, y: (tf.image.resize(_norm_img(x), [image_size, image_size]), tf.cast(y, tf.int32)),
             num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train, test, 10, x_train.shape[0], x_test.shape[0]


def build_cifar100_sup(image_size: int, batch_size: int):
    """
    CIFAR-100 supervised pipes for feature extraction.

    Returns
    -------
    train_ds, test_ds, num_classes, n_train, n_test
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(lambda x, y: (tf.image.resize(_norm_img(x), [image_size, image_size]), tf.cast(y, tf.int32)),
             num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(lambda x, y: (tf.image.resize(_norm_img(x), [image_size, image_size]), tf.cast(y, tf.int32)),
             num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train, test, 100, x_train.shape[0], x_test.shape[0]


# ------------------- encoder / projector (match trainer) ----------------------
def conv_block(x: tf.Tensor, filters: int, k: int = 3, s: int = 1) -> tf.Tensor:
    """
    Conv2D -> BatchNorm -> ReLU block. Keeps everything inside Keras layers.
    """
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_encoder(image_size: int) -> keras.Model:
    """
    Tiny CNN encoder -> Dense(2048) with name 'feat'.

    Returns
    -------
    keras.Model named 'encoder' whose final 2048-dim output is used for kNN.
    """
    inp = keras.Input(shape=(image_size, image_size, 3))
    x = conv_block(inp, 64)
    x = conv_block(x, 64, s=2)
    x = conv_block(x, 128)
    x = conv_block(x, 128, s=2)
    x = conv_block(x, 256)
    x = conv_block(x, 256, s=2)
    x = conv_block(x, 512)
    x = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(2048, name="feat")(x)
    return keras.Model(inp, feat, name="encoder")


def build_projector(in_dim: int, out_dim: int, num_layers: int) -> keras.Model:
    """
    MLP projector (rebuilt only for loading weights).

    Notes
    -----
    - Variable names must match training so HDF5 `load_weights()` succeeds.
    """
    assert num_layers >= 1
    inp = keras.Input(shape=(in_dim,))
    x = inp
    hidden = max(2048, out_dim)
    for i in range(num_layers - 1):
        x = layers.Dense(hidden, use_bias=False, name=f"proj_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"proj_bn_{i}")(x)
        x = layers.ReLU(name=f"proj_relu_{i}")(x)
    out = layers.Dense(out_dim, name="proj_out")(x)
    return keras.Model(inp, out, name="projector")


def load_encoder_from_ckpt(ckpt: str, image_size: int, proj_out: int, proj_layers: int) -> keras.Model:
    """
    Restore encoder weights by building a small input->encoder->projector graph.

    Returns
    -------
    keras.Model
        Encoder with weights restored.
    """
    encoder = build_encoder(image_size)
    projector = build_projector(2048, proj_out, proj_layers)
    inp = keras.Input(shape=(image_size, image_size, 3))
    z = projector(encoder(inp))
    tiny = keras.Model(inp, z, name="tiny_pretrain_model")
    _ = tiny(tf.zeros([1, image_size, image_size, 3]), training=False)
    print(f"[knn_eval] Loading weights: {ckpt}")
    tiny.load_weights(ckpt)
    print("[knn_eval] Weights loaded.")
    return encoder


# ------------------------------ kNN helpers -----------------------------------
def l2_normalize(x: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """
    L2-normalize a batch of features along `axis`.

    Returns
    -------
    tf.Tensor
        Same shape as x, L2 norm = 1 along `axis`.
    """
    return tf.math.l2_normalize(x, axis=axis)


def extract_features(encoder: keras.Model, ds: tf.data.Dataset, total: int, device: str) -> tf.Tensor:
    """
    Run encoder over dataset and return a dense [total, dim] tensor.

    Parameters
    ----------
    encoder : keras.Model
        Frozen feature extractor.
    ds : tf.data.Dataset
        Batches of (image, label).
    total : int
        Total number of items in ds (for logging only).
    device : str
        TF device string used in a context manager.

    Returns
    -------
    tf.Tensor (float32) of shape [N, D]
        Concatenated features in input order.
    """
    feats = []
    with tf.device(device):
        for xb, _ in ds:
            # Forward pass (inference mode); cast to float32 for stability.
            f = encoder(xb, training=False)
            f = tf.cast(f, tf.float32)
            feats.append(f)
    return tf.concat(feats, axis=0)


def knn_predict(
    feat_train: tf.Tensor,
    y_train: tf.Tensor,
    feat_test: tf.Tensor,
    k: int = 200,
    T: float = 0.07,
    block: int = 1000,
) -> tf.Tensor:
    """
    Temperature-weighted cosine-similarity kNN prediction (Top-1).

    Parameters
    ----------
    feat_train : tf.Tensor [N, D]
        Training features.
    y_train : tf.Tensor [N, 1]
        Training labels (integer class ids).
    feat_test : tf.Tensor [M, D]
        Test features.
    k : int
        Number of neighbors.
    T : float
        Temperature to sharpen similarity scores before softmax.
    block : int
        Number of test vectors to handle per block to bound memory.

    Returns
    -------
    tf.Tensor [M]
        Predicted class ids for each test sample.

    Procedure
    ---------
    - L2 normalize both train and test features.
    - For each block of test features:
        * Compute cosine sims vs all train features.
        * Take Top-k, divide by T, softmax to get neighbor weights.
        * Compute weighted vote in one-hot class space and argmax.
    """
    feat_train = l2_normalize(feat_train)
    feat_test = l2_normalize(feat_test)
    n_test = feat_test.shape[0]
    preds = []

    for start in range(0, n_test, block):
        end = min(n_test, start + block)
        ft = feat_test[start:end]  # [B, D]
        sim = tf.matmul(ft, feat_train, transpose_b=True)  # [B, N]
        topk = tf.math.top_k(sim, k=k)
        idx = topk.indices                   # [B, k]
        val = topk.values / T                # [B, k]
        w = tf.nn.softmax(val, axis=-1)      # [B, k] (weights per neighbor)

        # gather neighbor labels and do weighted voting
        y_n = tf.gather(y_train, idx)        # [B, k, 1]
        y_n = tf.squeeze(y_n, axis=-1)       # [B, k]
        num_classes = int(tf.reduce_max(y_train)) + 1
        oh = tf.one_hot(y_n, depth=num_classes)  # [B, k, C]
        vote = tf.reduce_sum(w[..., None] * oh, axis=1)  # [B, C]
        pred = tf.argmax(vote, axis=-1)       # [B]
        preds.append(pred)

    return tf.concat(preds, axis=0)


# ---------------------------------- CLI --------------------------------------
def parse_args():
    """
    Parse configuration flags for kNN evaluation.
    """
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", type=str, default="checkpoints_tf/vicreg_tf.weights.h5")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512, help="Feature extraction batch size.")
    p.add_argument("--proj-out", type=int, default=8192)
    p.add_argument("--proj-layers", type=int, default=3)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--T", type=float, default=0.07)
    p.add_argument("--block", type=int, default=1000, help="Test block size for memory-friendly scoring.")
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


# ---------------------------------- main --------------------------------------
def main():
    """
    Entrypoint: restore encoder -> extract features -> run kNN -> save results.

    Side Effects
    ------------
    - Writes `results/knn_eval_*.json`.
    - Prints Top-1 accuracy.
    """
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    device = decide_device(_pre_args.device)
    print(f"[knn_eval] Using device: {device}")

    with tf.device(device):
        # 1) Build supervised datasets for feature extraction
        if args.dataset == "cifar10":
            train, test, num_classes, n_train, n_test = build_cifar10_sup(args.image_size, args.batch_size)
        else:
            train, test, num_classes, n_train, n_test = build_cifar100_sup(args.image_size, args.batch_size)

        # Extract ground-truth labels in order matching features
        y_train_all = [yb for _, yb in train]
        y_test_all = [yb for _, yb in test]
        y_train = tf.concat(y_train_all, axis=0)  # [N,1]
        y_test = tf.concat(y_test_all, axis=0)    # [M,1]

        # 2) Load encoder from checkpoint
        encoder = load_encoder_from_ckpt(args.ckpt, args.image_size, args.proj_out, args.proj_layers)

        # 3) Extract features
        print("[knn_eval] Extracting train features...")
        feat_train = extract_features(encoder, train, n_train, device)
        print("[knn_eval] Extracting test features...")
        feat_test = extract_features(encoder, test, n_test, device)

        # 4) kNN classification
        print(f"[knn_eval] Running kNN (k={args.k}, T={args.T})...")
        t0 = time.time()
        pred = knn_predict(feat_train, y_train, feat_test, k=args.k, T=args.T, block=args.block)
        dur = time.time() - t0

        # 5) Accuracy
        acc = tf.reduce_mean(tf.cast(tf.equal(pred[:, None], y_test), tf.float32)).numpy().item()
        print(f"[knn_eval] Top-1 accuracy: {acc:.4f}  (knn seconds={dur:.1f})")

        # 6) Persist results
        out = {
            "task": "knn_eval",
            "dataset": args.dataset,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "proj_out": args.proj_out,
            "proj_layers": args.proj_layers,
            "k": args.k,
            "T": args.T,
            "block": args.block,
            "device": device,
            "test_acc": acc,
            "knn_seconds": dur,
            "ckpt": args.ckpt,
        }
        out_path = os.path.join(args.results_dir, f"knn_eval_{args.dataset}_{int(time.time())}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[knn_eval] wrote {out_path}")


if __name__ == "__main__":
    main()
