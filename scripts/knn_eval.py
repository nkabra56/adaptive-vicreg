"""
Script Title: kNN Evaluation on Frozen Encoder Features (CIFAR-10/100)

What the script does
--------------------
1) Loads my encoder architecture with the exact feature dimensionality I used
   during pretraining (e.g., feat_dim=2048).
2) Restores the encoder weights from a checkpoint file
   (typically "vicreg_encoder.weights.h5").  If I pass a directory, the script
   will try to resolve a typical filename inside it.
3) Extracts **train** and **test** features from CIFAR-10 or CIFAR-100 using a
   simple, deterministic preprocessing pipeline.
4) Runs **cosine-similarity kNN** with **temperature-weighted voting** to
   predict test labels from the train feature bank.
5) Reports **top-1 accuracy** and appends a single row to a CSV for bookkeeping.

Relationship to my Adaptive VICReg method
-----------------------------------------
This evaluation script is **method-agnostic**: it does not know whether the
encoder was trained with baseline VICReg or **Adaptive VICReg**. I simply pass
`--method-name` so that the row in my CSV makes it clear which training
configuration produced the encoder.  This is intentional: it ensures a fair,
identical evaluation pipeline for both baseline and adaptive variants.

Typical usage
-------------
# Example (CIFAR-10, k=200, T=0.1)
python3 scripts/knn_eval.py \
  --encoder-ckpt checkpoints_tf/pretrain-c10_checktrainer_*/vicreg_encoder.weights.h5 \
  --dataset cifar10 --image-size 32 --batch-size 512 \
  --feat-dim 2048 \
  --k 200 --temperature 0.1 \
  --out-csv results/knn_eval.csv \
  --method-name VICReg

Notes on resources and stability
--------------------------------
• I L2-normalize all features and use cosine similarity.  This makes kNN robust
  to scale and typically improves retrieval quality.
• I compute similarities in **chunks** to keep peak memory usage low.
• I set `TF_CPP_MIN_LOG_LEVEL=1` to reduce TF verbosity.
• If I want CPU-only evaluation or to ensure I use a specific device, I can pass
  `--device cpu` or `--device gpu`. The script also tries a quick GPU probe.

Author: Nishant Kabra
Date: 11/03/2025
"""

from __future__ import annotations

# ── Make sure my local package (under <repo>/src) is importable ─────────────────
# I add <repo>/src to sys.path so `from vicreg_tf import ...` works whether I run
# from project root or from inside scripts/. This matches my training scripts.
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]   # <repo> (one level above scripts/)
_SRC_DIR = _REPO_ROOT / "src"
if not _SRC_DIR.exists():
    # If this fails, I'm probably running from the wrong working directory.
    raise RuntimeError(f"Could not find expected source directory: {_SRC_DIR}")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# ────────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

# I reuse these helpers from my package. They are small utilities to print
# devices, enable memory growth, and quick-probe the GPU by running a trivial op.
from vicreg_tf import build_encoder, print_devices, enable_memory_growth, gpu_probe_ok


# ==============================================================================
# Device selection (I parse --device early to control CUDA visibility before TF)
# ==============================================================================

def _preparse_device() -> str:
    """
    Parse --device **before** TensorFlow initializes.

    Why I do this:
    --------------
    If I want to force CPU mode (or control GPU visibility) I need to set the
    relevant environment variables (e.g., CUDA_VISIBLE_DEVICES) **before**
    importing/initializing CUDA drivers. Parsing here lets me do that cleanly.

    Returns
    -------
    device_flag : str
        One of {"auto","gpu","cpu"} to be used by `decide_device`.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    args, _ = p.parse_known_args()

    # If user asked for CPU, hide CUDA devices **before** TF loads them.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Quiet down TF logging a bit (still shows warnings/errors).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    return args.device


_DEVICE_FLAG = _preparse_device()


def decide_device(device_flag: str) -> str:
    """
    Decide the TensorFlow device string to use for feature extraction.

    Behavior
    --------
    • "cpu": force CPU ("/CPU:0").
    • "gpu": prefer GPU ("/GPU:0"), do not fall back.
    • "auto": probe GPU with a tiny Conv2D; if it fails, fall back to CPU.

    Returns
    -------
    device_str : str
        TensorFlow device string, e.g., "/GPU:0" or "/CPU:0".
    """
    if device_flag == "cpu":
        print("[knn] Forcing CPU mode per flag.")
        return "/CPU:0"

    # Print and configure GPUs for good behavior (memory growth).
    print_devices()
    enable_memory_growth()

    if device_flag == "gpu":
        print("[knn] Requested GPU; will not fall back.")
        return "/GPU:0"

    # "auto" path: attempt a tiny GPU op to verify kernels and drivers.
    ok = gpu_probe_ok()
    if not ok:
        print("[knn] GPU probe failed; falling back to CPU.")
    return "/GPU:0" if ok else "/CPU:0"


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments for my kNN evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed flags including encoder path, dataset, k, temperature, etc.
    """
    p = argparse.ArgumentParser(parents=[argparse.ArgumentParser(add_help=False)])
    p.add_argument("--encoder-ckpt", type=str, required=True,
                   help="Path to encoder .weights.h5 (or a directory containing it).")
    p.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10",
                   help="Which dataset to evaluate on.")
    p.add_argument("--image-size", type=int, default=32, help="Square side used to preprocess inputs.")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size for feature extraction.")
    p.add_argument("--feat-dim", type=int, default=2048, help="Encoder feature width (must match training).")
    p.add_argument("--k", type=int, default=200, help="Number of neighbors for kNN retrieval.")
    p.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature for voting.")
    p.add_argument("--out-csv", type=str, required=True, help="CSV file where I append a results row.")
    p.add_argument("--method-name", type=str, default="AdaptiveVICReg",
                   help="Free-form label written to CSV (e.g., 'VICReg' or 'AdaptiveVICReg'). "
                        "This keeps evaluation identical while allowing me to distinguish encoders.")
    return p.parse_args()


# ==============================================================================
# Data utilities (CIFAR loaders and preprocessing)
# ==============================================================================

def _load_cifar(dataset: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """
    Load CIFAR-10 or CIFAR-100 and return train/test arrays plus class count.

    Args
    ----
    dataset : {"cifar10","cifar100"}
        Which dataset to load via `keras.datasets`.

    Returns
    -------
    (x_train, y_train), (x_test, y_test), num_classes : tuple
        Numpy arrays of images and labels along with number of classes.

    Implementation notes
    --------------------
    • Labels come back as shape [N,1]; I flatten to shape [N] for convenience.
    • I do **not** perform mean/std normalization here—kNN with cosine similarity
      and L2-normalized features is fairly stable without it for CIFAR.
    """
    if dataset == "cifar10":
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
        num_classes = 10
    else:
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar100.load_data(label_mode="fine")
        num_classes = 100

    y_tr = y_tr.reshape(-1)  # [N]
    y_te = y_te.reshape(-1)  # [N]
    return (x_tr, y_tr), (x_te, y_te), num_classes


def _preprocess_images(x: np.ndarray, image_size: int) -> np.ndarray:
    """
    Convert images to float32 in [0,1] and resize if needed.

    Args
    ----
    x : np.ndarray
        Input images as uint8 in [0,255] (CIFAR default).
    image_size : int
        Target square size; if not 32, I bilinearly resize.

    Returns
    -------
    np.ndarray
        Images as float32 in [0,1], shape [N, H, W, 3].

    Rationale
    ---------
    I deliberately keep preprocessing simple and deterministic for evaluation.
    The goal is to measure the *representation quality* learned during
    pretraining, not to squeeze accuracy through heavy test-time augmentation.
    """
    x = x.astype("float32") / 255.0
    if image_size != x.shape[1]:
        # Resize with TF once (operates on NHWC tensors).
        xt = tf.convert_to_tensor(x)
        xt = tf.image.resize(xt, (image_size, image_size), method="bilinear")
        x = xt.numpy()
    return x


def _build_ds(x: np.ndarray, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
    """
    Build a simple input pipeline for inference (no shuffling, no repeats).

    Args
    ----
    x, y : np.ndarray
        Images and labels.
    batch_size : int
        Inference batch size.

    Returns
    -------
    tf.data.Dataset
        Dataset yielding (images, labels) mini-batches for the encoder.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ==============================================================================
# Feature extraction (frozen encoder)
# ==============================================================================

def _extract_features(encoder: tf.keras.Model,
                      ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run my frozen encoder on a dataset to collect features and labels.

    Args
    ----
    encoder : tf.keras.Model
        The backbone created by `build_encoder(image_size, feat_dim)`.
    ds : tf.data.Dataset
        Batched dataset yielding (images, labels) pairs.

    Returns
    -------
    (feats, labels) : (np.ndarray, np.ndarray)
        • feats: shape [N, D] (I flatten spatial dims if present).
        • labels: shape [N], integer class ids.

    Implementation detail
    ---------------------
    I use `training=False` so BatchNorm (if any) and other layers behave in eval
    mode.  I also reshape features to 2D so the kNN math is straightforward.
    """
    feats = []
    labels = []
    for xb, yb in ds:
        z = encoder(xb, training=False)
        z = tf.reshape(z, [tf.shape(z)[0], -1])   # ensure [N, D] even if encoder outputs [N, H, W, C]
        feats.append(z.numpy())
        labels.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


# ==============================================================================
# kNN core (cosine similarity + temperature-weighted voting)
# ==============================================================================

def _l2_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize each row vector of a 2D array.

    Args
    ----
    a : np.ndarray
        Input of shape [N, D].
    eps : float
        Numerical guard against division by zero.

    Returns
    -------
    np.ndarray
        Row-normalized array with the same shape.
    """
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return a / nrm


def _knn_predict(train_feats: np.ndarray,
                 train_labels: np.ndarray,
                 test_feats: np.ndarray,
                 k: int,
                 temperature: float,
                 num_classes: int,
                 chunk: int = 1024) -> np.ndarray:
    """
    Predict labels for test features using cosine-similarity kNN.

    I do chunked similarity computation to reduce peak memory usage:
    instead of building a giant [N_test, N_train] matrix at once,
    I process contiguous slices of the test set.

    Args
    ----
    train_feats : np.ndarray
        Feature bank for training set, shape [N_train, D].
    train_labels : np.ndarray
        Integer class ids for training set, shape [N_train].
    test_feats : np.ndarray
        Feature matrix for test set, shape [N_test, D].
    k : int
        Number of nearest neighbors to consider.
    temperature : float
        Softmax temperature for turning similarities into weights. Smaller
        temperature sharpens the distribution (more peaky).
    num_classes : int
        Number of classes for the voting bins.
    chunk : int
        Size of the test slice processed per loop iteration.

    Returns
    -------
    np.ndarray
        Predicted labels for test set, shape [N_test].

    Implementation notes
    --------------------
    • I **L2-normalize** both train and test features first. Cosine similarity
      then reduces to a dot-product.
    • I use `np.argpartition` to get top-k indices efficiently without a full
      sort of all similarities.
    • I apply temperature-scaled exp(.) to top-k similarities and vote into
      class bins via `np.add.at` to avoid Python loops per class.
    """
    # Normalize features once (cosine similarity as dot product).
    train_feats = _l2_normalize(train_feats.astype(np.float32))
    test_feats  = _l2_normalize(test_feats.astype(np.float32))

    n_test = test_feats.shape[0]
    preds = np.empty((n_test,), dtype=np.int32)

    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        q = test_feats[start:end]                     # [B, D]
        sim = np.matmul(q, train_feats.T)            # [B, N_train], cosine similarity (dot product)

        # Top-k indices via partial selection (faster + lower memory than full argsort).
        topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]

        # Gather the top-k similarities and corresponding labels.
        rows = np.arange(end - start)[:, None]
        topk_sim = sim[rows, topk_idx]               # [B, k]
        topk_lbl = train_labels[topk_idx]            # [B, k]

        # Temperature-scaled softmax weights (no normalization needed for voting).
        w = np.exp(topk_sim / float(temperature))    # [B, k]

        # Accumulate weighted votes into per-class bins.
        votes = np.zeros((end - start, num_classes), dtype=np.float64)
        for j in range(k):
            # votes[b, lbl_j] += w[b, j]  for each row b
            np.add.at(votes, (np.arange(end - start), topk_lbl[:, j]), w[:, j])

        preds[start:end] = votes.argmax(axis=1)      # predicted class = argmax of votes

    return preds


def _top1_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute top-1 accuracy.

    Args
    ----
    y_true, y_pred : np.ndarray
        True and predicted labels as shape [N] integer arrays.

    Returns
    -------
    float
        Fraction of correct predictions in [0,1].
    """
    return float((y_true == y_pred).mean())


# ==============================================================================
# Checkpoint resolution helper
# ==============================================================================

def _resolve_encoder_ckpt(path: str) -> str:
    """
    Resolve an encoder checkpoint path robustly.

    Args
    ----
    path : str
        Either a direct file path to `*.weights.h5` or a directory that
        contains a typical filename like "vicreg_encoder.weights.h5".

    Returns
    -------
    str
        Resolved file path to the encoder weights.

    Raises
    ------
    FileNotFoundError
        If no suitable file is found.

    Why I do this
    -------------
    My training script writes checkpoints into per-run directories. Sometimes I
    just copy/paste the run directory into this flag; this helper saves me from
    having to type the full filename.
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "vicreg_encoder.weights.h5"),
            os.path.join(path, "encoder.weights.h5"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
    raise FileNotFoundError(
        f"Could not find encoder weights. Given: {path}\n"
        "If you passed a run directory, make sure it contains "
        "'vicreg_encoder.weights.h5'."
    )


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    """
    Entry point: load data, build encoder, extract features, run kNN, write CSV.

    Steps
    -----
    1) Decide device (CPU/GPU/auto) and print it.
    2) Load CIFAR (train/test) and preprocess to [0,1] float; resize if needed.
    3) Build the encoder (`build_encoder(image_size, feat_dim)`) and load weights.
    4) Extract train/test features in batches with `training=False`.
    5) Run kNN with cosine similarity and temperature voting.
    6) Append a results row to `--out-csv` with columns:
       [dataset, method, k, temperature, top1].

    **ADAPTIVE VICREG NOTE**:
    -------------------------
    The *only* place that is aware of "Adaptive VICReg" here is the string I pass
    via `--method-name`. This script does not change behavior based on the method;
    it intentionally remains identical for both baseline and adaptive encoders to
    keep evaluation fair and comparable.
    """
    args = parse_args()
    device_str = decide_device(_DEVICE_FLAG)
    print(f"[knn] Using device: {device_str}")

    # 1) Load and preprocess data
    (x_tr, y_tr), (x_te, y_te), num_classes = _load_cifar(args.dataset)
    x_tr = _preprocess_images(x_tr, args.image_size)
    x_te = _preprocess_images(x_te, args.image_size)

    # 2) Build datasets for feature extraction
    ds_tr = _build_ds(x_tr, y_tr, args.batch_size)
    ds_te = _build_ds(x_te, y_te, args.batch_size)

    with tf.device(device_str):
        # 3) Build encoder and load weights
        enc = build_encoder(args.image_size, feat_dim=args.feat_dim)
        ckpt = _resolve_encoder_ckpt(args.encoder_ckpt)
        print(f"[knn] Loading encoder weights from: {ckpt}")
        enc.load_weights(ckpt)  # pure load; encoder is kept frozen

        # 4) Extract features
        print("[knn] Extracting train features...")
        f_tr, y_tr_np = _extract_features(enc, ds_tr)
        print("[knn] Extracting test features...")
        f_te, y_te_np = _extract_features(enc, ds_te)

        # 5) kNN predict
        print(f"[knn] Running kNN: k={args.k} T={args.temperature}")
        y_pred = _knn_predict(f_tr, y_tr_np, f_te, args.k, args.temperature, num_classes)
        acc1 = _top1_acc(y_te_np, y_pred)

    # 6) Write/append CSV row with a consistent header
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ["dataset", "method", "k", "temperature", "top1"]
    exists = os.path.isfile(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)  # create header only once
        w.writerow([args.dataset, args.method_name, args.k, args.temperature, f"{acc1:.4f}"])

    print(f"[knn] top-1 accuracy: {acc1:.4f}")
    print(f"[knn] Wrote results row to: {args.out_csv}")


if __name__ == "__main__":
    main()
