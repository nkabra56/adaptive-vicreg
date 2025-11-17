"""
Script Title: kNN Evaluation on Frozen Encoder Features (CIFAR-10/100)

Purpose
-------
I use this script to evaluate a pretrained encoder by kNN classification on
frozen features. It loads my encoder weights, extracts features for train/test,
and computes top-1 accuracy with temperature-weighted voting.

Typical usage
-------------
python3 scripts/knn_eval.py \
  --encoder-ckpt checkpoints_tf/pretrain-c10_model9_20251117-1530/vicreg_encoder.weights.h5 \
  --dataset cifar10 --image-size 32 --batch-size 512 \
  --feat-dim 2048 \
  --k 200 --temperature 0.1 \
  --out-csv results/pretrain-c10_model9/pretrain-c10_model9_knn_eval.csv \
  --method-name AdaptiveVICReg

Notes
-----
• I keep memory use sane by batching the test set for similarity computation.
• If I pass a directory path instead of a file for --encoder-ckpt, the script
  will try to locate a typical filename like "vicreg_encoder.weights.h5".
• The features come from `build_encoder(image_size, feat_dim)` and I freeze it.
• I L2-normalize features and use cosine similarity with temperature.

Author: Nishant Kabra
Date: 11/17/2025
"""
from __future__ import annotations

# --- Make sure my local package (under <repo>/src) is importable. --------------
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]   # <repo>
_SRC_DIR = _REPO_ROOT / "src"
if not _SRC_DIR.exists():
    raise RuntimeError(f"Could not find expected source directory: {_SRC_DIR}")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# ------------------------------------------------------------------------------

import argparse
import csv
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vicreg_tf import build_encoder, print_devices, enable_memory_growth, gpu_probe_ok


# ----------------------- CLI and device handling -------------------------------

def _preparse_device() -> str:
    """Parse --device early so I can hide GPUs before TF initializes."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    args, _ = p.parse_known_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    return args.device


_DEVICE_FLAG = _preparse_device()


def decide_device(device_flag: str) -> str:
    """
    Decide TF device string based on my preference and a quick GPU probe.
    """
    if device_flag == "cpu":
        print("[knn] Forcing CPU mode per flag.")
        return "/CPU:0"

    print_devices()
    enable_memory_growth()

    if device_flag == "gpu":
        print("[knn] Requested GPU; will not fall back.")
        return "/GPU:0"

    ok = gpu_probe_ok()
    if not ok:
        print("[knn] GPU probe failed; falling back to CPU.")
    return "/GPU:0" if ok else "/CPU:0"


def parse_args() -> argparse.Namespace:
    """
    Define and parse CLI arguments for my kNN evaluation.
    """
    p = argparse.ArgumentParser(parents=[argparse.ArgumentParser(add_help=False)])
    p.add_argument("--encoder-ckpt", type=str, required=True,
                   help="Path to encoder .weights.h5 (or folder containing it).")
    p.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10",
                   help="Evaluation dataset.")
    p.add_argument("--image-size", type=int, default=32, help="Square crop size.")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size for feature extraction.")
    p.add_argument("--feat-dim", type=int, default=2048, help="Encoder feature width.")
    p.add_argument("--k", type=int, default=200, help="Number of neighbors for kNN.")
    p.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature for voting.")
    p.add_argument("--out-csv", type=str, required=True, help="Where to append a results row.")
    p.add_argument("--method-name", type=str, default="AdaptiveVICReg", help="Name to record in CSV.")
    return p.parse_args()


# -------------------------- Data utilities -------------------------------------

def _load_cifar(dataset: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """
    Load CIFAR-10/100 using Keras datasets. I return (x_train, y_train), (x_test, y_test), num_classes.
    """
    if dataset == "cifar10":
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
        num_classes = 10
    else:
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar100.load_data(label_mode="fine")
        num_classes = 100

    # Flatten labels to shape [N]
    y_tr = y_tr.reshape(-1)
    y_te = y_te.reshape(-1)
    return (x_tr, y_tr), (x_te, y_te), num_classes


def _preprocess_images(x: np.ndarray, image_size: int) -> np.ndarray:
    """
    Convert to float32 in [0,1]; optionally resize to image_size (if not 32).
    I keep it simple and avoid whitening/mean-std normalization here.
    """
    x = x.astype("float32") / 255.0
    if image_size != x.shape[1]:
        # Resize with TF once (NHWC)
        xt = tf.convert_to_tensor(x)
        xt = tf.image.resize(xt, (image_size, image_size), method="bilinear")
        x = xt.numpy()
    return x


def _build_ds(x: np.ndarray, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
    """
    Build a simple tf.data pipeline for inference (no shuffles, no repeats).
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------- Feature extraction -----------------------------------

def _extract_features(encoder: tf.keras.Model,
                      ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run frozen encoder on a dataset to collect features and labels.
    """
    feats = []
    labels = []
    for xb, yb in ds:
        z = encoder(xb, training=False)
        z = tf.reshape(z, [tf.shape(z)[0], -1])  # flatten feature if needed
        feats.append(z.numpy())
        labels.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


# ----------------------------- kNN core -----------------------------------------

def _l2_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize rows of a 2D array (N, D).
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
    Predict labels for test features using cosine-similarity kNN with
    temperature-weighted voting. I compute in chunks to save memory.
    """
    # Normalize features once
    train_feats = _l2_normalize(train_feats.astype(np.float32))
    test_feats  = _l2_normalize(test_feats.astype(np.float32))

    n_test = test_feats.shape[0]
    preds = np.empty((n_test,), dtype=np.int32)

    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        q = test_feats[start:end]                     # [B, D]
        sim = np.matmul(q, train_feats.T)            # [B, Ntrain]

        # Take top-k indices
        # Use argpartition then gather for speed/memory
        topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        # Gather top-k sims and labels
        rows = np.arange(end - start)[:, None]
        topk_sim = sim[rows, topk_idx]               # [B, k]
        topk_lbl = train_labels[topk_idx]            # [B, k]

        # Temperature-softmax weights per test sample over its k neighbors
        w = np.exp(topk_sim / float(temperature))    # [B, k]

        # Accumulate weighted votes into class bins via advanced indexing
        votes = np.zeros((end - start, num_classes), dtype=np.float64)
        # For each position in the k list, accumulate
        for j in range(k):
            np.add.at(votes, (np.arange(end - start), topk_lbl[:, j]), w[:, j])

        preds[start:end] = votes.argmax(axis=1)

    return preds


def _top1_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute top-1 accuracy in [0, 1]."""
    return float((y_true == y_pred).mean())


# ----------------------------- Checkpoint utils ---------------------------------

def _resolve_encoder_ckpt(path: str) -> str:
    """
    Resolve an encoder checkpoint path.
    If I pass a directory, try common filenames. Else require that file exists.
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


# ------------------------------------ main --------------------------------------

def main() -> None:
    """
    Entry point: load data, build encoder, extract features, run kNN, write CSV row.
    """
    args = parse_args()
    device_str = decide_device(_DEVICE_FLAG)
    print(f"[knn] Using device: {device_str}")

    (x_tr, y_tr), (x_te, y_te), num_classes = _load_cifar(args.dataset)
    x_tr = _preprocess_images(x_tr, args.image_size)
    x_te = _preprocess_images(x_te, args.image_size)

    ds_tr = _build_ds(x_tr, y_tr, args.batch_size)
    ds_te = _build_ds(x_te, y_te, args.batch_size)

    with tf.device(device_str):
        # Build encoder and load weights robustly
        enc = build_encoder(args.image_size, feat_dim=args.feat_dim)
        ckpt = _resolve_encoder_ckpt(args.encoder_ckpt)
        print(f"[knn] Loading encoder weights from: {ckpt}")
        enc.load_weights(ckpt)

        # Extract features
        print("[knn] Extracting train features...")
        f_tr, y_tr_np = _extract_features(enc, ds_tr)
        print("[knn] Extracting test features...")
        f_te, y_te_np = _extract_features(enc, ds_te)

        # kNN predict
        print(f"[knn] Running kNN: k={args.k} T={args.temperature}")
        y_pred = _knn_predict(f_tr, y_tr_np, f_te, args.k, args.temperature, num_classes)
        acc1 = _top1_acc(y_te_np, y_pred)

    # Append results to CSV (create parent dir if needed)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ["dataset", "method", "k", "temperature", "top1"]
    exists = os.path.isfile(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([args.dataset, args.method_name, args.k, args.temperature, f"{acc1:.4f}"])
    print(f"[knn] top-1 accuracy: {acc1:.4f}")
    print(f"[knn] Wrote results row to: {args.out_csv}")


if __name__ == "__main__":
    main()
