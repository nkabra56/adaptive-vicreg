"""
kNN evaluation on frozen encoder features (CIFAR-10/100).

Loads the encoder architecture (matching the `feat_dim` used during
pretraining), restores its weights from a checkpoint file or directory,
extracts train/test features with a simple deterministic preprocessing
pipeline, and runs cosine-similarity kNN with temperature-weighted voting to
predict test labels from the train feature bank. Reports top-1 accuracy and
appends one row to a CSV.

This script is method-agnostic: it doesn't know whether the encoder was
trained with baseline or Adaptive VICReg. `--method-name` is only a label
for the output CSV row, so both variants go through an identical, fair
evaluation pipeline.

Example:
  python3 scripts/knn_eval.py \
    --encoder-ckpt checkpoints_tf/pretrain-c10_checktrainer_2*/vicreg_encoder.weights.h5 \
    --dataset cifar10 --image-size 32 --batch-size 512 \
    --feat-dim 2048 \
    --k 200 --temperature 0.1 \
    --out-csv results/knn_eval.csv \
    --method-name AdaptiveVICReg

Notes:
- Features are L2-normalized and compared by cosine similarity, for
  scale-robust, generally higher-quality retrieval.
- Similarities are computed in chunks to bound peak memory.
- TF_CPP_MIN_LOG_LEVEL=1 is set to reduce TF log verbosity.
- Pass --device cpu or --device gpu to force a device; "auto" (default)
  probes the GPU and falls back to CPU on failure.
"""

from __future__ import annotations

# Make local src/ importable so `from vicreg_tf import ...` resolves, whether
# run from the repo root or from inside scripts/.
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if not _SRC_DIR.exists():
    raise RuntimeError(f"Could not find expected source directory: {_SRC_DIR}")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
import csv
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vicreg_tf import build_encoder, print_devices, enable_memory_growth, gpu_probe_ok


def _preparse_device() -> str:
    """Parse --device before TF initializes CUDA, so CPU-only mode can hide GPUs in time."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    args, _ = p.parse_known_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    return args.device


_DEVICE_FLAG = _preparse_device()


def decide_device(device_flag: str) -> str:
    """Resolve --device to a TF device string, probing the GPU in "auto" mode."""
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
    p.add_argument("--out-csv", type=str, required=True, help="CSV file to append a results row to.")
    p.add_argument("--method-name", type=str, default="AdaptiveVICReg",
                   help="Free-form label written to CSV (e.g. 'VICReg' or 'AdaptiveVICReg') "
                        "to distinguish encoders while keeping evaluation identical.")
    return p.parse_args()


def _load_cifar(dataset: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """Load CIFAR-10 or CIFAR-100 via `keras.datasets`, returning ((x_train, y_train), (x_test, y_test), num_classes)."""
    if dataset == "cifar10":
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
        num_classes = 10
    else:
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar100.load_data(label_mode="fine")
        num_classes = 100

    y_tr = y_tr.reshape(-1)
    y_te = y_te.reshape(-1)
    return (x_tr, y_tr), (x_te, y_te), num_classes


def _preprocess_images(x: np.ndarray, image_size: int) -> np.ndarray:
    """
    Convert uint8 [0,255] images to float32 in [0,1], bilinearly resizing if
    `image_size` differs from the source. Preprocessing is deliberately
    simple and deterministic, since this measures representation quality
    from pretraining, not test-time augmentation.
    """
    x = x.astype("float32") / 255.0
    if image_size != x.shape[1]:
        xt = tf.convert_to_tensor(x)
        xt = tf.image.resize(xt, (image_size, image_size), method="bilinear")
        x = xt.numpy()
    return x


def _build_ds(x: np.ndarray, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
    """Batched, prefetched inference dataset (no shuffling, no repeat)."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _extract_features(encoder: tf.keras.Model,
                      ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the frozen encoder (`training=False`) over `ds`, returning
    (features [N, D], labels [N]). Encoder outputs are flattened to 2D in
    case they aren't already.
    """
    feats = []
    labels = []
    for xb, yb in ds:
        z = encoder(xb, training=False)
        z = tf.reshape(z, [tf.shape(z)[0], -1])
        feats.append(z.numpy())
        labels.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _l2_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of a 2D array."""
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
    Predict test labels via cosine-similarity kNN with temperature-scaled
    soft voting.

    Args:
        train_feats: Train feature bank, [N_train, D].
        train_labels: Train integer class ids, [N_train].
        test_feats: Test features, [N_test, D].
        k: Number of neighbors.
        temperature: Softmax temperature for similarity-to-weight conversion;
            smaller sharpens the distribution.
        num_classes: Number of classes to vote over.
        chunk: Test-set slice size per iteration, to bound peak memory
            instead of building one [N_test, N_train] matrix.

    Returns:
        Predicted labels, [N_test].
    """
    # Cosine similarity reduces to a dot product after L2 normalization.
    train_feats = _l2_normalize(train_feats.astype(np.float32))
    test_feats  = _l2_normalize(test_feats.astype(np.float32))

    n_test = test_feats.shape[0]
    preds = np.empty((n_test,), dtype=np.int32)

    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        q = test_feats[start:end]
        sim = np.matmul(q, train_feats.T)

        # Partial top-k selection is cheaper than a full argsort.
        topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]

        rows = np.arange(end - start)[:, None]
        topk_sim = sim[rows, topk_idx]
        topk_lbl = train_labels[topk_idx]

        w = np.exp(topk_sim / float(temperature))

        votes = np.zeros((end - start, num_classes), dtype=np.float64)
        for j in range(k):
            np.add.at(votes, (np.arange(end - start), topk_lbl[:, j]), w[:, j])

        preds[start:end] = votes.argmax(axis=1)

    return preds


def _top1_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions, in [0, 1]."""
    return float((y_true == y_pred).mean())


def _resolve_encoder_ckpt(path: str) -> str:
    """
    Resolve `path` to an encoder weights file: returned as-is if it's a
    file, or searched for a known filename if it's a directory (so a run
    directory can be passed directly).
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


def main() -> None:
    args = parse_args()
    device_str = decide_device(_DEVICE_FLAG)
    print(f"[knn] Using device: {device_str}")

    (x_tr, y_tr), (x_te, y_te), num_classes = _load_cifar(args.dataset)
    x_tr = _preprocess_images(x_tr, args.image_size)
    x_te = _preprocess_images(x_te, args.image_size)

    ds_tr = _build_ds(x_tr, y_tr, args.batch_size)
    ds_te = _build_ds(x_te, y_te, args.batch_size)

    with tf.device(device_str):
        enc = build_encoder(args.image_size, feat_dim=args.feat_dim)
        ckpt = _resolve_encoder_ckpt(args.encoder_ckpt)
        print(f"[knn] Loading encoder weights from: {ckpt}")
        enc.load_weights(ckpt)

        print("[knn] Extracting train features...")
        f_tr, y_tr_np = _extract_features(enc, ds_tr)
        print("[knn] Extracting test features...")
        f_te, y_te_np = _extract_features(enc, ds_te)

        print(f"[knn] Running kNN: k={args.k} T={args.temperature}")
        y_pred = _knn_predict(f_tr, y_tr_np, f_te, args.k, args.temperature, num_classes)
        acc1 = _top1_acc(y_te_np, y_pred)

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
