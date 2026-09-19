"""kNN evaluation of a frozen encoder on CIFAR-10 or CIFAR-100.

Extracts features for the train and test sets, L2-normalizes them, and predicts each test label by
temperature-weighted voting among the `k` most cosine-similar training features. Appends one row with
the top-1 accuracy to a CSV. `--method-name` is only a label for that row: every encoder goes through
the same pipeline.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vicreg_tf import add_device_arg, build_encoder, preparse_device, select_device

# Has to run before anything initializes CUDA, so it lives at import time.
preparse_device()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kNN evaluation of a frozen encoder.")
    p.add_argument("--encoder-ckpt", type=str, required=True,
                   help="Encoder .weights.h5 file, or a run directory that contains one.")
    p.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10",
                   help="Dataset to evaluate on.")
    p.add_argument("--image-size", type=int, default=32, help="Input size after resizing.")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size for feature extraction.")
    p.add_argument("--feat-dim", type=int, default=2048, help="Encoder feature width (must match training).")
    p.add_argument("--k", type=int, default=200, help="Number of neighbors.")
    p.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature for the vote weights.")
    p.add_argument("--out-csv", type=str, required=True, help="CSV to append a result row to.")
    p.add_argument("--method-name", type=str, default="AdaptiveVICReg",
                   help="Label written to the CSV, for example 'VICReg' or 'AdaptiveVICReg'.")
    add_device_arg(p)
    return p.parse_args()


def _load_cifar(dataset: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """Return ((x_train, y_train), (x_test, y_test), num_classes) with labels flattened to 1D."""
    if dataset == "cifar10":
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
        num_classes = 10
    else:
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar100.load_data(label_mode="fine")
        num_classes = 100

    return (x_tr, y_tr.reshape(-1)), (x_te, y_te.reshape(-1)), num_classes


def _preprocess_images(x: np.ndarray, image_size: int) -> np.ndarray:
    """Scale uint8 images to float32 in [0, 1] and resize bilinearly if `image_size` differs.

    There is no augmentation: this measures the representation, not test-time tricks.
    """
    x = x.astype("float32") / 255.0
    if image_size != x.shape[1]:
        xt = tf.convert_to_tensor(x)
        xt = tf.image.resize(xt, (image_size, image_size), method="bilinear")
        x = xt.numpy()
    return x


def _build_ds(x: np.ndarray, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _extract_features(encoder: keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Run the encoder over `ds` and return features [N, D] and labels [N]."""
    feats = []
    labels = []
    for xb, yb in ds:
        z = encoder(xb, training=False)
        z = tf.reshape(z, [tf.shape(z)[0], -1])
        feats.append(z.numpy())
        labels.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _l2_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row."""
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
    """Predict test labels by cosine-similarity kNN with temperature-weighted voting.

    Args:
        train_feats: [N_train, D] feature bank.
        train_labels: [N_train] integer class ids.
        test_feats: [N_test, D] features to classify.
        k: Number of neighbors.
        temperature: Neighbor weights are exp(similarity / temperature), so lower values sharpen the vote.
        num_classes: Number of classes.
        chunk: Test samples per iteration, which bounds memory use.

    Returns:
        [N_test] predicted labels.
    """
    # After L2 normalization, cosine similarity is a plain dot product.
    train_feats = _l2_normalize(train_feats.astype(np.float32))
    test_feats = _l2_normalize(test_feats.astype(np.float32))

    n_test = test_feats.shape[0]
    preds = np.empty((n_test,), dtype=np.int32)

    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        q = test_feats[start:end]
        sim = np.matmul(q, train_feats.T)

        # argpartition avoids a full sort of every row.
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
    return float((y_true == y_pred).mean())


def _resolve_encoder_ckpt(path: str) -> str:
    """Return `path` if it is a file, or the encoder weights file inside it if it is a run directory."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ("vicreg_encoder.weights.h5", "encoder.weights.h5"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        f"Could not find encoder weights. Given: {path}\n"
        "If you passed a run directory, make sure it contains 'vicreg_encoder.weights.h5'."
    )


def main() -> None:
    args = parse_args()
    device_str = select_device(args.device, "knn")
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
