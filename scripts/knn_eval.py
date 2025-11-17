# scripts/knn_eval.py
"""
Non-parametric kNN evaluation on frozen encoder features.

What this script does
---------------------
Computes an embedding bank on CIFAR train set, then classifies test images by
soft kNN with temperature scaling over cosine similarities. Writes one CSV row
so `report_metrics.py` can merge it with the linear probe table.

Author: Nishant Kabra
"""

from __future__ import annotations
import argparse, csv, os, sys, glob
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Make local src/ importable
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vicreg_tf import build_encoder
from vicreg_tf import enable_memory_growth


def _load_cifar(name: str):
    if name == "cifar10":
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        K = 10
    elif name == "cifar100":
        (xtr, ytr), (xte, yte) = keras.datasets.cifar100.load_data(label_mode="fine")
        K = 100
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return (xtr, ytr.squeeze()), (xte, yte.squeeze()), K


def _make_ds_images(x, image_size: int, batch: int, shuffle: bool):
    x = tf.convert_to_tensor(x, tf.float32) / 255.0
    ds = tf.data.Dataset.from_tensor_slices(x)
    if shuffle:
        ds = ds.shuffle(10000)
    if image_size != 32:
        ds = ds.map(lambda im: tf.image.resize(im, (image_size, image_size)),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def _l2_normalize(feat: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(feat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return feat / n


def _resolve_ckpt(user_path: str) -> str:
    """
    Make loading robust:
      • If user_path is a file, use it.
      • If it's a directory, look for 'vicreg_encoder.weights.h5' inside.
      • If empty or not found, search common checkpoints dirs and pick newest.
    """
    if user_path and os.path.isfile(user_path):
        return user_path

    # If a directory was passed, try the standard filename inside it.
    if user_path and os.path.isdir(user_path):
        candidate = os.path.join(user_path, "vicreg_encoder.weights.h5")
        if os.path.isfile(candidate):
            return candidate

    # Fall back to glob search (newest). Search typical roots.
    roots = []
    if user_path:
        roots.append(user_path)
    roots.extend([
        str(_REPO / "checkpoints_tf"),
        str(_REPO / "checkpoints"),
    ])
    candidates: list[tuple[float, str]] = []
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "vicreg_encoder.weights.h5"), recursive=True):
            try:
                candidates.append((os.path.getmtime(p), p))
            except Exception:
                pass

    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    # Helpful error with tips, including showing the path that was attempted
    attempted = user_path or "<empty>"
    raise FileNotFoundError(
        "Could not locate encoder weights.\n"
        f"  • received --encoder-ckpt = {attempted}\n"
        "  • tried: <encoder-ckpt>, <encoder-ckpt>/vicreg_encoder.weights.h5,\n"
        "           and a recursive search under 'checkpoints_tf' / 'checkpoints'.\n"
        "Fix: pass the full file path to 'vicreg_encoder.weights.h5' or point\n"
        "      --encoder-ckpt to the run directory that contains it."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-ckpt", type=str, required=True)
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--feat-dim", type=int, default=2048)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--method-name", type=str, default="VICReg")
    return p.parse_args()


def main():
    args = parse_args()
    enable_memory_growth()

    # Load data
    (xtr, ytr), (xte, yte), K = _load_cifar(args.dataset)
    ds_train = _make_ds_images(xtr, args.image_size, args.batch_size, shuffle=False)
    ds_test  = _make_ds_images(xte, args.image_size, args.batch_size, shuffle=False)

    # Build and load encoder (robustly resolve the checkpoint path)
    ckpt_path = _resolve_ckpt(args.encoder_ckpt)
    enc = build_encoder(args.image_size, feat_dim=args.feat_dim)
    try:
        enc.load_weights(ckpt_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load encoder weights from: {ckpt_path}\n"
            "Make sure this is the *encoder* snapshot ('vicreg_encoder.weights.h5')."
        ) from e
    enc.trainable = False
    print(f"[knn-eval] Loaded encoder weights -> {ckpt_path}")

    # Build feature bank (train set)
    bank = []
    for xb in ds_train:
        fb = enc(xb, training=False)
        bank.append(fb.numpy())
    bank = np.concatenate(bank, axis=0).astype(np.float32)
    bank = _l2_normalize(bank)

    # Encode test set
    test_feats = []
    for xb in ds_test:
        fb = enc(xb, training=False)
        test_feats.append(fb.numpy())
    test_feats = np.concatenate(test_feats, axis=0).astype(np.float32)
    test_feats = _l2_normalize(test_feats)

    # kNN classification (soft voting with temperature).
    # Note: to avoid large memory spikes, compute similarities in chunks of test features.
    k = min(args.k, bank.shape[0])
    Nt = test_feats.shape[0]
    preds = np.empty(Nt, dtype=np.int32)
    chunk = 1024  # chunk size for test features

    for i in range(0, Nt, chunk):
        j = min(Nt, i + chunk)
        Q = test_feats[i:j]                     # [B, d]
        S = Q @ bank.T                          # [B, Ntrain]  (cosine since both L2-normalized)

        idx = np.argpartition(S, -k, axis=1)[:, -k:]         # top-k indices (unordered)
        topk = np.take_along_axis(S, idx, axis=1)            # [B, k] similarities
        w = np.exp(topk / max(1e-12, args.temperature))
        w /= np.sum(w, axis=1, keepdims=True)

        neigh_labels = ytr[idx]                                # [B, k]
        scores = np.zeros((j - i, K), dtype=np.float32)
        for b in range(j - i):
            # Accumulate soft votes for the k neighbors of this sample.
            np.add.at(scores[b], neigh_labels[b], w[b])
        preds[i:j] = np.argmax(scores, axis=1)

    top1 = float(np.mean(preds == yte))

    # CSV row
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ["method","dataset","batch","image_size","feat_dim","k","temperature","top1"]
    row = [args.method_name, args.dataset, args.batch_size, args.image_size,
           args.feat_dim, k, args.temperature, f"{top1:.4f}"]
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"[knn-eval] top1={top1:.4f} -> {args.out_csv}")

if __name__ == "__main__":
    main()
