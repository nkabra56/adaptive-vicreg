"""
plots.py — diagnostics for Adaptive VICReg.

Outputs (saved under ./plots/):
  - std_hist.png             : histogram of per-dimension std of features
  - gamma_curve.png          : gamma_t vs epoch (if 'gamma_t' is in logs/train.csv)
  - cov_heatmap.png          : trace-normalized covariance heatmap (C/trace(C))
  - cov_delta_heatmap.png    : heatmap of C/trace(C) - (1/d)I

Dependencies: matplotlib, pandas (add to requirements.txt).

Author: Nishant Kabra
Date: 11/8/2025
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from src.vicreg_tf import (
    build_cifar10_supervised, build_cifar100_supervised,
    build_stl10_supervised, build_folder_supervised,
)

import sys, pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# ---------- tiny helpers ----------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def maybe_read_gamma(csv_path: str):
    """Return DataFrame with ['epoch','gamma_t'] if found; else None."""
    try:
        df = pd.read_csv(csv_path)
        if 'epoch' not in df.columns:
            df.insert(0, 'epoch', np.arange(len(df), dtype=int))  # fabricate epoch index if missing
        if 'gamma_t' in df.columns:
            return df[['epoch', 'gamma_t']].copy()
    except Exception:
        pass
    return None

# ---------- data loading ----------

def load_split(dataset: str, data_root: str, image_size: int, batch_size: int, tfds_dir: str | None, split: str):
    """Build supervised dataset; return train or test per `split`."""
    if dataset == "cifar10":
        tr, te, _ = build_cifar10_supervised(image_size, batch_size)
        return tr if split == "train" else te
    if dataset == "cifar100":
        tr, te, _ = build_cifar100_supervised(image_size, batch_size)
        return tr if split == "train" else te
    if dataset == "stl10":
        tr, te, _ = build_stl10_supervised(image_size, batch_size, data_dir=tfds_dir)
        return tr if split == "train" else te
    tr, va, _ = build_folder_supervised(data_root, image_size, batch_size)
    return tr if split == "train" else va

def load_encoder(encoder_path: str, probe: str, image_size: int) -> tf.keras.Model:
    """
    Load SavedModel and return a feature-extractor model for either:
      - 'pool': outputs the 'feat_pool' layer (if it exists),
      - 'projector': outputs the final projector vector.
    """
    enc = tf.keras.models.load_model(encoder_path, compile=False)
    enc.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    if probe == "pool":
        try:
            feat = tf.keras.Model(enc.input, enc.get_layer("feat_pool").output)
            h = feat(inputs, training=False)
            return tf.keras.Model(inputs, h, name="probe_pool")
        except Exception:
            pass  # fallback to projector
    z = enc(inputs, training=False)
    return tf.keras.Model(inputs, z, name="probe_projector")

# ---------- feature extraction ----------

def extract(model: tf.keras.Model, ds: tf.data.Dataset):
    """Materialize features and labels as NumPy arrays."""
    feats, labels = [], []
    for x, y in ds:
        z = model(x, training=False).numpy()
        feats.append(z)
        labels.append(y.numpy())
    return np.concatenate(feats, 0), np.concatenate(labels, 0).reshape(-1)

# ---------- plotting ----------

def plot_std_hist(std_vec: np.ndarray, out_path: str, gamma_latest: float | None = None):
    plt.figure(figsize=(8, 5))
    plt.hist(std_vec, bins=50)                                 # histogram bars
    med = float(np.median(std_vec))
    plt.axvline(med, linestyle="--")                           # median reference line
    plt.text(med, plt.ylim()[1]*0.9, f"median={med:.3f}", rotation=90, va="top", ha="right")
    if gamma_latest is not None:
        plt.axvline(gamma_latest, linestyle=":")               # latest gamma_t (from logs) for comparison
        plt.text(gamma_latest, plt.ylim()[1]*0.8, f"gamma_t={gamma_latest:.3f}", rotation=90, va="top", ha="right")
    plt.title("Per-dimension std (feature space)")
    plt.xlabel("std"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_gamma_curve(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"].values, df["gamma_t"].values)         # epoch vs gamma_t
    plt.title("gamma_t vs epoch")
    plt.xlabel("epoch"); plt.ylabel("gamma_t (EMA-clipped median)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_cov_heatmaps(Z: np.ndarray, out_cov: str, out_delta: str, eps: float = 1e-8):
    Zc = Z - Z.mean(axis=0, keepdims=True)                     # center features
    N = max(Zc.shape[0] - 1, 1)
    C = (Zc.T @ Zc) / float(N)                                 # covariance matrix
    trace = float(np.trace(C)) + eps
    Cn = C / trace                                             # trace-normalized covariance
    D = Cn.shape[0]
    I_scaled = np.eye(D) / float(D)
    Delta = Cn - I_scaled                                      # deviation from (1/d)I

    plt.figure(figsize=(6,5))
    plt.imshow(Cn, aspect='auto'); plt.colorbar()
    plt.title("Trace-normalized covariance C/trace(C)")
    plt.tight_layout(); plt.savefig(out_cov, dpi=150); plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(Delta, aspect='auto'); plt.colorbar()
    plt.title("C/trace(C) - (1/d)I")
    plt.tight_layout(); plt.savefig(out_delta, dpi=150); plt.close()

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Plot diagnostics for Adaptive VICReg")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100","stl10","folder"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--split", type=str, default="test", choices=["train","test"])
    p.add_argument("--probe", type=str, default="pool", choices=["pool","projector"])
    p.add_argument("--encoder-path", type=str, default=os.path.join("artifacts","encoder_savedmodel"))
    p.add_argument("--log-csv", type=str, default=os.path.join("logs","train.csv"))
    p.add_argument("--tfds-data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="plots")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    ds = load_split(args.dataset, args.data_root, args.image_size, args.batch_size,
                    args.tfds_data_dir, args.split)
    feature_model = load_encoder(args.encoder_path, args.probe, args.image_size)
    X, y = extract(feature_model, ds)                           # features [N,D], labels [N]
    std_vec = X.std(axis=0, ddof=1)                             # per-dim std (sample std)

    gamma_df = maybe_read_gamma(args.log_csv)                   # may be None
    gamma_latest = float(gamma_df['gamma_t'].iloc[-1]) if gamma_df is not None and len(gamma_df) else None

    plot_std_hist(std_vec, os.path.join(args.out_dir, "std_hist.png"), gamma_latest)
    if gamma_df is not None:
        plot_gamma_curve(gamma_df, os.path.join(args.out_dir, "gamma_curve.png"))
    plot_cov_heatmaps(X,
                      os.path.join(args.out_dir, "cov_heatmap.png"),
                      os.path.join(args.out_dir, "cov_delta_heatmap.png"))

    print(f"[OK] Saved figures to: {args.out_dir}")

if __name__ == "__main__":
    main()
