"""
k-NN evaluation on top of a frozen encoder trained via (Adaptive) VICReg.

What this script does
---------------------
1) Rebuilds the SAME encoder (and projector skeleton) you used at train time.
   We **build** variables on dummy data, then load weights from `--ckpt`.
2) Extracts features from CIFAR train/test using either:
      --feature-layer feat  (Dense->feat identity; 2048-d)
      --feature-layer gap   (named 'gap' GlobalAveragePooling2D)
3) Runs cosine k-NN with temperature scaling (SimCLR-style soft voting).

Typical usage
-------------
python3 scripts/knn_eval.py \
  --device cpu \
  --ckpt checkpoints_tf/pretrain-c10_model1/pretrain_YYYYMMDD-HHMM/vicreg_encoder.weights.h5 \
  --dataset cifar10 --image-size 32 \
  --batch-size 1024 --k 200 --T 0.07 \
  --feature-layer feat \
  --out-dir results --run-name knn-feat_c10 --save-probs

Notes
-----
- If you pass an encoder-only checkpoint, we load directly into the encoder.
- If you pass a full-trainer checkpoint, we create a tiny wrapper with
  sublayers named exactly as in training: 'encoder' and 'proj'.
- Projector is **not used** for k-NN; it is only there to satisfy loaders.

Author: Nishant Kabra
Date: 11/16/2025
"""
from __future__ import annotations

import os
import argparse
import time
from typing import Tuple

# -------- device flag first (to let you pin CPU cleanly) ----------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===================== Model builders (from train_vicreg.py) =====================
def _named_conv_block(x: tf.Tensor, filters: int, conv_name: str, bn_name: str) -> tf.Tensor:
    """Conv2D -> BatchNorm -> ReLU block with explicit names (stable for loading)."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=conv_name)(x)
    x = layers.BatchNormalization(name=bn_name)(x)
    x = layers.ReLU()(x)
    return x

def build_encoder(image_size: int, feat_dim: int = 2048) -> keras.Model:
    """
    CIFAR encoder. Key named layers for eval/weight loading:
      - conv2d ... conv2d_6: conv blocks
      - gap                : GlobalAveragePooling2D
      - dense              : final feature FC
      - feat               : identity exposing the feature vector
    """
    inp = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = _named_conv_block(inp, 64,  "conv2d",   "batch_normalization")
    x = _named_conv_block(x,   64,  "conv2d_1", "batch_normalization_1")
    x = layers.MaxPool2D()(x)

    x = _named_conv_block(x,   128, "conv2d_2", "batch_normalization_2")
    x = _named_conv_block(x,   128, "conv2d_3", "batch_normalization_3")
    x = layers.MaxPool2D()(x)

    x = _named_conv_block(x,   256, "conv2d_4", "batch_normalization_4")
    x = _named_conv_block(x,   256, "conv2d_5", "batch_normalization_5")
    x = _named_conv_block(x,   256, "conv2d_6", "batch_normalization_6")

    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    f = layers.Dense(feat_dim, use_bias=True, name="dense")(gap)
    feat = layers.Lambda(lambda t: t, name="feat")(f)
    return keras.Model(inp, feat, name="encoder")

def build_projector(in_dim: int, out_dim: int, num_layers: int) -> keras.Model:
    """
    VICReg projector MLP with 1/2/3 layers and stable names:
      L=1: dense
      L=2: dense -> bn -> relu -> dense_1
      L=3: dense -> bn -> relu -> dense_1 -> bn_1 -> relu -> dense_2
    Model name is 'proj' to match training runs that used this naming.
    """
    assert num_layers >= 1, "proj-layers must be >= 1"
    inp = keras.Input(shape=(in_dim,), name="proj_in")
    x = inp
    if num_layers <= 1:
        out = layers.Dense(out_dim, use_bias=False, name="dense")(x)
    elif num_layers == 2:
        x = layers.Dense(out_dim, use_bias=False, name="dense")(x)
        x = layers.BatchNormalization(name="batch_normalization")(x)
        x = layers.ReLU()(x)
        out = layers.Dense(out_dim, use_bias=False, name="dense_1")(x)
    else:
        x = layers.Dense(out_dim, use_bias=False, name="dense")(x)
        x = layers.BatchNormalization(name="batch_normalization")(x)
        x = layers.ReLU()(x)
        x = layers.Dense(out_dim, use_bias=False, name="dense_1")(x)
        x = layers.BatchNormalization(name="batch_normalization_1")(x)
        x = layers.ReLU()(x)
        out = layers.Dense(out_dim, use_bias=False, name="dense_2")(x)
    return keras.Model(inp, out, name="proj")

# ============================== Loader helpers =================================
class VICRegTrainerSkeleton(keras.Model):
    """Minimal wrapper so we can load a *full* trainer checkpoint if needed."""
    def __init__(self, encoder: keras.Model, proj: keras.Model):
        super().__init__(name="vicreg_trainer")
        self.encoder = encoder   # must be literally named 'encoder'
        self.proj = proj         # must be named 'proj' (matches your training)

    def call(self, inputs, training=None):
        x1, x2 = inputs
        f1 = self.encoder(x1, training=False)
        f2 = self.encoder(x2, training=False)
        z1 = self.proj(f1, training=False)
        z2 = self.proj(f2, training=False)
        return z1, z2

def _force_build(encoder: keras.Model, proj: keras.Model, image_size: int):
    _ = encoder(tf.zeros([1, image_size, image_size, 3], tf.float32), training=False)
    _ = proj(tf.zeros([1, 2048], tf.float32), training=False)

def _is_encoder_only_h5(path: str) -> bool:
    """Heuristic: if filename says 'encoder' or H5 lacks any 'proj' group."""
    if "encoder" in os.path.basename(path):
        return True
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            # Keras v3 stores nested 'layers/...'; scan text for 'proj'
            txt = "|".join(keys + ["/".join(f[k].keys()) for k in keys if isinstance(f[k], h5py.Group)])
            return ("proj" not in txt) and ("projector" not in txt)
    except Exception:
        return False

def load_ckpt_flex(encoder: keras.Model, proj: keras.Model, ckpt_path: str, image_size: int):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    _force_build(encoder, proj, image_size)

    if _is_encoder_only_h5(ckpt_path):
        print(f"[loader] encoder-only checkpoint → {ckpt_path}")
        encoder.load_weights(ckpt_path)
        print("[loader] encoder weights loaded.")
        return

    print(f"[loader] full-trainer checkpoint → {ckpt_path}")
    # Build a tiny trainer to let Keras map 'encoder/...' and 'proj/...'
    trainer = VICRegTrainerSkeleton(encoder, proj)
    dummy = tf.zeros([1, image_size, image_size, 3], tf.float32)
    _ = trainer((dummy, dummy), training=False)
    try:
        trainer.load_weights(ckpt_path)
        print("[loader] full weights loaded.")
    except Exception as e:
        print("[loader] full load failed → fallback to encoder-only:", repr(e))
        encoder.load_weights(ckpt_path)
        print("[loader] encoder-only load succeeded.")

# ============================== Data utilities =================================
def load_dataset(name: str, image_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    elif name in {"cifar100", "cifar-100"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar100.load_data()
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    xtr = tf.image.resize(tf.convert_to_tensor(xtr), [image_size, image_size]).numpy()
    xte = tf.image.resize(tf.convert_to_tensor(xte), [image_size, image_size]).numpy()
    xtr = (xtr.astype("float32") / 255.0)
    xte = (xte.astype("float32") / 255.0)
    ytr = ytr.astype("int32").squeeze()
    yte = yte.astype("int32").squeeze()
    return xtr, ytr, xte, yte

def batched_features(model: keras.Model, x: np.ndarray, batch_size: int) -> np.ndarray:
    ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    outs = []
    for xb in ds:
        outs.append(model(xb, training=False).numpy())
    return np.concatenate(outs, axis=0)

# ============================== k-NN (cosine soft voting) ======================
def knn_predict(train_feats, train_labels, test_feats, k: int, T: float) -> np.ndarray:
    # Normalize
    train = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-9)
    test = test_feats / (np.linalg.norm(test_feats, axis=1, keepdims=True) + 1e-9)
    # Cosine similarity
    sims = test @ train.T  # [Nt, Ntr]
    # Top-k indices per row
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    part = np.take_along_axis(sims, idx, axis=1)  # [Nt, k]
    weights = np.exp(part / max(T, 1e-6))
    # Gather labels and vote
    yk = train_labels[idx]  # [Nt, k]
    num_classes = int(train_labels.max()) + 1
    votes = np.zeros((test.shape[0], num_classes), dtype=np.float32)
    for c in range(num_classes):
        votes[:, c] = weights * (yk == c)
        votes[:, c] = votes[:, c].sum(axis=1)
    return votes.argmax(axis=1)

# =================================== CLI ======================================
def parse_args():
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--T", type=float, default=0.07)
    p.add_argument("--feature-layer", choices=["feat", "gap"], default="feat")
    p.add_argument("--proj-out", type=int, default=8192)
    p.add_argument("--proj-layers", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--run-name", type=str, default="knn")
    p.add_argument("--save-probs", action="store_true")
    return p.parse_args()

# ================================== main ======================================
def main():
    args = parse_args()
    if _pre_args.device == "cpu":
        print("[knn_eval] Forcing CPU.")
    print(f"[knn_eval] Using device {'/CPU:0' if _pre_args.device=='cpu' else 'auto'}")

    # Build models
    encoder = build_encoder(args.image_size, feat_dim=2048)
    proj = build_projector(2048, args.proj_out, args.proj_layers)

    # Load weights
    print(f"[knn_eval] Loading weights: {args.ckpt}")
    load_ckpt_flex(encoder, proj, args.ckpt, args.image_size)

    # Feature extractors
    if args.feature_layer == "feat":
        feat_model = encoder  # output is the 'feat' identity (2048-d)
    else:
        # tap the named GAP layer
        feat_model = keras.Model(encoder.input, encoder.get_layer("gap").output)

    # Data
    xtr, ytr, xte, yte = load_dataset(args.dataset, args.image_size)

    t0 = time.time()
    ftr = batched_features(feat_model, xtr, args.batch_size)
    fte = batched_features(feat_model, xte, args.batch_size)
    preds = knn_predict(ftr, ytr, fte, k=args.k, T=args.T)
    acc = float((preds == yte).mean())
    elapsed = time.time() - t0

    # Save + print
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{time.strftime('%Y%m%d-%H%M%S')}__knn-{args.run_name}"
    np.savez_compressed(os.path.join(args.out_dir, f"{tag}.npz"),
                        preds=preds, y=yte, k=args.k, T=args.T, acc=acc)
    if args.save_probs:
        # Optionally save raw features for downstream analysis
        np.savez_compressed(os.path.join(args.out_dir, f"{tag}_feats.npz"),
                            train=ftr, test=fte, ytr=ytr, yte=yte)
    print(f"[knn_eval] Done. Acc={acc:.4f}. Outputs -> {os.path.join(args.out_dir, tag)}  (elapsed {elapsed:.1f}s)")

if __name__ == "__main__":
    main()
