"""
Linear evaluation on top of a frozen encoder trained via (Adaptive) VICReg.

What this script does
---------------------
1) Rebuilds a tiny model that matches the *training* object graph
   (encoder + projector skeleton), **builds** it on dummy data, then loads
   the pretrain weights from `--ckpt`. This ensures Keras creates the
   variables before loading (avoids “expected N variables, got 0”).
2) Extracts only the encoder trunk (projector is ignored for linear eval).
3) Trains a scikit-learn Logistic Regression classifier on **L2-normalized**
   frozen features from CIFAR train set; evaluates on the test set.
4) Handles GPU probe failures and pins to CPU cleanly if requested.

Typical usage
-------------
python3 scripts/eval_linear.py \
  --device cpu \
  --ckpt checkpoints_tf/pretrain-c10_model1/pretrain_YYYYMMDD-HHMM/vicreg_encoder.weights.h5 \
  --dataset cifar10 --image-size 32 \
  --batch-size 512 --epochs 100 \
  --augment --feature-layer feat \
  --proj-out 8192 --proj-layers 3 \
  --out-dir results --run-name lp-feat --lr 1e-3 --save-clf

Notes
-----
- Make sure --proj-out and --proj-layers match your pretraining run if you
  load a full-trainer checkpoint; for encoder-only checkpoints, projector
  is not used but we still build it for robust loading.
- By default we select device "auto": try GPU once, otherwise CPU (you can
  force CPU with --device cpu).

Author: Nishant Kabra
Date: 11/16/2025
"""
from __future__ import annotations

import os
import argparse
import time
from typing import Tuple

# -------- device flag first ----------
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ===================== Model builders (from train_vicreg.py) =====================
def _named_conv_block(x: tf.Tensor, filters: int, conv_name: str, bn_name: str) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=conv_name)(x)
    x = layers.BatchNormalization(name=bn_name)(x)
    x = layers.ReLU()(x)
    return x

def build_encoder(image_size: int, feat_dim: int = 2048) -> keras.Model:
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
    def __init__(self, encoder: keras.Model, proj: keras.Model):
        super().__init__(name="vicreg_trainer")
        self.encoder = encoder
        self.proj = proj

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
    if "encoder" in os.path.basename(path):
        return True
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
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
def load_dataset(name: str, image_size: int):
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        ncls = 10
    elif name in {"cifar100", "cifar-100"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar100.load_data()
        ncls = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    xtr = tf.image.resize(tf.convert_to_tensor(xtr), [image_size, image_size]).numpy().astype("float32") / 255.0
    xte = tf.image.resize(tf.convert_to_tensor(xte), [image_size, image_size]).numpy().astype("float32") / 255.0
    ytr = ytr.astype("int32").squeeze()
    yte = yte.astype("int32").squeeze()
    return xtr, ytr, xte, yte, ncls

def batched_features(model: keras.Model, x, batch_size: int) -> np.ndarray:
    ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    outs = []
    for xb in ds:
        outs.append(model(xb, training=False).numpy())
    return np.concatenate(outs, axis=0)

def l2norm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

# =================================== CLI ======================================
def parse_args():
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100, help="Max iters for LogisticRegression")
    p.add_argument("--feature-layer", choices=["feat", "gap"], default="feat")
    p.add_argument("--augment", action="store_true", help="(kept for API parity; eval uses clean images)")
    p.add_argument("--proj-out", type=int, default=8192)
    p.add_argument("--proj-layers", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--run-name", type=str, default="lp")
    p.add_argument("--lr", type=float, default=1e-3, help="(not used; placeholder for parity)")
    p.add_argument("--save-clf", action="store_true")
    return p.parse_args()

# ================================== main ======================================
def main():
    args = parse_args()
    if _pre_args.device == "cpu":
        print("[eval_linear] Forcing CPU.")
    print(f"[eval_linear] Using device {'/CPU:0' if _pre_args.device=='cpu' else 'auto'}")

    # Build + load
    encoder = build_encoder(args.image_size, feat_dim=2048)
    proj = build_projector(2048, args.proj_out, args.proj_layers)

    print(f"[eval_linear] Loading weights: {args.ckpt}")
    load_ckpt_flex(encoder, proj, args.ckpt, args.image_size)

    # Feature taps
    if args.feature_layer == "feat":
        feat_model = encoder
    else:
        feat_model = keras.Model(encoder.input, encoder.get_layer("gap").output)

    # Data
    xtr, ytr, xte, yte, ncls = load_dataset(args.dataset, args.image_size)

    t0 = time.time()
    ftr = batched_features(feat_model, xtr, args.batch_size)
    fte = batched_features(feat_model, xte, args.batch_size)
    ftr = l2norm(ftr); fte = l2norm(fte)

    # Linear probe (multinomial LR)
    clf = LogisticRegression(
        max_iter=args.epochs,
        solver="lbfgs",
        multi_class="auto",
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(ftr, ytr)
    preds = clf.predict(fte)
    acc = accuracy_score(yte, preds)
    elapsed = time.time() - t0

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{time.strftime('%Y%m%d-%H%M%S')}__lp-{args.run_name}"
    np.savez_compressed(os.path.join(args.out_dir, f"{tag}.npz"),
                        acc=acc, preds=preds, y=yte)
    if args.save_clf:
        joblib.dump(clf, os.path.join(args.out_dir, f"{tag}_clf.joblib"))

    print(f"[eval_linear] Done. Acc={acc:.4f}. Outputs -> {os.path.join(args.out_dir, tag)}  (elapsed {elapsed:.1f}s)")

if __name__ == "__main__":
    main()
