"""
Linear probe on frozen encoder features.

Key changes
-----------
- Uses your named encoder build to match training
- Adds --save-probs to emit standardized:
    <out_dir>/<STAMP>__lp-<run_name>/probs_test.npz
  which contains: y_true, probs
- Also keeps a small features dump for debugging

Usage
-----
python3 scripts/eval_linear.py --device cpu \
  --ckpt checkpoints_tf/.../vicreg_encoder.weights.h5 \
  --dataset cifar10 --image-size 32 \
  --batch-size 512 --epochs 100 \
  --feature-layer feat \
  --out-dir results --run-name lp-feat_c10_model1 \
  --save-clf --save-probs
"""

import os, argparse, datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# scikit-learn import path is 'sklearn'
from sklearn.linear_model import LogisticRegression
from joblib import dump

# ------------- Device -------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["auto", "gpu", "cpu"], default="cpu")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# ------------- Encoder (named) -------------
def _named_conv_block(x: tf.Tensor, filters: int, conv_name: str, bn_name: str) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=conv_name)(x)
    x = layers.BatchNormalization(name=bn_name)(x)
    x = layers.ReLU()(x)
    return x

def build_encoder(image_size: int, feat_dim: int = 2048) -> keras.Model:
    inp = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = _named_conv_block(inp,  64, "conv2d",   "batch_normalization")
    x = _named_conv_block(x,    64, "conv2d_1", "batch_normalization_1")
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

# ------------- Data -------------
def load_cifar(name: str):
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        C = 10
    elif name in {"cifar100", "cifar-100"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar100.load_data()
        C = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    return (xtr, ytr.reshape(-1).astype(np.int64)), (xte, yte.reshape(-1).astype(np.int64)), C

def make_ds(x, bs):
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds

def extract(enc, x, bs, which: str):
    ds = make_ds(x, bs)
    outs = []
    if which == "gap":
        # submodel to tap 'gap'
        x0 = enc.get_layer("conv2d").input
        sub = keras.Model(x0, enc.get_layer("gap").output)
    else:
        sub = enc  # 'feat'
    for b in ds:
        b = tf.image.convert_image_dtype(b, tf.float32)
        y = sub(b, training=False)
        outs.append(y.numpy())
    return np.concatenate(outs, axis=0)

# ------------- CLI + main -------------
def parse_args():
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)  # kept for compatibility; not used
    p.add_argument("--feature-layer", choices=["feat", "gap"], default="feat")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--run-name", type=str, default="lp")
    p.add_argument("--lr", type=float, default=1e-3)  # kept for compatibility
    p.add_argument("--save-clf", action="store_true")
    p.add_argument("--save-probs", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    print("[eval_linear] Forcing CPU.")
    print("[eval_linear] Using device /CPU:0")

    (xtr, ytr), (xte, yte), C = load_cifar(args.dataset)
    enc = build_encoder(args.image_size)
    _ = enc(tf.zeros([1, args.image_size, args.image_size, 3], tf.float32), training=False)

    print(f"[eval_linear] Loading weights: {args.ckpt}")
    enc.load_weights(args.ckpt)
    print("[loader] encoder weights loaded.")

    Xtr = extract(enc, xtr, args.batch_size, which=args.feature_layer)
    Xte = extract(enc, xte, args.batch_size, which=args.feature_layer)

    # simple standardization helps LR
    mu, sigma = Xtr.mean(axis=0, keepdims=True), Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sigma
    Xte = (Xte - mu) / sigma

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1 if hasattr(os, "cpu_count") else None,
        verbose=0,
    )
    clf.fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, f"{stamp}__lp-{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)

    # features dump for debugging (optional but handy)
    np.savez_compressed(
        os.path.join(run_dir, f"{stamp}__lp-{args.run_name}_feats.npz"),
        Xtr=Xtr.astype(np.float32), ytr=ytr,
        Xte=Xte.astype(np.float32), yte=yte
    )

    if args.save_clf:
        dump(clf, os.path.join(run_dir, "linear_probe.joblib"))

    if args.save_probs:
        probs = clf.predict_proba(Xte).astype(np.float32)
        probs_path = os.path.join(run_dir, "probs_test.npz")
        np.savez_compressed(probs_path, y_true=yte, probs=probs)
        print(f"[eval_linear] Saved probs to: {probs_path}")
        print(f"[eval_linear] Use with: python3 scripts/report_metrics.py --npz {probs_path} --dataset {args.dataset} --run-name {args.run_name}")

    print(f"[eval_linear] Done. Acc={acc:.4f}. Outputs -> {run_dir}")

if __name__ == "__main__":
    main()
