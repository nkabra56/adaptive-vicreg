"""
KNN evaluation on CIFAR10/100 features extracted by the trained encoder.

Key changes
-----------
- Uses your named encoder: conv2d...conv2d_6, gap, dense, feat
- Robust weight loading for encoder-only .h5 (builds vars, then load)
- Saves standardized metrics file:
    <out_dir>/<STAMP>__knn-<run_name>/probs_test.npz
  which contains: y_true (int64), probs (float32, N x C)
- Also saves feature dumps: *_feats.npz for debugging
- Prints the exact saved path so you can feed it to report_metrics.py

Usage
-----
python3 scripts/knn_eval.py --device cpu \
  --ckpt checkpoints_tf/.../vicreg_encoder.weights.h5 \
  --dataset cifar10 --image-size 32 \
  --batch-size 1024 --k 200 --T 0.07 \
  --feature-layer feat \
  --out-dir results --run-name knn-feat_c10_model1 --save-probs
"""

import os, argparse, time, datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Device setup ----------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["auto", "gpu", "cpu"], default="cpu")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# ---------------- Encoder (named exactly like training) ----------------
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

# ---------------- Data ----------------
def load_cifar(name: str):
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        num_classes = 10
    elif name in {"cifar100", "cifar-100"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar100.load_data()
        num_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    ytr = ytr.reshape(-1).astype(np.int64)
    yte = yte.reshape(-1).astype(np.int64)
    return (xtr, ytr), (xte, yte), num_classes

def make_ds(x, bs):
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- Feature extraction ----------------
def extract_features(encoder, x, bs, which: str):
    ds = make_ds(x, bs)
    outs = []
    for batch in ds:
        batch = tf.image.convert_image_dtype(batch, tf.float32)
        feats = encoder(batch, training=False)
        if which == "gap":
            # Take 'gap' output by tapping the layer
            gap_layer = encoder.get_layer("gap")
            # Re-run just to be exact (cheap)
            x = encoder.get_layer("conv2d").input
            sub = keras.Model(x, gap_layer.output)
            feats = sub(batch, training=False)
        outs.append(feats.numpy())
    return np.concatenate(outs, axis=0)

# ---------------- Soft kNN ----------------
def soft_knn_probs(ftr, ytr, fte, k=200, T=0.07, chunk=256):
    # L2 normalize
    def l2n(a):
        n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
        return a / n
    ftr = l2n(ftr.astype(np.float32))
    fte = l2n(fte.astype(np.float32))

    Ntr, Cdim = ftr.shape
    Nte = fte.shape[0]
    classes = int(np.max(ytr)) + 1
    probs = np.zeros((Nte, classes), dtype=np.float32)

    # chunk over test set to keep memory bounded
    for i in range(0, Nte, chunk):
        te = fte[i:i+chunk]                           # (b, d)
        sim = te @ ftr.T                              # (b, Ntr) cosine sim
        idx = np.argpartition(sim, -k, axis=1)[:, -k:]         # (b, k) indices
        part = np.take_along_axis(sim, idx, axis=1)            # (b, k)
        # sort top-k descending (optional)
        ord = np.argsort(-part, axis=1)
        topk_idx = np.take_along_axis(idx, ord, axis=1)        # (b, k)
        topk_sim = np.take_along_axis(part, ord, axis=1)       # (b, k)
        weights = np.exp(topk_sim / float(T))                   # (b, k)
        yk = ytr[topk_idx]                                      # (b, k)

        # accumulate class votes
        for c in range(classes):
            probs[i:i+te.shape[0], c] = (weights * (yk == c)).sum(axis=1)

    # normalize rows to sum=1
    s = probs.sum(axis=1, keepdims=True) + 1e-12
    probs /= s
    return probs

# ---------------- CLI + main ----------------
def parse_args():
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--T", type=float, default=0.07)
    p.add_argument("--feature-layer", choices=["feat", "gap"], default="feat")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--run-name", type=str, default="knn")
    p.add_argument("--save-probs", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    print("[knn_eval] Forcing CPU.")
    print("[knn_eval] Using device /CPU:0")

    (xtr, ytr), (xte, yte), num_classes = load_cifar(args.dataset)
    enc = build_encoder(args.image_size)

    # build vars, then load weights
    _ = enc(tf.zeros([1, args.image_size, args.image_size, 3], tf.float32), training=False)
    print(f"[knn_eval] Loading weights: {args.ckpt}")
    enc.load_weights(args.ckpt)
    print("[loader] encoder weights loaded.")

    # features
    ftr = extract_features(enc, xtr, args.batch_size, which=args.feature_layer)
    fte = extract_features(enc, xte, args.batch_size, which=args.feature_layer)

    # save features dump for debugging
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, f"{stamp}__knn-{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)
    feats_path = os.path.join(run_dir, f"{stamp}__knn-{args.run_name}_feats.npz")
    np.savez_compressed(feats_path, ftr=ftr, ytr=ytr, fte=fte, yte=yte)
    # probs
    probs = soft_knn_probs(ftr, ytr, fte, k=args.k, T=args.T, chunk=256)

    pred = probs.argmax(axis=1)
    acc = float((pred == yte).mean())
    print(f"[knn_eval] Done. Acc={acc:.4f}.")

    if args.save_probs:
        probs_path = os.path.join(run_dir, "probs_test.npz")
        np.savez_compressed(probs_path, y_true=yte, probs=probs)
        print(f"[knn_eval] Saved probs to: {probs_path}")
        print(f"[knn_eval] Use with: python3 scripts/report_metrics.py --npz {probs_path} --dataset {args.dataset} --run-name {args.run_name}")
    else:
        # also save legacy single file for convenience
        legacy = os.path.join(run_dir, f"{stamp}__knn-{args.run_name}.npz")
        np.savez_compressed(legacy, y_true=yte, probs=probs)
        print(f"[knn_eval] Saved (legacy) to: {legacy}")

if __name__ == "__main__":
    main()
