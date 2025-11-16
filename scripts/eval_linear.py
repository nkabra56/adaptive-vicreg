"""
Linear evaluation on top of a frozen encoder trained via (Adaptive) VICReg.

What this script does
---------------------
1) Rebuilds the SAME encoder+projector graph and a minimal
   VICRegTrainer skeleton (name='vicreg_trainer'), **builds variables**,
   then loads weights from --ckpt (exact structural match).
2) Extracts encoder features ("feat" or "gap"), L2-normalizes them.
3) Trains a linear classifier (Dense logits) with SGD+momentum.

Author: Nishant Kabra
Date: 11/16/2025
"""
from __future__ import annotations

import os
import argparse
import numpy as np

# -------------------- device BEFORE importing TF --------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# -------------------- import TF/Keras --------------------
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE
_PIN_CPU = (_pre_args.device == "cpu")


def _gpu_probe() -> bool:
    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform([1, 16, 16, 3])
            y = layers.Conv2D(4, 3, padding="same")(x)
            _ = tf.reduce_sum(y).numpy()
        print("[eval_linear] GPU probe OK.")
        return True
    except Exception as e:
        print("[eval_linear] GPU probe FAILED:", repr(e))
        return False


if _pre_args.device != "cpu":
    if _pre_args.device == "gpu":
        print("[eval_linear] --device gpu requested; not falling back.")
        _PIN_CPU = False
    else:
        _PIN_CPU = not _gpu_probe()
        if _PIN_CPU:
            print("[eval_linear] Pinning to CPU due to probe failure.")
else:
    print("[eval_linear] Forcing CPU.")
    _PIN_CPU = True


# -------------------- data --------------------
def load_dataset(name: str, image_size: int):
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        n_classes = 10
    elif name in {"cifar100", "cifar-100"}:
        (xtr, ytr), (xte, yte) = keras.datasets.cifar100.load_data()
        n_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    xtr = tf.image.resize(tf.cast(xtr, tf.float32) / 255.0, (image_size, image_size)).numpy()
    xte = tf.image.resize(tf.cast(xte, tf.float32) / 255.0, (image_size, image_size)).numpy()
    ytr = ytr.reshape(-1).astype(np.int32)
    yte = yte.reshape(-1).astype(np.int32)
    return (xtr, ytr), (xte, yte), n_classes


# -------------------- encoder/projector (match training) --------------------
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


class VICRegTrainerSkeleton(keras.Model):
    def __init__(self, encoder: keras.Model, projector: keras.Model):
        super().__init__(name="vicreg_trainer")
        self.encoder = encoder
        self.projector = projector

    def call(self, inputs, training=None):
        x1, x2 = inputs
        f1 = self.encoder(x1, training=training)
        f2 = self.encoder(x2, training=training)
        z1 = self.projector(f1, training=training)
        z2 = self.projector(f2, training=training)
        return z1, z2


def load_ckpt_into_skeleton(encoder, projector, ckpt_path: str, image_size: int):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    trainer = VICRegTrainerSkeleton(encoder, projector)
    dummy = tf.zeros([1, image_size, image_size, 3], tf.float32)
    _ = trainer((dummy, dummy), training=False)
    print(f"[eval_linear] Loading weights: {ckpt_path}")
    trainer.load_weights(ckpt_path)
    print("[eval_linear] Weights loaded into trainer skeleton.")


# -------------------- features & probe --------------------
def build_feature_fn(encoder: keras.Model, feature_layer: str):
    if feature_layer == "feat":
        fwd = keras.Model(encoder.input, encoder.output)
    elif feature_layer == "gap":
        fwd = keras.Model(encoder.input, encoder.get_layer("gap").output)
    else:
        raise ValueError("--feature-layer must be 'feat' or 'gap'")

    @tf.function
    def _feat_fn(x):
        z = fwd(x, training=False)
        z = tf.nn.l2_normalize(z, axis=-1)
        return z
    return _feat_fn


def extract_in_batches(x: np.ndarray, batch_size: int, fn):
    feats = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        z = fn(tf.convert_to_tensor(xb, dtype=tf.float32))
        feats.append(z.numpy())
    return np.concatenate(feats, axis=0)


def build_linear_head(in_dim: int, n_classes: int) -> keras.Model:
    inp = keras.Input(shape=(in_dim,))
    logits = layers.Dense(n_classes, use_bias=True, name="linear_head")(inp)
    return keras.Model(inp, logits, name="linear_probe")


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--proj-out", type=int, default=8192)
    p.add_argument("--proj-layers", type=int, default=3)
    p.add_argument("--feature-layer", choices=["feat", "gap"], default="feat")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--run-name", type=str, default="lp")
    p.add_argument("--save-clf", action="store_true")
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "/CPU:0" if _PIN_CPU else "/GPU:0"
    print(f"[eval_linear] Using device {device}")

    (xtr, ytr), (xte, yte), n_classes = load_dataset(args.dataset, args.image_size)

    if args.augment:
        def aug_batch(xb):
            x = tf.convert_to_tensor(xb, tf.float32)
            x = tf.image.resize_with_crop_or_pad(x, args.image_size + 8, args.image_size + 8)
            x = tf.image.random_crop(x, [tf.shape(x)[0], args.image_size, args.image_size, 3])
            x = tf.image.random_flip_left_right(x)
            return x.numpy()
    else:
        aug_batch = lambda xb: xb

    with tf.device(device):
        encoder = build_encoder(args.image_size)
        projector = build_projector(2048, args.proj_out, args.proj_layers)
        load_ckpt_into_skeleton(encoder, projector, args.ckpt, args.image_size)
        feat_fn = build_feature_fn(encoder, args.feature_layer)

        print("[eval_linear] Extracting train features...")
        ztr = []
        for i in range(0, len(xtr), args.batch_size):
            xb = aug_batch(xtr[i:i + args.batch_size])
            ztr.append(feat_fn(tf.convert_to_tensor(xb, tf.float32)).numpy())
        ztr = np.concatenate(ztr, axis=0)

        print("[eval_linear] Extracting test features...")
        zte = extract_in_batches(xte, args.batch_size, feat_fn)

        in_dim = ztr.shape[1]
        head = build_linear_head(in_dim, n_classes)
        opt = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, clipnorm=1.0)
        head.compile(
            optimizer=opt,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        ds_tr = tf.data.Dataset.from_tensor_slices((ztr, ytr)).shuffle(10000).batch(1024).prefetch(AUTOTUNE)
        ds_te = tf.data.Dataset.from_tensor_slices((zte, yte)).batch(1024).prefetch(AUTOTUNE)

        print(f"[eval_linear] Training linear probe for {args.epochs} epochs...")
        hist = head.fit(ds_tr, validation_data=ds_te, epochs=args.epochs, verbose=1)

        loss, acc = head.evaluate(ds_te, verbose=0)
        print(f"[eval_linear] Done. Acc={acc:.4f}")

        base = os.path.join(args.out_dir, f"lp_{args.run_name}_{args.feature_layer}")
        np.savez_compressed(base + "_history.npz", **{k: np.array(v) for k, v in hist.history.items()})
        if args.save_clf:
            head.save_weights(base + "_linear_head.weights.h5")

        if args.save_plots:
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(hist.history.get("accuracy", []), label="train")
                if "val_accuracy" in hist.history:
                    plt.plot(hist.history["val_accuracy"], label="val")
                plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
                plt.savefig(base + "_acc.png", dpi=150)
                plt.close()
            except Exception as e:
                print("[eval_linear] Plot save failed:", repr(e))


if __name__ == "__main__":
    main()
