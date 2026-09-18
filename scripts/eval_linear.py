"""
Linear evaluation on a frozen encoder (TensorFlow + Keras).

Freezes a pretrained VICReg/Adaptive-VICReg encoder and trains a single
linear classifier on top of its features, as a quick, apples-to-apples
measure of representation quality without fine-tuning the backbone.

Supports three optimizers for the head: SGD with Nesterov momentum, Adam,
and AdamW (native Keras, falling back to TensorFlow Addons if unavailable).

Checkpoint resolution is robust to a direct file, a directory, or a glob; if
`--encoder-ckpt` accidentally points at the full trainer weights instead of
the encoder-only weights, a by_name+skip_mismatch load is attempted
automatically.

Writes one CSV row (test accuracy plus hyperparameters) to --out-csv.

Example:
  python3 scripts/eval_linear.py \
    --encoder-ckpt checkpoints_tf/pretrain-c10_checktrainer_2*/vicreg_encoder.weights.h5 \
    --dataset cifar10 --image-size 32 --batch-size 512 \
    --epochs 10 --lr 0.003 --l2 1e-4 \
    --feat-dim 2048 \
    --opt adamw \
    --out-csv results/linear_eval.csv \
    --method-name AdaptiveVICReg
"""

from __future__ import annotations
import argparse, csv, os, sys, glob
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Make local src/ importable so `from vicreg_tf import ...` resolves.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vicreg_tf import build_encoder
from vicreg_tf import enable_memory_growth


def _load_cifar(name: str):
    """Load CIFAR-10 or CIFAR-100 via `keras.datasets`, returning ((x_train, y_train), (x_test, y_test), num_classes)."""
    if name == "cifar10":
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        num_classes = 10
    elif name == "cifar100":
        (xtr, ytr), (xte, yte) = keras.datasets.cifar100.load_data(label_mode="fine")
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return (xtr, ytr.squeeze()), (xte, yte.squeeze()), num_classes


def _make_ds(x, y, image_size: int, batch: int, shuffle: bool):
    """Build a minimal batched, prefetched tf.data.Dataset, normalized to [0,1] and resized if `image_size != 32`."""
    x = tf.convert_to_tensor(x, tf.float32) / 255.0
    y = tf.convert_to_tensor(y, tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(10000)
    if image_size != 32:
        ds = ds.map(
            lambda im, lab: (tf.image.resize(im, (image_size, image_size)), lab),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def _newest(paths: list[str]) -> str | None:
    """Return the most recently modified existing path, or None if none exist."""
    if not paths:
        return None
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _resolve_encoder_ckpt(spec: str) -> tuple[str, dict]:
    """
    Resolve `spec` to an encoder checkpoint path plus `load_weights` kwargs.

    Tries, in order: wildcard expansion (newest match); a direct file; known
    filenames inside a directory; the same search under the parent
    directory; then a recursive sweep under checkpoints_tf. If the resolved
    file looks like a full trainer checkpoint rather than encoder-only,
    returns `{"by_name": True, "skip_mismatch": True}` so only the matching
    encoder variables load.

    Raises:
        FileNotFoundError: If no plausible weights file is found.
    """
    p = Path(spec)

    if any(ch in spec for ch in ["*", "?", "["]):
        hit = _newest(glob.glob(spec))
        if hit:
            return hit, {}

    if p.is_file():
        return str(p), {}

    if p.is_dir():
        candidates = []
        candidates += glob.glob(str(p / "vicreg_encoder.weights.h5"))
        candidates += glob.glob(str(p / "*encoder*.weights.h5"))
        candidates += glob.glob(str(p / "vicreg_full.weights.h5"))
        candidates += glob.glob(str(p / "*.weights.h5"))
        hit = _newest(candidates)
        if hit:
            if "full" in os.path.basename(hit) and "encoder" not in os.path.basename(hit):
                return hit, {"by_name": True, "skip_mismatch": True}
            return hit, {}

    parent = p.parent
    if parent.exists():
        candidates = []
        candidates += glob.glob(str(parent / "vicreg_encoder.weights.h5"))
        candidates += glob.glob(str(parent / "*encoder*.weights.h5"))
        candidates += glob.glob(str(parent / "vicreg_full.weights.h5"))
        candidates += glob.glob(str(parent / "*.weights.h5"))
        hit = _newest(candidates)
        if hit:
            if "full" in os.path.basename(hit) and "encoder" not in os.path.basename(hit):
                return hit, {"by_name": True, "skip_mismatch": True}
            return hit, {}

    root = _REPO / "checkpoints_tf"
    if root.exists():
        candidates = []
        candidates += glob.glob(str(root / "**" / "vicreg_encoder.weights.h5"), recursive=True)
        candidates += glob.glob(str(root / "**" / "*encoder*.weights.h5"), recursive=True)
        candidates += glob.glob(str(root / "**" / "vicreg_full.weights.h5"), recursive=True)
        candidates += glob.glob(str(root / "**" / "*.weights.h5"), recursive=True)
        hit = _newest(candidates)
        if hit:
            if "full" in os.path.basename(hit) and "encoder" not in os.path.basename(hit):
                return hit, {"by_name": True, "skip_mismatch": True}
            return hit, {}

    raise FileNotFoundError(
        f"Could not resolve encoder checkpoint from spec: {spec}\n"
        f"Tried direct path, directory search, parent search, and checkpoints_tf fallback."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-ckpt", type=str, required=True)
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--feat-dim", type=int, default=2048)
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--method-name", type=str, default="VICReg")
    p.add_argument("--opt", choices=["sgd", "adam", "adamw"], default="sgd")
    return p.parse_args()


def _make_optimizer(args):
    """
    Build the head optimizer from `args.opt`.

    "adamw" prefers native `keras.optimizers.AdamW`, falling back to
    TensorFlow Addons if unavailable, and reuses `args.l2` as its
    `weight_decay`. Note the linear head's L2 regularizer stays active
    regardless of optimizer, so AdamW runs get both L2 and decoupled weight
    decay unless `--l2 0` is passed.

    Raises:
        RuntimeError: If "adamw" was requested but neither implementation is available.
    """
    if args.opt == "sgd":
        return keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    if args.opt == "adam":
        return keras.optimizers.Adam(learning_rate=args.lr)
    # adamw
    try:
        return keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.l2)
    except Exception:
        try:
            import tensorflow_addons as tfa  # type: ignore
            return tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.l2)
        except Exception as e:
            raise RuntimeError(
                "AdamW requested but neither keras.optimizers.AdamW nor "
                "tensorflow_addons.optimizers.AdamW is available. "
                "Use --opt adam or install TensorFlow Addons."
            ) from e


def main():
    args = parse_args()
    enable_memory_growth()

    (xtr, ytr), (xte, yte), num_classes = _load_cifar(args.dataset)
    ds_train = _make_ds(xtr, ytr, args.image_size, args.batch_size, shuffle=True)
    ds_test  = _make_ds(xte, yte, args.image_size, args.batch_size, shuffle=False)

    enc = build_encoder(args.image_size, feat_dim=args.feat_dim)

    ckpt_path, load_kw = _resolve_encoder_ckpt(args.encoder_ckpt)
    print(f"[linear-eval] Loading encoder weights from: {ckpt_path}")
    try:
        enc.load_weights(ckpt_path, **load_kw)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load weights from {ckpt_path} with kwargs {load_kw}. "
            f"If this was a full trainer checkpoint, by_name+skip_mismatch is tried automatically. "
            f"Original error: {e}"
        ) from e

    enc.trainable = False

    inp = keras.Input(shape=(args.image_size, args.image_size, 3))
    feat = enc(inp, training=False)
    logits = keras.layers.Dense(
        num_classes,
        use_bias=True,
        name="linear_head",
        kernel_regularizer=keras.regularizers.l2(args.l2),
    )(feat)
    model = keras.Model(inp, logits, name="linear_eval")

    opt = _make_optimizer(args)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    model.fit(ds_train, epochs=args.epochs, validation_data=ds_test, verbose=2)
    test_metrics = model.evaluate(ds_test, verbose=0)
    test_acc = float(test_metrics[1])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ["method","dataset","epochs","batch","image_size","feat_dim","lr","l2","opt","test_acc"]
    row = [
        args.method_name,
        args.dataset,
        args.epochs,
        args.batch_size,
        args.image_size,
        args.feat_dim,
        args.lr,
        args.l2,
        args.opt,
        f"{test_acc:.4f}",
    ]
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"[linear-eval] opt={args.opt} test_acc={test_acc:.4f} -> {args.out_csv}")


if __name__ == "__main__":
    main()
