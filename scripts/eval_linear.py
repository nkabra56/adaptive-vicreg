"""Linear evaluation of a frozen encoder on CIFAR-10 or CIFAR-100.

Freezes a pretrained encoder, trains one linear layer on its features, and appends a CSV row with the
test accuracy and hyperparameters. `--encoder-ckpt` can be an encoder weights file, a directory, or a
glob (the newest match wins).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import tensorflow as tf
from tensorflow import keras

from vicreg_tf import build_encoder, enable_memory_growth

_WEIGHT_PATTERNS = ("vicreg_encoder.weights.h5", "*encoder*.weights.h5", "vicreg_full.weights.h5", "*.weights.h5")


def _load_cifar(name: str):
    """Return ((x_train, y_train), (x_test, y_test), num_classes) for CIFAR-10 or CIFAR-100."""
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
    """Batched dataset scaled to [0, 1], resized if `image_size` isn't 32."""
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
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def _newest(paths: list[str]) -> str | None:
    """The most recently modified path that exists, or None."""
    existing = [p for p in paths if os.path.exists(p)]
    return max(existing, key=os.path.getmtime) if existing else None


def _search(root: Path, recursive: bool = False) -> tuple[str, dict] | None:
    """Find the newest weights file under `root`, with the `load_weights` kwargs to use for it.

    A file named like a full trainer checkpoint gets `by_name` and `skip_mismatch`.
    """
    candidates: list[str] = []
    for pattern in _WEIGHT_PATTERNS:
        pattern_path = str(root / "**" / pattern) if recursive else str(root / pattern)
        candidates += glob.glob(pattern_path, recursive=recursive)
    hit = _newest(candidates)
    if hit is None:
        return None
    name = os.path.basename(hit)
    if "full" in name and "encoder" not in name:
        return hit, {"by_name": True, "skip_mismatch": True}
    return hit, {}


def _resolve_encoder_ckpt(spec: str) -> tuple[str, dict]:
    """Resolve `spec` to a weights file and the kwargs to pass to `load_weights`.

    Tried in order: a glob (newest match), an existing file, the weights files in a directory, the weights
    files next to `spec`, then any weights file under checkpoints_tf/.

    Raises:
        FileNotFoundError: If nothing matches.
    """
    p = Path(spec)

    if any(ch in spec for ch in "*?["):
        hit = _newest(glob.glob(spec))
        if hit:
            return hit, {}

    if p.is_file():
        return str(p), {}

    if p.is_dir():
        found = _search(p)
        if found:
            return found

    if p.parent.exists():
        found = _search(p.parent)
        if found:
            return found

    root = _REPO / "checkpoints_tf"
    if root.exists():
        found = _search(root, recursive=True)
        if found:
            return found

    raise FileNotFoundError(
        f"Could not resolve encoder checkpoint from spec: {spec}\n"
        "Tried the path itself, its directory, its parent directory and checkpoints_tf/."
    )


def parse_args():
    p = argparse.ArgumentParser(description="Linear evaluation of a frozen encoder.")
    p.add_argument("--encoder-ckpt", type=str, required=True, help="Encoder weights file, directory or glob.")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--image-size", type=int, default=32, help="Input size after resizing.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40, help="Epochs to train the linear layer.")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=0.0,
                   help="L2 penalty on the linear layer. With --opt adamw it is also the weight decay.")
    p.add_argument("--feat-dim", type=int, default=2048, help="Encoder feature width (must match training).")
    p.add_argument("--out-csv", type=str, required=True, help="CSV to append a result row to.")
    p.add_argument("--method-name", type=str, default="VICReg", help="Label written to the CSV.")
    p.add_argument("--opt", choices=["sgd", "adam", "adamw"], default="sgd", help="Optimizer for the linear layer.")
    return p.parse_args()


def _make_optimizer(args):
    if args.opt == "sgd":
        return keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    if args.opt == "adam":
        return keras.optimizers.Adam(learning_rate=args.lr)
    return keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.l2)


def main():
    args = parse_args()
    enable_memory_growth()

    (xtr, ytr), (xte, yte), num_classes = _load_cifar(args.dataset)
    ds_train = _make_ds(xtr, ytr, args.image_size, args.batch_size, shuffle=True)
    ds_test = _make_ds(xte, yte, args.image_size, args.batch_size, shuffle=False)

    enc = build_encoder(args.image_size, feat_dim=args.feat_dim)

    ckpt_path, load_kw = _resolve_encoder_ckpt(args.encoder_ckpt)
    print(f"[linear-eval] Loading encoder weights from: {ckpt_path}")
    try:
        enc.load_weights(ckpt_path, **load_kw)
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {ckpt_path} with kwargs {load_kw}: {e}") from e

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

    model.compile(
        optimizer=_make_optimizer(args),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    model.fit(ds_train, epochs=args.epochs, validation_data=ds_test, verbose=2)
    test_metrics = model.evaluate(ds_test, verbose=0)
    test_acc = float(test_metrics[1])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ["method", "dataset", "epochs", "batch", "image_size", "feat_dim", "lr", "l2", "opt", "test_acc"]
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
