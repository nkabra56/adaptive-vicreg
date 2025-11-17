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

from vicreg_tf import build_encoder  # use same backbone
from vicreg_tf import enable_memory_growth


def _load_cifar(name: str):
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
    """Pick newest path by mtime."""
    if not paths:
        return None
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _resolve_encoder_ckpt(spec: str) -> tuple[str, dict]:
    """
    Resolve user-supplied checkpoint spec to an existing file path and load kwargs.

    Accepts:
      • A file path to *.weights.h5
      • A directory containing weights (prefers vicreg_encoder.weights.h5)
      • A glob pattern

    Returns:
      (resolved_path, load_kwargs)
    """
    p = Path(spec)

    # If spec has wildcards, expand first.
    if any(ch in spec for ch in ["*", "?", "["]):
        hit = _newest(glob.glob(spec))
        if hit:
            return hit, {}

    # If a file path exists already, use it directly.
    if p.is_file():
        return str(p), {}

    # If a directory is given, search common filenames inside it.
    if p.is_dir():
        # Prefer encoder-only weights
        candidates = []
        candidates += glob.glob(str(p / "vicreg_encoder.weights.h5"))
        candidates += glob.glob(str(p / "*encoder*.weights.h5"))
        # Fallback to full trainer weights
        candidates += glob.glob(str(p / "vicreg_full.weights.h5"))
        candidates += glob.glob(str(p / "*.weights.h5"))
        hit = _newest(candidates)
        if hit:
            # If we fell back to full weights, load by name to pick only encoder vars.
            if "full" in os.path.basename(hit) and "encoder" not in os.path.basename(hit):
                return hit, {"by_name": True, "skip_mismatch": True}
            return hit, {}

    # If a non-existing file path was given, try its parent directory.
    parent = p.parent
    if parent.exists():
        candidates = []
        # Try the exact filename user intended (maybe different timestamp folder exists)
        candidates += glob.glob(str(parent / "vicreg_encoder.weights.h5"))
        candidates += glob.glob(str(parent / "*encoder*.weights.h5"))
        candidates += glob.glob(str(parent / "vicreg_full.weights.h5"))
        candidates += glob.glob(str(parent / "*.weights.h5"))
        hit = _newest(candidates)
        if hit:
            if "full" in os.path.basename(hit) and "encoder" not in os.path.basename(hit):
                return hit, {"by_name": True, "skip_mismatch": True}
            return hit, {}

    # Final attempt: look under common root 'checkpoints_tf'
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
    return p.parse_args()


def main():
    args = parse_args()
    enable_memory_growth()

    # Data
    (xtr, ytr), (xte, yte), num_classes = _load_cifar(args.dataset)
    ds_train = _make_ds(xtr, ytr, args.image_size, args.batch_size, shuffle=True)
    ds_test  = _make_ds(xte, yte, args.image_size, args.batch_size, shuffle=False)

    # Frozen encoder + linear head
    enc = build_encoder(args.image_size, feat_dim=args.feat_dim)

    # Robust checkpoint resolution + load
    ckpt_path, load_kw = _resolve_encoder_ckpt(args.encoder_ckpt)
    print(f"[linear-eval] Loading encoder weights from: {ckpt_path}")
    try:
        enc.load_weights(ckpt_path, **load_kw)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load weights from {ckpt_path} with kwargs {load_kw}. "
            f"If this was a full trainer checkpoint, we try by_name+skip_mismatch automatically. "
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

    opt = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    model.fit(ds_train, epochs=args.epochs, validation_data=ds_test, verbose=2)
    test_metrics = model.evaluate(ds_test, verbose=0)
    test_acc = float(test_metrics[1])

    # CSV row for report_metrics.py
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ["method","dataset","epochs","batch","image_size","feat_dim","lr","l2","test_acc"]
    row = [
        args.method_name,
        args.dataset,
        args.epochs,
        args.batch_size,
        args.image_size,
        args.feat_dim,
        args.lr,
        args.l2,
        f"{test_acc:.4f}",
    ]
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"[linear-eval] test_acc={test_acc:.4f} -> {args.out_csv}")


if __name__ == "__main__":
    main()