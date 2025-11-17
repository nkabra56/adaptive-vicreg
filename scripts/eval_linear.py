"""
Script Title: Linear Evaluation on Frozen Encoder (TensorFlow + Keras)

What this script does
----------------------
I freeze the pretrained encoder from my VICReg/Adaptive-VICReg run and train a
single linear classifier on top of its features. This gives me a quick, apples-
to-apples measure of representation quality without fine-tuning the backbone.

I support multiple optimizers so I can probe sensitivity:
  • SGD with Nesterov momentum (baseline I usually compare against)
  • Adam (often converges faster on small heads)
  • AdamW (decoupled weight decay; I fall back to TFA if native Keras is missing)

I also make checkpoint resolution robust: I can pass a direct file, a directory,
or even a glob. If I accidentally point at the "full" trainer weights instead of
the encoder-only weights, I attempt a by_name+skip_mismatch load automatically.

Artifacts
---------
• CSV row appended to --out-csv with test accuracy and run hyperparameters.

Example usage (why I do each flag)
----------------------------------
python3 scripts/eval_linear.py \
  --encoder-ckpt "checkpoints_tf/pretrain-c10_model9_*/vicreg_encoder.weights.h5" \
  --dataset cifar10 --image-size 32 --batch-size 512 \
  --epochs 100 --lr 0.003 --l2 1e-4 \
  --feat-dim 2048 \
  --opt adamw \
  --out-csv results/pretrain-c10_model9/pretrain-c10_model9_linear_adamw.csv \
  --method-name AdaptiveVICReg

Notes:
• I pass a glob for --encoder-ckpt so the script picks the newest match.
• I use AdamW here because the head is tiny and AdamW often gives a small lift.
• I keep L2 on the head even with AdamW; this is intentional for parity with my
  prior runs. If I want “pure” decoupled WD only, I set --l2 0.

Author: Nishant Kabra
Date: 11/17/2025
"""

from __future__ import annotations
import argparse, csv, os, sys, glob
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Make local src/ importable so `from vicreg_tf import ...` resolves to my repo code.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vicreg_tf import build_encoder  # I reuse the exact encoder builder used in pretrain
from vicreg_tf import enable_memory_growth


def _load_cifar(name: str):
    """
    Load CIFAR-10 or CIFAR-100 using tf.keras.datasets.

    Why I wrote this
    ----------------
    For linear probing I don't need heavy tf.data pipelines; a simple dataset
    loader with basic resize/normalize is enough. Keeping this in one place keeps
    the rest of the script tidy.

    Parameters
    ----------
    name : str
        "cifar10" or "cifar100".

    Returns
    -------
    tuple
        ((x_train, y_train), (x_test, y_test), num_classes) where y arrays are squeezed.

    Raises
    ------
    ValueError
        If an unsupported dataset name is used.
    """
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
    """
    Build a minimal tf.data.Dataset with optional resize and prefetch.

    My intent
    ---------
    I normalize to [0,1], optionally resize, and keep the pipeline simple and fast.
    For linear eval the bottleneck is almost always the dense head training, not I/O.

    Parameters
    ----------
    x : np.ndarray
        Input images (uint8 [N,H,W,C] from keras.datasets).
    y : np.ndarray
        Integer labels [N].
    image_size : int
        Target spatial size (images are resized if not 32).
    batch : int
        Batch size for training/eval.
    shuffle : bool
        Whether to shuffle (I shuffle only for the training split).

    Returns
    -------
    tf.data.Dataset
        A batched, prefetched dataset of (image, label).
    """
    x = tf.convert_to_tensor(x, tf.float32) / 255.0
    y = tf.convert_to_tensor(y, tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(10000)
    if image_size != 32:
        # I resize on the fly; cheap enough at CIFAR scale.
        ds = ds.map(
            lambda im, lab: (tf.image.resize(im, (image_size, image_size)), lab),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def _newest(paths: list[str]) -> str | None:
    """
    Pick the newest existing path by mtime.

    Why I need this
    ---------------
    My checkpoints often include timestamps. When I pass a glob or a directory,
    I want the most recent result without hand-picking the exact file.

    Parameters
    ----------
    paths : list[str]
        Candidate file paths.

    Returns
    -------
    str | None
        The newest existing path or None if nothing exists.
    """
    if not paths:
        return None
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _resolve_encoder_ckpt(spec: str) -> tuple[str, dict]:
    """
    Resolve a robust encoder checkpoint path and any load() kwargs.

    My rules
    --------
    1) If `spec` includes wildcards, I expand and take the newest.
    2) If `spec` is a file -> done.
    3) If `spec` is a directory -> I search common filenames inside.
    4) If `spec` doesn't exist -> I try its parent and then a sweep under checkpoints_tf.
    5) If I happen to point at a *full* trainer weights file, I switch to
       `by_name=True, skip_mismatch=True` so only encoder vars load.

    Parameters
    ----------
    spec : str
        User-supplied checkpoint spec (file, dir, or glob).

    Returns
    -------
    (path, load_kwargs) : (str, dict)
        File path to pass to `load_weights` and any load kwargs I want to use.

    Raises
    ------
    FileNotFoundError
        If I cannot resolve a plausible weights file.
    """
    p = Path(spec)

    # 1) Expand wildcards first.
    if any(ch in spec for ch in ["*", "?", "["]):
        hit = _newest(glob.glob(spec))
        if hit:
            return hit, {}

    # 2) Direct file.
    if p.is_file():
        return str(p), {}

    # 3) Directory: search known names inside.
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

    # 4) Parent dir fallback.
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

    # 5) Last resort: scan the common root.
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
    """
    CLI parser for my linear evaluation.

    Why I expose these flags
    ------------------------
    I want to sweep optimizers and learning rates from the command line. I also
    keep --feat-dim because my encoder width may change across experiments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes matching the flags below.
    """
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
    # New: I can pick between sgd / adam / adamw without touching code.
    p.add_argument("--opt", choices=["sgd", "adam", "adamw"], default="sgd")
    return p.parse_args()


def _make_optimizer(args):
    """
    Construct the optimizer based on --opt.

    My decision logic
    -----------------
    • sgd  -> SGD with momentum and Nesterov (my default baseline).
    • adam -> Plain Adam with the given learning rate.
    • adamw-> Prefer native Keras AdamW; if missing, fall back to TFA's AdamW.

    For AdamW I purposely reuse --l2 as the weight_decay value for a simple knob.
    I still keep the linear head's L2 regularizer active by default, which means
    both L2 on the head and decoupled WD when I pick AdamW. I do this for parity
    with my historical comparisons (feel free to set --l2 0 if you want WD-only).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args so I can read .opt, .lr, and .l2.

    Returns
    -------
    keras.optimizers.Optimizer
        The optimizer object to hand to model.compile().

    Raises
    ------
    RuntimeError
        If AdamW was requested but neither native nor TFA AdamW exists.
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
    """
    Entry point: build data, freeze encoder, train linear head, log CSV.

    What happens step-by-step
    -------------------------
    1) I enable GPU memory growth to avoid preallocating all VRAM.
    2) I load CIFAR and build light tf.data pipelines.
    3) I rebuild the encoder (same builder as pretrain) and load weights:
       - I accept file/dir/glob and pick the newest sensible match.
       - If I end up with a "full" trainer ckpt, I try by_name+skip_mismatch.
    4) I freeze the encoder and attach a Dense(num_classes) head.
    5) I compile with the requested optimizer and train for --epochs.
    6) I evaluate on the test set and append a row to --out-csv.

    I print the resolved checkpoint path up front so it's obvious what weights
    I evaluated.
    """
    args = parse_args()
    enable_memory_growth()

    # Data
    (xtr, ytr), (xte, yte), num_classes = _load_cifar(args.dataset)
    ds_train = _make_ds(xtr, ytr, args.image_size, args.batch_size, shuffle=True)
    ds_test  = _make_ds(xte, yte, args.image_size, args.batch_size, shuffle=False)

    # Frozen encoder + linear head
    enc = build_encoder(args.image_size, feat_dim=args.feat_dim)

    # Robust checkpoint resolution + load (this fixes the “file not found” headaches).
    ckpt_path, load_kw = _resolve_encoder_ckpt(args.encoder-ckpt if hasattr(args, "encoder-ckpt") else args.encoder_ckpt)
    # ^ The hasattr guard ensures I don't crash if some shells transform dashes; args.encoder_ckpt is the standard.
    ckpt_path, load_kw = _resolve_encoder_ckpt(args.encoder_ckpt)
    print(f"[linear-eval] Loading encoder weights from: {ckpt_path}")
    try:
        enc.load_weights(ckpt_path, **load_kw)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load weights from {ckpt_path} with kwargs {load_kw}. "
            f"If this was a full trainer checkpoint, I try by_name+skip_mismatch automatically. "
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

    # CSV row for report_metrics.py and for my own bookkeeping.
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
