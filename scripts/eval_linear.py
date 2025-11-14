"""
Linear evaluation (frozen encoder) for VICReg / Adaptive-VICReg checkpoints.

Overview
--------
This script freezes the self-supervised encoder you trained and learns a single
linear classification head on top of its features. This "linear probe" reflects
the linearly-separable quality of the learned representation.

Key design choices
------------------
1) **Device selection before TF import**:
   We parse --device early (cpu|gpu|auto) and set CUDA visibility accordingly to
   avoid creating GPU context accidentally, which would make later switching
   impossible.

2) **Checkpoint loading via a tiny loader model**:
   During pretraining, you saved weights with layer names for both "encoder"
   and "projector". To reliably reload those weights, we rebuild *both*
   submodules with the *same names* and create a minimal model:
     input -> encoder -> projector
   We then call `tiny.load_weights(h5_path)`. After loading, we only keep the
   `encoder` to extract features for linear probing. This avoids the common
   KerasTensor / symbolic issues from mixing raw tf.* ops into the Functional
   graph.

3) **Safety around dtype/mixed-precision**:
   We run in float32 to remove dtype mismatch surprises. (If you want AMP, make
   sure your losses and metrics can handle bf16/fp16 cleanly.)

Outputs
-------
- Trains a linear head and prints test accuracy.
- Saves a JSON summary under `results/linear_eval_*.json` for later plotting.

Author: Nishant Kabra
Date: 11/10/25
"""
from __future__ import annotations
import os
import json
import time
import argparse
from typing import Tuple

# -------------------- device pre-parse (before TF import!) --------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument(
    "--device",
    choices=["auto", "cpu", "gpu"],
    default="auto",
    help="Where to run the computation. Decided before importing TensorFlow.",
)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    # Hide all CUDA devices so TF never initializes a GPU context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Reduce TF verbosity a bit (set to '0' for full logs during debugging).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# ------------------------------- TensorFlow -----------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------- utilities -----------------------------------
def _enable_mem_growth():
    """
    Enable per-GPU memory growth to prevent TF from pre-allocating all VRAM.

    Why: On laptops or shared GPUs this avoids OOM at startup. No-op on CPU.
    """
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("[eval_linear] set_memory_growth warning:", repr(e))


def _gpu_probe() -> bool:
    """
    Try a tiny GPU op to verify kernels can be loaded successfully.

    Returns
    -------
    bool
        True if a trivial Conv2D + sum executes on GPU; False otherwise.

    Notes
    -----
    - This catches issues like invalid PTX, missing kernels, or driver
      mismatches early and lets us fall back to CPU automatically.
    """
    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform([1, 8, 8, 3])
            y = layers.Conv2D(4, 3, padding="same")(x)
            _ = tf.reduce_sum(y).numpy()  # forces execution
        print("[eval_linear] GPU probe OK.")
        return True
    except Exception as e:
        print("[eval_linear] GPU probe FAILED:", repr(e))
        return False


def decide_device(policy: str) -> str:
    """
    Decide runtime device string from the requested policy.

    Parameters
    ----------
    policy : {'cpu','gpu','auto'}
        'cpu' : force CPU (/CPU:0)
        'gpu' : force GPU (/GPU:0) — may raise if kernels/driver mismatch
        'auto': prefer GPU if a small probe succeeds, else CPU

    Returns
    -------
    str
        TensorFlow device string like '/CPU:0' or '/GPU:0'.
    """
    if policy == "cpu":
        print("[eval_linear] --device cpu -> using /CPU:0")
        return "/CPU:0"
    _enable_mem_growth()
    if policy == "gpu":
        print("[eval_linear] --device gpu requested; using /GPU:0 (may error).")
        return "/GPU:0"
    # auto
    return "/GPU:0" if _gpu_probe() else "/CPU:0"


# ------------------------ data pipelines (CIFAR) ------------------------------
def _norm_img(x: tf.Tensor) -> tf.Tensor:
    """
    Cast uint8 image to float32 in [0,1].

    Parameters
    ----------
    x : tf.Tensor
        [H, W, 3] uint8

    Returns
    -------
    tf.Tensor
        float32 in [0,1]
    """
    return tf.image.convert_image_dtype(x, tf.float32)


def build_cifar10_sup(image_size: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Build supervised train/test pipelines for CIFAR-10 (resized to `image_size`).

    Returns
    -------
    train_ds : tf.data.Dataset
        (image, label) batches for training the linear head
    test_ds : tf.data.Dataset
        (image, label) batches for evaluation
    num_classes : int
        10 for CIFAR-10
    n_train : int
        Number of training images
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    num_classes = 10

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Resize + cast + pack batches
    train = (
        train.shuffle(10_000, reshuffle_each_iteration=True)
        .map(
            lambda x, y: (
                tf.image.resize(_norm_img(x), [image_size, image_size]),
                tf.cast(y, tf.int32),
            ),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test = (
        test.map(
            lambda x, y: (
                tf.image.resize(_norm_img(x), [image_size, image_size]),
                tf.cast(y, tf.int32),
            ),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train, test, num_classes, x_train.shape[0]


def build_cifar100_sup(image_size: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Same as `build_cifar10_sup` but for CIFAR-100.

    Returns
    -------
    train_ds, test_ds, num_classes (100), n_train
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    num_classes = 100

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train = (
        train.shuffle(10_000, reshuffle_each_iteration=True)
        .map(
            lambda x, y: (
                tf.image.resize(_norm_img(x), [image_size, image_size]),
                tf.cast(y, tf.int32),
            ),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test = (
        test.map(
            lambda x, y: (
                tf.image.resize(_norm_img(x), [image_size, image_size]),
                tf.cast(y, tf.int32),
            ),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train, test, num_classes, x_train.shape[0]


# ------------------- encoder / projector (match trainer) ----------------------
def conv_block(x: tf.Tensor, filters: int, k: int = 3, s: int = 1) -> tf.Tensor:
    """
    A simple Conv2D -> BatchNorm -> ReLU block.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.
    filters : int
        Number of output channels.
    k : int
        Kernel size.
    s : int
        Stride.

    Returns
    -------
    tf.Tensor
        Output feature map.

    Notes
    -----
    - We avoid using raw tf.* ops on symbolic KerasTensors outside layers.
    """
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_encoder(image_size: int) -> keras.Model:
    """
    Tiny CNN encoder producing a 2048-dim feature vector.

    Parameters
    ----------
    image_size : int
        Input resolution (height==width).

    Returns
    -------
    keras.Model
        Keras Functional model named 'encoder' with output Dense(2048) named 'feat'.

    Important
    ---------
    The layer names ('encoder', 'feat', etc.) match those used during
    pretraining so `load_weights()` can locate variables by name.
    """
    inp = keras.Input(shape=(image_size, image_size, 3))
    x = conv_block(inp, 64)
    x = conv_block(x, 64, s=2)
    x = conv_block(x, 128)
    x = conv_block(x, 128, s=2)
    x = conv_block(x, 256)
    x = conv_block(x, 256, s=2)
    x = conv_block(x, 512)
    x = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(2048, name="feat")(x)  # final embeddings used by the probe
    return keras.Model(inp, feat, name="encoder")


def build_projector(in_dim: int, out_dim: int, num_layers: int) -> keras.Model:
    """
    MLP projector used during pretraining.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (encoder output), e.g., 2048.
    out_dim : int
        Projection output dimension (e.g., 8192).
    num_layers : int
        Number of MLP layers (including the output). Must be >= 1.

    Returns
    -------
    keras.Model
        Model named 'projector' with appropriately named Dense/BN/ReLU layers.

    Why rebuild here?
    -----------------
    We reconstruct it only so that the tiny loader model has the same variable
    names as your training graph; that allows `load_weights` to restore all
    moving statistics in BatchNorm and Dense kernels.
    """
    assert num_layers >= 1, "num_layers must be >= 1"
    inp = keras.Input(shape=(in_dim,))
    x = inp
    hidden = max(2048, out_dim)  # reasonable default width
    # stack L-1 (Dense+BN+ReLU), then final Dense
    for i in range(num_layers - 1):
        x = layers.Dense(hidden, use_bias=False, name=f"proj_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"proj_bn_{i}")(x)
        x = layers.ReLU(name=f"proj_relu_{i}")(x)
    out = layers.Dense(out_dim, name="proj_out")(x)
    return keras.Model(inp, out, name="projector")


def load_encoder_from_ckpt(ckpt: str, image_size: int, proj_out: int, proj_layers: int) -> keras.Model:
    """
    Load encoder weights by building a tiny model: input -> encoder -> projector.

    Parameters
    ----------
    ckpt : str
        Path to HDF5 weights saved during pretraining (save_weights_only).
    image_size : int
        Resolution the encoder expects.
    proj_out : int
        Projection size used during pretrain. Must match the checkpoint.
    proj_layers : int
        Projector depth used during pretrain. Must match the checkpoint.

    Returns
    -------
    keras.Model
        The encoder model with weights restored.

    Why this approach?
    ------------------
    Keras weight loading matches by layer *name* and variable *name*. The
    pretrainer saved weights from a model that contained *both* encoder and
    projector. Rebuilding the same structure ensures all variables exist so
    `load_weights()` can place parameters correctly. After loading, we discard
    the tiny wrapper and return just the encoder.
    """
    encoder = build_encoder(image_size)
    projector = build_projector(2048, proj_out, proj_layers)

    # Build a tiny wrapper with the same submodule names.
    inp = keras.Input(shape=(image_size, image_size, 3))
    z = projector(encoder(inp))
    tiny = keras.Model(inp, z, name="tiny_pretrain_model")

    # Create variables (build) before loading weights.
    _ = tiny(tf.zeros([1, image_size, image_size, 3]), training=False)
    print(f"[eval_linear] Loading weights: {ckpt}")
    tiny.load_weights(ckpt)
    print("[eval_linear] Weights loaded into encoder/projector.")
    return encoder


# -------------------------- linear probe head ---------------------------------
def build_linear_probe(encoder: keras.Model, num_classes: int) -> keras.Model:
    """
    Compose a frozen encoder with a trainable single Dense head.

    Parameters
    ----------
    encoder : keras.Model
        Pretrained encoder whose weights will be frozen.
    num_classes : int
        Number of classification classes.

    Returns
    -------
    keras.Model
        Model(input=image) -> logits[num_classes]

    Notes
    -----
    - We call encoder with `training=False` to make sure its BN layers (if any)
      run in inference mode during probing, as is standard for linear eval.
    """
    encoder.trainable = False  # freeze backbone
    inp = keras.Input(shape=encoder.input_shape[1:])
    feat = encoder(inp, training=False)
    logits = layers.Dense(num_classes, name="linear_head")(feat)
    return keras.Model(inp, logits, name="linear_eval_model")


# ---------------------------------- CLI --------------------------------------
def parse_args():
    """
    Parse command-line arguments for linear evaluation.

    Returns
    -------
    argparse.Namespace
        All runtime options needed by `main()`.
    """
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", type=str, default="checkpoints_tf/vicreg_tf.weights.h5",
                   help="Path to weights saved by trainer (HDF5).")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--proj-out", type=int, default=8192)
    p.add_argument("--proj-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50, help="Linear head training epochs.")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


# ---------------------------------- main --------------------------------------
def main():
    """
    Entrypoint: load data, restore encoder, train linear head, evaluate, save JSON.

    Side Effects
    ------------
    - Creates `results/linear_eval_*.json`.
    - Prints progress + final accuracy to stdout.
    """
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = decide_device(_pre_args.device)
    print(f"[eval_linear] Using device: {device}")

    with tf.device(device):
        # 1) Build supervised dataset
        if args.dataset == "cifar10":
            train, test, num_classes, n_train = build_cifar10_sup(args.image_size, args.batch_size)
        else:
            train, test, num_classes, n_train = build_cifar100_sup(args.image_size, args.batch_size)
        steps_per_epoch = max(1, n_train // args.batch_size)

        # 2) Load encoder from checkpoint
        encoder = load_encoder_from_ckpt(args.ckpt, args.image_size, args.proj_out, args.proj_layers)

        # 3) Build linear probe and optimizer
        model = build_linear_probe(encoder, num_classes)
        try:
            # AdamW if available; otherwise SGD with momentum is fine for a linear head.
            import tensorflow_addons as tfa
            opt = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.wd)
        except Exception:
            opt = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)

        model.compile(
            optimizer=opt,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )

        # 4) Train linear head
        print(f"[eval_linear] Training linear head for {args.epochs} epochs "
              f"({steps_per_epoch} steps/epoch)...")
        t0 = time.time()
        model.fit(train, epochs=args.epochs, steps_per_epoch=steps_per_epoch, verbose=1)
        dur = time.time() - t0

        # 5) Evaluate
        test_metrics = model.evaluate(test, verbose=0)
        acc = float(test_metrics[1])  # index 1 = 'acc' metric

        # 6) Persist results
        out = {
            "task": "linear_eval",
            "dataset": args.dataset,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "proj_out": args.proj_out,
            "proj_layers": args.proj_layers,
            "epochs": args.epochs,
            "lr": args.lr,
            "wd": args.wd,
            "device": device,
            "test_acc": acc,
            "train_seconds": dur,
            "ckpt": args.ckpt,
        }
        out_path = os.path.join(args.results_dir, f"linear_eval_{args.dataset}_{int(time.time())}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval_linear] Test acc: {acc:.4f}  |  wrote {out_path}")


if __name__ == "__main__":
    main()
