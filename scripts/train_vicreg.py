"""
VICReg / Adaptive VICReg trainer (TensorFlow + Keras)

Overview
--------
- Decides device BEFORE importing TensorFlow:
    --device cpu  -> disables GPU entirely (sets CUDA_VISIBLE_DEVICES="")
    --device gpu  -> forces GPU, errors if kernels aren't compatible
    --device auto -> tries a tiny GPU probe; on failure, pins training to CPU

- If GPU probe fails (common on very new GPUs with older TF builds), we DO NOT
  try to "hide" GPUs after TF has initialized. Instead we place the whole
  model/training under tf.device('/CPU:0') so the run proceeds reliably.

- Loss is computed in float32 internally (safe if you later turn on bf16), and
  all schedules are implemented in pure Python (no .numpy()) so they work in
  graph mode.

- IMPORTANT: We *force-build* the sublayers and mark the subclassed Keras Model
  as built (trainer.built = True) before training so ModelCheckpoint with
  save_weights_only=True will work.

What you get
------------
- The classic Keras `.fit()` progress bar (per-batch, per-epoch) you asked for.
- Stable, evaluation-friendly layer names:
  encoder: conv2d, conv2d_1, ..., gap, dense, feat
  proj   : dense, batch_normalization, dense_1, batch_normalization_1, dense_2
- Flexible output:
  * --ckpt-out  : write a single full-model weights file here (plus encoder-only)
  * --model-dir : otherwise, create a timestamped run folder (optionally --run-name)

Author: Nishant Kabra
Date: 11/15/2025
"""
from __future__ import annotations

import os
import math
import json
import argparse
import datetime as _dt
from dataclasses import dataclass
from typing import Tuple

# -----------------------------------------------------------------------------
# 1) Parse only the device flag BEFORE importing TensorFlow
# -----------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument(
    "--device", choices=["auto", "gpu", "cpu"], default="auto",
    help="Device placement: auto|gpu|cpu (default: auto)"
)
_pre_args, _ = _pre.parse_known_args()

# If user forced CPU, hide GPUs before TF import so TF never touches CUDA
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Reduce TF info spam a little (set 0 for full logs)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# -----------------------------------------------------------------------------
# 2) Now import TensorFlow & friends
# -----------------------------------------------------------------------------
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE

# Global flag: if True, we wrap build/fit inside CPU device scope
PIN_CPU = (_pre_args.device == "cpu")


def _print_devices() -> None:
    """Log visible physical GPU devices (or empty on CPU-only)."""
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[train_vicreg] Visible GPUs: {gpus}")


def _enable_memory_growth() -> None:
    """Enable per-GPU memory growth to avoid pre-allocating all VRAM."""
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("[train_vicreg] set_memory_growth warning:", repr(e))


def _gpu_probe() -> bool:
    """
    Run a tiny Conv2D once on /GPU:0 to validate kernels/driver.
    Returns True if OK, else False (invalid PTX / mismatch etc.).
    """
    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform([1, 16, 16, 3])
            y = layers.Conv2D(4, 3, padding="same")(x)
            _ = tf.reduce_sum(y).numpy()  # materialize a kernel launch
        print("[train_vicreg] GPU probe OK.")
        return True
    except Exception as e:
        print("[train_vicreg] GPU probe FAILED:", repr(e))
        return False


# Decide device policy now
if _pre_args.device != "cpu":
    _print_devices()
    _enable_memory_growth()
    if _pre_args.device == "gpu":
        # Force GPU usage; if it fails later, it will error out (by request)
        print("[train_vicreg] --device gpu requested; not falling back.")
        PIN_CPU = False
    else:
        # auto: try GPU once, otherwise pin CPU
        PIN_CPU = not _gpu_probe()
        if PIN_CPU:
            print("[train_vicreg] Pinning to CPU due to probe failure.")
else:
    print("[train_vicreg] --device cpu -> GPU disabled before TF import.")
    PIN_CPU = True


# -----------------------------------------------------------------------------
# Mixed precision helper (kept OFF by default for stability)
# -----------------------------------------------------------------------------
def set_mixed_precision(enable: bool) -> None:
    """
    Optionally enable mixed_bfloat16; OFF by default to avoid dtype mismatches.
    """
    try:
        from tensorflow.keras import mixed_precision as mp
    except Exception:
        mp = None
    if mp is None:
        print("[train_vicreg] Mixed precision not available in this TF build.")
        return
    mp.set_global_policy("mixed_bfloat16" if enable else "float32")
    print(f"[train_vicreg] Policy set to: {mp.global_policy()}")


# -----------------------------------------------------------------------------
# Data pipeline (CIFAR-10/100 with two-view augmentation)
# -----------------------------------------------------------------------------
def steps_for_dataset(name: str, batch_size: int) -> Tuple[int, int]:
    """Return (num_train_images, steps_per_epoch) for the dataset."""
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        n = 50_000
    elif name in {"cifar100", "cifar-100"}:
        n = 50_000
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    return n, max(1, n // batch_size)


def color_jitter(x: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    """Light color jitter: brightness, contrast, saturation; clip to [0,1]."""
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    return tf.clip_by_value(x, 0.0, 1.0)


def random_augment(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Basic SSL-style spatial + color augmentation for a single view."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, image_size + 8, image_size + 8)
    image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image


def two_view_map(image: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Return two independently augmented views of the same input image."""
    return random_augment(image, image_size), random_augment(image, image_size)


def build_cifar10(image_size: int, batch_size: int) -> tf.data.Dataset:
    """Two-view pipeline over CIFAR-10 training set (50k images)."""
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_cifar100(image_size: int, batch_size: int) -> tf.data.Dataset:
    """Two-view pipeline over CIFAR-100 training set (50k images)."""
    (x_train, _), _ = keras.datasets.cifar100.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_dataset(name: str, image_size: int, batch_size: int) -> tf.data.Dataset:
    """Return an **infinite** dataset of (x1, x2) two-view batches for training."""
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        ds = build_cifar10(image_size, batch_size)
    elif name in {"cifar100", "cifar-100"}:
        ds = build_cifar100(image_size, batch_size)
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    return ds.repeat()  # infinite stream for fit(steps_per_epoch=...)


# -----------------------------------------------------------------------------
# Encoder + projector with stable, eval-friendly names
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# VICReg loss (computed in float32)
# -----------------------------------------------------------------------------
@dataclass
class VICRegWeights:
    """Weights for the three VICReg terms."""
    sim: float = 25.0
    var: float = 25.0
    cov: float = 1.0


def invariance_loss(z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
    """L2 distance between paired projections z1 and z2."""
    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z: tf.Tensor, gamma: float = 1.0) -> tf.Tensor:
    """Penalize per-dimension stddev lower than gamma (prevents collapse)."""
    z = tf.cast(z, tf.float32)
    std = tf.math.reduce_std(z, axis=0)
    return tf.reduce_mean(tf.nn.relu(float(gamma) - std))


def covariance_loss(z: tf.Tensor, nu: float = 0.0) -> tf.Tensor:
    """Reduce off-diagonal covariance (decorrelation term)."""
    z = tf.cast(z, tf.float32)
    z = z - tf.reduce_mean(z, axis=0, keepdims=True)
    n = tf.cast(tf.shape(z)[0], tf.float32)
    cov = (tf.transpose(z) @ z) / (n - 1.0)
    diag = tf.linalg.tensor_diag_part(cov)
    off = cov - tf.linalg.diag(diag)
    return tf.reduce_mean(tf.square(off - float(nu)))


def vicreg_total(z1: tf.Tensor, z2: tf.Tensor, w: VICRegWeights, gamma: float, nu: float):
    """Combine the three VICReg terms into a total loss and component logs."""
    inv = invariance_loss(z1, z2)
    var = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov = covariance_loss(z1, nu) + covariance_loss(z2, nu)
    total = w.sim * inv + w.var * var + w.cov * cov
    return total, {"inv": inv, "var": var, "cov": cov, "total": total}


# -----------------------------------------------------------------------------
# Schedules (PURE PYTHON — SAFE UNDER tf.function)
# -----------------------------------------------------------------------------
def cosine_schedule(start: float, end: float, t: float) -> float:
    """Cosine interpolation between start and end for t in [0,1]."""
    t = float(max(0.0, min(1.0, t)))
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t)))


class AdaptiveTargets:
    """Produce adaptive targets gamma_t and nu_t over training progress."""
    def __init__(self, use_schedules: bool):
        self.use_schedules = use_schedules

    def gamma(self, f: float) -> float:
        return cosine_schedule(0.8, 1.0, f) if self.use_schedules else 1.0

    def nu(self, f: float) -> float:
        return cosine_schedule(0.1, 0.0, f) if self.use_schedules else 0.0


class WeightSchedules:
    """Optionally schedule the invariance weight (sim); others constant."""
    def __init__(self, w0: VICRegWeights, use_schedules: bool):
        self.w0 = w0
        self.use_schedules = use_schedules

    def weights(self, f: float) -> VICRegWeights:
        if not self.use_schedules:
            return self.w0
        sim = cosine_schedule(self.w0.sim * 0.5, self.w0.sim, f)
        return VICRegWeights(sim=sim, var=self.w0.var, cov=self.w0.cov)


# -----------------------------------------------------------------------------
# Trainer (subclassed Keras.Model with custom train_step) — uses KERAS .fit()
# -----------------------------------------------------------------------------
class VICRegTrainer(keras.Model):
    """
    Wrap encoder+projector and implement custom train_step.

    Note
    ----
    We don't implement call(...), because training is entirely in train_step.
    We'll force-build once so ModelCheckpoint(save_weights_only=True) works.
    """
    def __init__(
        self, encoder: keras.Model, projector: keras.Model,
        w0: VICRegWeights, adaptive: bool, sched: bool,
        steps_per_epoch: int, epochs: int
    ):
        super().__init__(name="vicreg_trainer")
        self.encoder = encoder
        self.projector = projector
        self.w0 = w0
        self.adaptive = adaptive
        self.targets = AdaptiveTargets(sched)
        self.schedules = WeightSchedules(w0, sched)
        self.total_steps = max(1, steps_per_epoch * epochs)
        self.curr_step = 0

        # Trackers -> appear in the Keras progress bar
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.inv_tracker = keras.metrics.Mean(name="inv")
        self.var_tracker = keras.metrics.Mean(name="var")
        self.cov_tracker = keras.metrics.Mean(name="cov")

    @property
    def metrics(self):
        return [self.loss_tracker, self.inv_tracker, self.var_tracker, self.cov_tracker]

    def train_step(self, data):
        """data: tuple(x1, x2) — two augmented views from the dataset."""
        x1, x2 = data
        frac = self.curr_step / float(self.total_steps)
        self.curr_step += 1

        gamma = self.targets.gamma(frac) if self.adaptive else 1.0
        nu = self.targets.nu(frac) if self.adaptive else 0.0
        w = self.schedules.weights(frac)

        with tf.GradientTape() as tape:
            f1 = self.encoder(x1, training=True)
            f2 = self.encoder(x2, training=True)
            z1 = self.projector(f1, training=True)
            z2 = self.projector(f2, training=True)
            total, logs = vicreg_total(z1, z2, w, gamma=gamma, nu=nu)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(logs["total"])
        self.inv_tracker.update_state(logs["inv"])
        self.var_tracker.update_state(logs["var"])
        self.cov_tracker.update_state(logs["cov"])
        return {m.name: m.result() for m in self.metrics}


# -----------------------------------------------------------------------------
# Argparse & helpers
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10|cifar100")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--feat-dim", type=int, default=2048)
    p.add_argument("--proj-out", type=int, default=8192)
    p.add_argument("--proj-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--use-schedules", action="store_true")
    p.add_argument("--model-dir", type=str, default="checkpoints_tf",
                   help="If --ckpt-out not set, write to a timestamped folder here.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Optional run name prefix for the timestamped folder.")
    p.add_argument("--ckpt-out", type=str, default=None,
                   help="Optional explicit .h5 weights path (full model).")
    return p.parse_args()


def force_build_for_saving(
    trainer: keras.Model, encoder: keras.Model, projector: keras.Model, image_size: int
) -> None:
    """
    Ensure variables exist and mark the subclassed model as built so that
    `ModelCheckpoint(save_weights_only=True)` can save weights safely.
    """
    _ = encoder(tf.zeros([1, image_size, image_size, 3]), training=False)
    _ = projector(tf.zeros([1, encoder.output_shape[-1]]), training=False)
    trainer.built = True
    print("[train_vicreg] Forced build complete; trainer.built = True.")


def main() -> None:
    args = parse_args()
    set_mixed_precision(enable=False)

    # Dataset + steps
    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    nimg, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(f"[train_vicreg] dataset={args.dataset} img={args.image_size} "
          f"bs={args.batch_size} epochs={args.epochs} steps/epoch={steps_per_epoch}")

    # Output layout
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    if args.ckpt_out:
        os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
        out_dir = os.path.dirname(args.ckpt_out)
        full_ckpt_path = args.ckpt_out
        enc_ckpt_path = (full_ckpt_path.replace(".weights.h5", ".encoder.weights.h5")
                         if full_ckpt_path.endswith(".weights.h5")
                         else os.path.join(out_dir, "vicreg_encoder.weights.h5"))
    else:
        base = args.model_dir or "checkpoints_tf"
        run_prefix = args.run_name if args.run_name else "pretrain"
        out_dir = os.path.join(base, f"{run_prefix}_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        full_ckpt_path = os.path.join(out_dir, "vicreg_full.weights.h5")
        enc_ckpt_path = os.path.join(out_dir, "vicreg_encoder.weights.h5")

    # Save minimal config for traceability
    run_info = {
        "timestamp": ts, "device": _pre_args.device, "pinned_cpu": PIN_CPU,
        "dataset": args.dataset, "n_images": nimg, "image_size": args.image_size,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "feat_dim": args.feat_dim, "proj_out": args.proj_out, "proj_layers": args.proj_layers,
        "lr": args.lr, "wd": args.wd, "adaptive": args.adaptive, "use_schedules": args.use_schedules,
        "full_ckpt_path": full_ckpt_path, "enc_ckpt_path": enc_ckpt_path,
    }
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    device_str = "/CPU:0" if PIN_CPU else "/GPU:0"
    print(f"[train_vicreg] Using device scope: {device_str}")

    with tf.device(device_str):
        # Build modules with stable names that match eval
        encoder = build_encoder(args.image_size, feat_dim=args.feat_dim)
        projector = build_projector(args.feat_dim, args.proj_out, args.proj_layers)

        # Optimizer: AdamW if available, else Adam
        try:
            import tensorflow_addons as tfa  # type: ignore
            opt = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.wd)
        except Exception:
            opt = keras.optimizers.Adam(learning_rate=args.lr)

        trainer = VICRegTrainer(
            encoder=encoder,
            projector=projector,
            w0=VICRegWeights(sim=25.0, var=25.0, cov=1.0),
            adaptive=args.adaptive,
            sched=args.use_schedules,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
        )
        trainer.compile(optimizer=opt)

        # *** CRITICAL: force-build so save_weights works on a subclassed model ***
        force_build_for_saving(trainer, encoder, projector, args.image_size)

        # Keras progress bar + best checkpoint on loss (full model weights)
        ckpt_cb = keras.callbacks.ModelCheckpoint(
            filepath=full_ckpt_path,
            save_weights_only=True,
            monitor="loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
        term_nan = keras.callbacks.TerminateOnNaN()

        trainer.fit(
            ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[ckpt_cb, term_nan],
            verbose=1,  # <- the classic Keras bar you wanted
        )

        # Always (re)save encoder-only weights at the end for eval convenience
        encoder.save_weights(enc_ckpt_path)

    print(f"[train_vicreg] Done.\n  Encoder -> {enc_ckpt_path}\n  Full -> {full_ckpt_path}\n  Logs -> {out_dir}")


if __name__ == "__main__":
    main()
