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
  cosine schedules are pure Python (no .numpy()) so they work in graph mode.

- IMPORTANT: We *force-build* the sublayers and mark the subclassed Keras Model
  as built (trainer.built = True) before training so ModelCheckpoint with
  save_weights_only=True will work.

Author: Nishant Kabra
Date: 11/14/2025
"""
from __future__ import annotations

import os
import math
import argparse
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
    """
    Return (num_train_images, steps_per_epoch) for the dataset.
    """
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        n = 50_000
    elif name in {"cifar100", "cifar-100"}:
        n = 50_000
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    return n, max(1, n // batch_size)


def color_jitter(x: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    """
    Light color jitter: brightness, contrast, saturation; clip to [0,1].
    """
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    return tf.clip_by_value(x, 0.0, 1.0)


def random_augment(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """
    Basic SSL-style spatial + color augmentation for a single view.
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, image_size + 8, image_size + 8)
    image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image


def two_view_map(image: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Return two independently augmented views of the same input image.
    """
    return random_augment(image, image_size), random_augment(image, image_size)


def build_cifar10(image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Two-view pipeline over CIFAR-10 training set (50k images).
    """
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_cifar100(image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Two-view pipeline over CIFAR-100 training set (50k images).
    """
    (x_train, _), _ = keras.datasets.cifar100.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_dataset(name: str, image_size: int, batch_size: int) -> tf.data.Dataset:
    """
    Return an **infinite** dataset of (x1, x2) two-view batches for training.
    """
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        ds = build_cifar10(image_size, batch_size)
    elif name in {"cifar100", "cifar-100"}:
        ds = build_cifar100(image_size, batch_size)
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    return ds.repeat()  # infinite stream for fit(steps_per_epoch=...)


# -----------------------------------------------------------------------------
# Encoder + projector (simple CNN backbone)
# -----------------------------------------------------------------------------
def conv_block(x: tf.Tensor, filters: int, k: int = 3, s: int = 1) -> tf.Tensor:
    """
    Conv2D -> BatchNorm -> ReLU block.
    """
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_encoder(image_size: int) -> keras.Model:
    """
    Tiny CNN encoder that outputs a 2048-dim feature vector.
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
    x = layers.Dense(2048, name="feat")(x)  # named for compatibility
    return keras.Model(inp, x, name="encoder")


def build_projector(in_dim: int, out_dim: int, num_layers: int) -> keras.Model:
    """
    VICReg projector MLP with (num_layers - 1) * [Dense + BN + ReLU] + Dense.
    """
    assert num_layers >= 1, "proj-layers must be >= 1"
    inp = keras.Input(shape=(in_dim,))
    x = inp
    hidden = max(2048, out_dim)
    for i in range(num_layers - 1):
        x = layers.Dense(hidden, use_bias=False, name=f"proj_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"proj_bn_{i}")(x)
        x = layers.ReLU(name=f"proj_relu_{i}")(x)
    x = layers.Dense(out_dim, name="proj_out")(x)
    return keras.Model(inp, x, name="projector")


# -----------------------------------------------------------------------------
# VICReg loss (computed in float32)
# -----------------------------------------------------------------------------
@dataclass
class VICRegWeights:
    """
    Weights for the three VICReg terms.
    """
    sim: float = 25.0
    var: float = 25.0
    cov: float = 1.0


def invariance_loss(z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
    """L2 distance between paired projections z1 and z2."""
    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z: tf.Tensor, gamma: float = 1.0) -> tf.Tensor:
    """
    Penalize per-dimension stddev lower than gamma (prevents collapse).
    """
    z = tf.cast(z, tf.float32)
    std = tf.math.reduce_std(z, axis=0)
    return tf.reduce_mean(tf.nn.relu(float(gamma) - std))


def covariance_loss(z: tf.Tensor, nu: float = 0.0) -> tf.Tensor:
    """
    Reduce off-diagonal covariance (decorrelation term).
    """
    z = tf.cast(z, tf.float32)
    z = z - tf.reduce_mean(z, axis=0, keepdims=True)
    n = tf.cast(tf.shape(z)[0], tf.float32)
    cov = (tf.transpose(z) @ z) / (n - 1.0)
    d = tf.linalg.tensor_diag_part(cov)
    off = cov - tf.linalg.diag(d)
    return tf.reduce_mean(tf.square(off - float(nu)))


def vicreg_total(z1: tf.Tensor, z2: tf.Tensor, w: VICRegWeights, gamma: float, nu: float):
    """
    Combine the three VICReg terms into a total loss and component logs.
    """
    inv = invariance_loss(z1, z2)
    var = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov = covariance_loss(z1, nu) + covariance_loss(z2, nu)
    total = w.sim * inv + w.var * var + w.cov * cov
    return total, {"inv": inv, "var": var, "cov": cov, "total": total}


# -----------------------------------------------------------------------------
# Schedules (PURE PYTHON — SAFE UNDER tf.function)
# -----------------------------------------------------------------------------
def cosine_schedule(start: float, end: float, t: float) -> float:
    """
    Cosine interpolation between start and end for t in [0,1].
    Returns a Python float (no Tensor ops, no .numpy()).
    """
    t = float(max(0.0, min(1.0, t)))
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t)))


class AdaptiveTargets:
    """
    Produce adaptive targets gamma_t and nu_t over training progress.
    """
    def __init__(self, use_schedules: bool):
        self.use_schedules = use_schedules

    def gamma(self, f: float) -> float:
        return cosine_schedule(0.8, 1.0, f) if self.use_schedules else 1.0

    def nu(self, f: float) -> float:
        return cosine_schedule(0.1, 0.0, f) if self.use_schedules else 0.0


class WeightSchedules:
    """
    Optionally schedule the invariance weight (sim); others constant.
    """
    def __init__(self, w0: VICRegWeights, use_schedules: bool):
        self.w0 = w0
        self.use_schedules = use_schedules

    def weights(self, f: float) -> VICRegWeights:
        if not self.use_schedules:
            return self.w0
        sim = cosine_schedule(self.w0.sim * 0.5, self.w0.sim, f)
        return VICRegWeights(sim=sim, var=self.w0.var, cov=self.w0.cov)


# -----------------------------------------------------------------------------
# Trainer (subclassed Keras.Model with custom train_step)
# -----------------------------------------------------------------------------
class VICRegTrainer(keras.Model):
    """
    Wrap encoder+projector and implement custom train_step.

    Note
    ----
    We don't implement call(...), because training is entirely in train_step.
    That means Keras does not "build" the model automatically. We'll explicitly
    force-build weights and set `trainer.built = True` before fit() so
    ModelCheckpoint(save_weights_only=True) works.
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
        self.total_steps = steps_per_epoch * epochs
        self.curr_step = 0

        # Trackers for nice logs
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.inv_tracker = keras.metrics.Mean(name="inv")
        self.var_tracker = keras.metrics.Mean(name="var")
        self.cov_tracker = keras.metrics.Mean(name="cov")

    @property
    def metrics(self):
        return [self.loss_tracker, self.inv_tracker, self.var_tracker, self.cov_tracker]

    def train_step(self, data):
        """
        data: tuple(x1, x2) — two augmented views from the dataset.
        """
        x1, x2 = data
        frac = self.curr_step / max(1, self.total_steps)  # Python float progress
        self.curr_step += 1

        gamma = self.targets.gamma(frac) if self.adaptive else 1.0   # floats
        nu = self.targets.nu(frac) if self.adaptive else 0.0
        w = self.schedules.weights(frac)                              # VICRegWeights

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
        # Keras will log keys matching tracker names
        return {m.name: m.result() for m in self.metrics}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10|cifar100")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--proj-out", type=int, default=8192)
    p.add_argument("--proj-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--use-schedules", action="store_true")
    # p.add_argument("--mixed-bf16", action="store_true")  # keep OFF by default
    p.add_argument("--model-dir", type=str, default="checkpoints_tf")
    return p.parse_args()


def force_build_for_saving(
    trainer: keras.Model, encoder: keras.Model, projector: keras.Model, image_size: int
) -> None:
    """
    Ensure variables exist and mark the subclassed model as built so that
    `ModelCheckpoint(save_weights_only=True)` can save weights safely.

    We run encoder+projector once on dummy inputs (to create variables), then
    set `trainer.built = True`. This is sufficient because save_weights() saves
    all sublayer variables under their names ('encoder/...', 'projector/...').
    """
    # Create variables on sublayers
    _ = encoder(tf.zeros([1, image_size, image_size, 3]), training=False)
    _ = projector(tf.zeros([1, 2048]), training=False)
    # Mark the trainer as "built" for Keras saving logic
    trainer.built = True
    print("[train_vicreg] Forced build complete; trainer.built = True.")


def main() -> None:
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # Keep float32 by default for max compatibility
    set_mixed_precision(enable=False)

    # Build dataset and steps/epoch
    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    _, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(f"[train_vicreg] dataset={args.dataset} img={args.image_size} "
          f"bs={args.batch_size} epochs={args.epochs} steps/epoch={steps_per_epoch}")

    # Build and train on the chosen device
    device_str = "/CPU:0" if PIN_CPU else "/GPU:0"
    print(f"[train_vicreg] Using device scope: {device_str}")

    with tf.device(device_str):
        # Build modules
        encoder = build_encoder(args.image_size)
        projector = build_projector(2048, args.proj_out, args.proj_layers)

        # Optimizer (AdamW if available)
        try:
            import tensorflow_addons as tfa
            opt = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.wd)
        except Exception:
            opt = keras.optimizers.Adam(learning_rate=args.lr)

        # Trainer model (subclassed)
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

        ckpt_path = os.path.join(args.model_dir, "vicreg_tf.weights.h5")
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                save_weights_only=True,
                monitor="loss",          # monitors trainer.loss_tracker (name="loss")
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.TerminateOnNaN(),
        ]

        trainer.fit(
            ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
        )

    print("[train_vicreg] Done. Best weights at:", ckpt_path)

if __name__ == "__main__":
    main()
