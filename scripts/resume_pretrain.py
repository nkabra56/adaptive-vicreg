"""
Resume self-supervised pretraining for VICReg / Adaptive VICReg (TensorFlow + Keras).

This script continues training from a previously saved **weights-only** checkpoint
(e.g., "checkpoints_tf/vicreg_tf.weights.h5") produced by train_vicreg.py.
It rebuilds the SAME encoder+projector+trainer graph, builds the model,
loads the weights, aligns the internal step counter with --initial-epoch,
and resumes fit() safely (LR warmup, gradient clipping, BN freeze, loss guard).

Device selection happens BEFORE importing TensorFlow:
  --device cpu   -> hides GPUs via CUDA_VISIBLE_DEVICES=""
  --device gpu   -> attempts GPU (no silent fallback)
  --device auto  -> tiny GPU probe; on failure, pins to CPU

Author: Nishant Kabra
Date: 11/14/25
"""
import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple

# -------------------- 1) Device flag before importing TensorFlow --------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                  help="Device placement: auto|gpu|cpu (default: auto)")
_pre_args, _ = _pre.parse_known_args()

if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# -------------------- 2) Import TF/Keras --------------------
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE
_PIN_CPU = (_pre_args.device == "cpu")


def _print_devices():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[resume_pretrain] Visible GPUs: {gpus}")


def _enable_memory_growth():
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("[resume_pretrain] set_memory_growth warning:", repr(e))


def _gpu_probe() -> bool:
    """Run a tiny Conv2D on GPU to flush obvious PTX/kernel issues."""
    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform([1, 16, 16, 3])
            y = layers.Conv2D(4, 3, padding="same")(x)
            _ = tf.reduce_sum(y).numpy()
        print("[resume_pretrain] GPU probe OK.")
        return True
    except Exception as e:
        print("[resume_pretrain] GPU probe FAILED:", repr(e))
        return False


if _pre_args.device != "cpu":
    _print_devices()
    _enable_memory_growth()
    if _pre_args.device == "gpu":
        print("[resume_pretrain] --device gpu requested; not falling back.")
        _PIN_CPU = False
    else:
        _PIN_CPU = not _gpu_probe()
        if _PIN_CPU:
            print("[resume_pretrain] Pinning to CPU due to probe failure.")
else:
    print("[resume_pretrain] --device cpu -> GPU disabled before TF import.")
    _PIN_CPU = True


# -------------------- Mixed precision helper (OFF by default) --------------------
def set_mixed_precision(enable: bool):
    """Keep float32 for stability when resuming."""
    try:
        from tensorflow.keras import mixed_precision as mp
    except Exception:
        mp = None
    if mp is None:
        print("[resume_pretrain] Mixed precision not available in this TF build.")
        return
    mp.set_global_policy("mixed_bfloat16" if enable else "float32")
    print(f"[resume_pretrain] Policy set to: {mp.global_policy()}")


# -------------------- Data pipeline --------------------
def steps_for_dataset(name: str, batch_size: int) -> Tuple[int, int]:
    """Return (num_samples, steps_per_epoch) for CIFAR10/100."""
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        n = 50_000
    elif name in {"cifar100", "cifar-100"}:
        n = 50_000
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    return n, max(1, n // batch_size)


def _color_jitter(x, s=0.5):
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, 1 - 0.8 * s, 1 + 0.8 * s)
    x = tf.image.random_saturation(x, 1 - 0.8 * s, 1 + 0.8 * s)
    return tf.clip_by_value(x, 0.0, 1.0)


def _random_augment(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Simple VICReg-style crop+flip+jitter."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, image_size + 8, image_size + 8)
    image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = _color_jitter(image, s=0.5)
    return image


def _two_view_map(image: tf.Tensor, image_size: int):
    """Return two independently augmented views of the same image."""
    return _random_augment(image, image_size), _random_augment(image, image_size)


def build_cifar10(image_size: int, batch_size: int) -> tf.data.Dataset:
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: _two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_cifar100(image_size: int, batch_size: int) -> tf.data.Dataset:
    (x_train, _), _ = keras.datasets.cifar100.load_data()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda x: _two_view_map(x, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_dataset(name: str, image_size: int, batch_size: int) -> tf.data.Dataset:
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        ds = build_cifar10(image_size, batch_size)
    elif name in {"cifar100", "cifar-100"}:
        ds = build_cifar100(image_size, batch_size)
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
    return ds.repeat()


# -------------------- Encoder + projector --------------------
def _conv_block(x, filters, k=3, s=1):
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_encoder(image_size: int) -> keras.Model:
    """A small ConvNet -> GAP -> 2048-d feature (acts as backbone/encoder)."""
    inp = keras.Input(shape=(image_size, image_size, 3))
    x = _conv_block(inp, 64)
    x = _conv_block(x, 64, s=2)
    x = _conv_block(x, 128)
    x = _conv_block(x, 128, s=2)
    x = _conv_block(x, 256)
    x = _conv_block(x, 256, s=2)
    x = _conv_block(x, 512)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2048)(x)
    return keras.Model(inp, x, name="encoder")


def build_projector(in_dim: int, out_dim: int, num_layers: int) -> keras.Model:
    """
    MLP projector (BN + ReLU on hidden layers, linear on the last layer).
    Shapes/ordering mirror common VICReg projector designs.
    """
    inp = keras.Input(shape=(in_dim,))
    x = inp
    hidden = max(2048, out_dim)
    for _ in range(num_layers - 1):
        x = layers.Dense(hidden, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.Dense(out_dim)(x)
    return keras.Model(inp, x, name="projector")


# -------------------- VICReg losses --------------------
@dataclass
class VICRegWeights:
    sim: float = 25.0
    var: float = 25.0
    cov: float = 1.0


def invariance_loss(z1, z2):
    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z, gamma: float = 1.0):
    z = tf.cast(z, tf.float32)
    std = tf.math.reduce_std(z, axis=0)
    return tf.reduce_mean(tf.nn.relu(gamma - std))


def covariance_loss(z, nu: float = 0.0):
    z = tf.cast(z, tf.float32)
    z = z - tf.reduce_mean(z, axis=0, keepdims=True)
    n = tf.cast(tf.shape(z)[0], tf.float32)
    cov = (tf.transpose(z) @ z) / (n - 1.0)
    d = tf.linalg.tensor_diag_part(cov)
    off = cov - tf.linalg.diag(d)
    return tf.reduce_mean(tf.square(off - nu))


def vicreg_total(z1, z2, w: VICRegWeights, gamma: float, nu: float):
    inv = invariance_loss(z1, z2)
    var = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov = covariance_loss(z1, nu) + covariance_loss(z2, nu)
    total = w.sim * inv + w.var * var + w.cov * cov
    return total, {"loss/inv": inv, "loss/var": var, "loss/cov": cov, "loss/total": total}


# -------------------- Schedules (pure Python) --------------------
def cosine_schedule(start: float, end: float, t: float) -> float:
    """Cosine interpolation between start->end at progress t in [0,1]."""
    t = float(max(0.0, min(1.0, t)))
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t)))


class AdaptiveTargets:
    def __init__(self, use_schedules: bool):
        self.use_schedules = use_schedules

    def gamma(self, f: float) -> float:
        return cosine_schedule(0.8, 1.0, f) if self.use_schedules else 1.0

    def nu(self, f: float) -> float:
        return cosine_schedule(0.1, 0.0, f) if self.use_schedules else 0.0


class WeightSchedules:
    def __init__(self, w0: VICRegWeights, use_schedules: bool):
        self.w0 = w0
        self.use_schedules = use_schedules

    def weights(self, f: float) -> VICRegWeights:
        if not self.use_schedules:
            return self.w0
        sim = cosine_schedule(self.w0.sim * 0.5, self.w0.sim, f)
        return VICRegWeights(sim=sim, var=self.w0.var, cov=self.w0.cov)


# -------------------- Trainer --------------------
class VICRegTrainer(keras.Model):
    """
    Keras Model wrapper that:
      - tracks global step (curr_step) to drive schedules,
      - lets us freeze BatchNorm updates for first N steps after resume,
      - exposes loss components as metrics.
    """
    def __init__(self, encoder, projector, w0: VICRegWeights,
                 adaptive: bool, sched: bool, steps_per_epoch: int, epochs: int,
                 bn_freeze_steps: int = 0):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.w0 = w0
        self.adaptive = adaptive
        self.targets = AdaptiveTargets(sched)
        self.schedules = WeightSchedules(w0, sched)
        self.total_steps = steps_per_epoch * epochs
        self.curr_step = 0
        self.bn_freeze_steps = int(bn_freeze_steps)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.inv_tracker = keras.metrics.Mean(name="inv")
        self.var_tracker = keras.metrics.Mean(name="var")
        self.cov_tracker = keras.metrics.Mean(name="cov")

    @property
    def metrics(self):
        return [self.loss_tracker, self.inv_tracker, self.var_tracker, self.cov_tracker]

    def call(self, inputs, training=None):
        x1, x2 = inputs
        # Freeze BN updates during warmup: treat them as inference initially.
        bn_train = (self.curr_step >= self.bn_freeze_steps)
        f1 = self.encoder(x1, training=bn_train if training is None else training)
        f2 = self.encoder(x2, training=bn_train if training is None else training)
        z1 = self.projector(f1, training=training)
        z2 = self.projector(f2, training=training)
        return z1, z2

    def train_step(self, data):
        x1, x2 = data
        frac = self.curr_step / max(1, self.total_steps)
        gamma = self.targets.gamma(frac) if self.adaptive else 1.0
        nu = self.targets.nu(frac) if self.adaptive else 0.0
        w = self.schedules.weights(frac)

        with tf.GradientTape() as tape:
            z1, z2 = self((x1, x2), training=True)
            total, logs = vicreg_total(z1, z2, w, gamma=gamma, nu=nu)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.curr_step += 1  # advance AFTER the update
        self.loss_tracker.update_state(logs["loss/total"])
        self.inv_tracker.update_state(logs["loss/inv"])
        self.var_tracker.update_state(logs["loss/var"])
        self.cov_tracker.update_state(logs["loss/cov"])
        return {m.name: m.result() for m in self.metrics}


# -------------------- Safety Callbacks --------------------
class WarmupLR(keras.callbacks.Callback):
    """Linear LR warmup for the first N steps (batches) after resume."""
    def __init__(self, base_lr: float, warmup_steps: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(0, warmup_steps))

    def on_train_batch_begin(self, batch, logs=None):
        if self.warmup_steps <= 0:
            return
        step = int(self.model.curr_step)
        if step < self.warmup_steps:
            # ramp from base_lr * 0.1 -> base_lr (starts gentle)
            scale = 0.1 + 0.9 * (step + 1) / float(self.warmup_steps)
            lr = self.base_lr * scale
            keras.backend.set_value(self.model.optimizer.learning_rate, lr)


class LossExplosionGuard(keras.callbacks.Callback):
    """If loss goes crazy high, shrink LR by 10x to recover."""
    def __init__(self, threshold: float = 1e8, factor: float = 0.1):
        super().__init__()
        self.threshold = float(threshold)
        self.factor = float(factor)

    def on_train_batch_end(self, batch, logs=None):
        if not logs:
            return
        loss = float(logs.get("loss", 0.0))
        if loss > self.threshold:
            lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = max(lr * self.factor, 1e-6)
            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(f"[loss-guard] loss={loss:.3e} > {self.threshold:.1e} → "
                  f"lr {lr:.2e} → {new_lr:.2e}")


# -------------------- CLI + main --------------------
def parse_args():
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", type=str, default="checkpoints_tf/vicreg_tf.weights.h5",
                   help="Path to weights-only checkpoint (.h5) to load.")
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10|cifar100")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--proj-out", type=int, default=8192)
    p.add_argument("--proj-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.2, help="Base LR used in training.")
    p.add_argument("--resume-lr", type=float, default=None,
                   help="LR to use after resume (before warmup). Defaults to --lr.")
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--use-schedules", action="store_true")
    p.add_argument("--initial-epoch", type=int, default=0,
                   help="Epoch to start from (completed epochs).")
    p.add_argument("--epochs", type=int, default=100,
                   help="Final epoch to train to (total, not extra).")
    p.add_argument("--model-dir", type=str, default="checkpoints_tf")

    # New safety knobs:
    p.add_argument("--clipnorm", type=float, default=1.0,
                   help="Gradient clipnorm (0 disables clipping).")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear LR warmup steps after resume.")
    p.add_argument("--bn-freeze-steps", type=int, default=0,
                   help="Keep BatchNorm in inference mode for first N steps.")
    p.add_argument("--loss-guard", type=float, default=1e12,
                   help="If batch loss exceeds this, shrink LR 10x.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    set_mixed_precision(enable=False)

    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    _, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(f"[resume_pretrain] dataset={args.dataset} img={args.image_size} "
          f"bs={args.batch_size} initial_epoch={args.initial_epoch} "
          f"final_epochs={args.epochs} steps/epoch={steps_per_epoch}")

    device_str = "/CPU:0" if _PIN_CPU else "/GPU:0"
    print(f"[resume_pretrain] Using device scope: {device_str}")

    with tf.device(device_str):
        # Build models
        encoder = build_encoder(args.image_size)
        projector = build_projector(2048, args.proj_out, args.proj_layers)

        trainer = VICRegTrainer(
            encoder=encoder,
            projector=projector,
            w0=VICRegWeights(sim=25.0, var=25.0, cov=1.0),
            adaptive=args.adaptive,
            sched=args.use_schedules,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            bn_freeze_steps=args.bn_freeze_steps,
        )

        # Build variables (create weights) and run a dummy forward
        trainer.build([
            (None, args.image_size, args.image_size, 3),
            (None, args.image_size, args.image_size, 3),
        ])
        _dummy = tf.zeros([1, args.image_size, args.image_size, 3], dtype=tf.float32)
        _ = trainer((_dummy, _dummy), training=False)

        # Load weights (weights-only .h5)
        print(f"[resume_pretrain] Loading weights from: {args.ckpt}")
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        trainer.load_weights(args.ckpt)
        print("[resume_pretrain] Weights loaded successfully.")

        # Optimizer with gradient clipping
        base_lr = args.lr if args.resume_lr is None else args.resume_lr
        try:
            import tensorflow_addons as tfa
            opt = tfa.optimizers.AdamW(
                learning_rate=base_lr, weight_decay=args.wd,
                clipnorm=(args.clipnorm if args.clipnorm > 0 else None)
            )
        except Exception:
            opt = keras.optimizers.Adam(
                learning_rate=base_lr,
                clipnorm=(args.clipnorm if args.clipnorm > 0 else None)
            )
        trainer.compile(optimizer=opt)

        # Align internal step so schedules pick up where you left off
        trainer.curr_step = int(args.initial_epoch) * int(steps_per_epoch)

        ckpt_path = os.path.join(args.model_dir, "vicreg_tf.weights.h5")
        callbacks = [
            WarmupLR(base_lr=base_lr, warmup_steps=args.warmup_steps),
            LossExplosionGuard(threshold=float(args.loss_guard), factor=0.1),
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                save_weights_only=True,
                monitor="loss",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(
                os.path.join(args.model_dir, "resume_log.csv"), append=True
            ),
        ]

        trainer.fit(
            ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=args.initial_epoch,
            callbacks=callbacks,
            verbose=1,
        )

    print("[resume_pretrain] Done. Best weights at:", ckpt_path)


if __name__ == "__main__":
    main()
