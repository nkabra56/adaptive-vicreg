"""
Encoder/projector builders and the VICReg training loop.

`VICRegTrainer` runs the two-view forward pass, computes the VICReg loss
components, and optimizes the encoder and projector. Two independent,
optional mechanisms can be layered on top of plain VICReg, each behind its
own flag:

- `adaptive_weights`: per-step sim/var/cov loss weights from
  `schedules.AdaptiveReweighter` (EMA loss-magnitude balancing plus an
  embedding-health reactive boost). This is what "Adaptive VICReg" means in
  the README. When off, weights are constant (`w0`).
- `adaptive_targets`: the variance floor `gamma` and redundancy target `nu`
  become time-varying via `schedules.AdaptiveTargets`, instead of the
  baseline constants `gamma=1.0, nu=0.0`. This changes what the loss targets,
  not how the three terms are weighted against each other.

Baseline VICReg is the same code path with both flags off.
"""
from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from .losses import vicreg_total, VICRegWeights
from .schedules import WeightSchedules, AdaptiveTargets, AdaptiveReweighter


def _conv_block(x, filters, k=3, s=1, name=None):
    """Conv -> BN -> ReLU block."""
    x = keras.layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False, name=None if name is None else f"{name}_conv")(x)
    x = keras.layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = keras.layers.ReLU(name=None if name is None else f"{name}_relu")(x)
    return x


def build_encoder(image_size: int, feat_dim: int = 2048) -> keras.Model:
    """
    Build the CIFAR-friendly CNN encoder: three conv stages, global average
    pooling, then a Dense projection to `feat_dim`. Outputs a feature vector,
    not logits.
    """
    inp = keras.Input(shape=(image_size, image_size, 3), name="enc_in")

    x = _conv_block(inp, 64, 3, 1, name="s1a")
    x = _conv_block(x,   64, 3, 1, name="s1b")

    x = _conv_block(x,  128, 3, 2, name="s2a")
    x = _conv_block(x,  128, 3, 1, name="s2b")

    x = _conv_block(x,  256, 3, 2, name="s3a")
    x = _conv_block(x,  256, 3, 1, name="s3b")

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    feat = keras.layers.Dense(feat_dim, use_bias=False, name="enc_out")(x)

    return keras.Model(inp, feat, name="encoder")


def _mlp(feat_in: int, proj_out: int, proj_layers: int, name_prefix: str) -> keras.Sequential:
    """Build a 1-3 layer MLP mapping `feat_in` to `proj_out`. No BatchNorm, to stay simple across batch sizes."""
    layers = []
    if proj_layers == 1:
        layers += [keras.layers.Dense(proj_out, activation=None, name=f"{name_prefix}_dense_out")]
    elif proj_layers == 2:
        layers += [
            keras.layers.Dense(feat_in, activation="relu", name=f"{name_prefix}_dense_h1"),
            keras.layers.Dense(proj_out, activation=None, name=f"{name_prefix}_dense_out"),
        ]
    else:  # 3 or more, clamped to 3
        layers += [
            keras.layers.Dense(feat_in, activation="relu", name=f"{name_prefix}_dense_h1"),
            keras.layers.Dense(feat_in, activation="relu", name=f"{name_prefix}_dense_h2"),
            keras.layers.Dense(proj_out, activation=None, name=f"{name_prefix}_dense_out"),
        ]
    return keras.Sequential(layers, name=f"{name_prefix}_mlp")


def build_projector(feat_in: int, proj_out: int, proj_layers: int) -> keras.Model:
    """Wrap `_mlp` as a Keras Model, with a unique name suffix to avoid layer-name clashes on rebuild."""
    uniq = hex(id(object()))[-6:]
    inp = keras.Input(shape=(feat_in,), name=f"proj_in_{uniq}")
    mlp = _mlp(feat_in, proj_out, proj_layers, name_prefix=f"proj_{feat_in}_{proj_out}_{proj_layers}_{uniq}")
    out = mlp(inp)
    return keras.Model(inp, out, name=f"proj_{uniq}")


class VICRegTrainer(keras.Model):
    """
    Custom Keras Model implementing (Adaptive) VICReg training.

    Args:
        encoder: Backbone model.
        projector: MLP head mapping encoder features to projection space.
        w0: Base weights for the (sim, var, cov) VICReg components.
        adaptive_weights: If True, per-step sim/var/cov weights come from
            `AdaptiveReweighter`. If False, weights are constant (`w0`).
        adaptive_targets: If True, gamma/nu are time-varying via
            `AdaptiveTargets`. If False, use the baseline constants
            gamma=1.0, nu=0.0.
        use_schedules: Whether to scale optimizer lr/wd via cosine schedules.
        steps_per_epoch: Steps per epoch, used for progress calculation.
        epochs: Total training epochs.
        base_lr: Base learning rate, for schedules and logging.
        base_wd: Base weight decay, for schedules and logging.
        reweighter_kwargs: Optional overrides passed to `AdaptiveReweighter`.
    """
    def __init__(
        self,
        *,
        encoder: keras.Model,
        projector: keras.Model,
        w0: VICRegWeights,
        adaptive_weights: bool = False,
        adaptive_targets: bool = False,
        use_schedules: bool,
        steps_per_epoch: int,
        epochs: int,
        base_lr: float,
        base_wd: float,
        reweighter_kwargs: Optional[dict] = None,
    ):
        super().__init__(name="vicreg_trainer")
        self.encoder = encoder
        self.projector = projector
        self.w0 = w0

        self.adaptive_weights = bool(adaptive_weights)
        self.adaptive_targets = bool(adaptive_targets)
        self.use_schedules = bool(use_schedules)

        self.steps_per_epoch = int(steps_per_epoch)
        self.epochs = int(epochs)
        self.total_steps = int(self.steps_per_epoch * self.epochs)
        self.curr_step = tf.Variable(0, dtype=tf.int64, trainable=False)

        # Constant VICReg weights; used only when adaptive_weights is False.
        self.schedules = WeightSchedules(
            w0=w0,
            use=self.use_schedules,
            base_lr=float(base_lr),
            base_wd=float(base_wd),
            total_steps=self.total_steps,
            warmup_frac=0.0,
            min_scale=0.0,
        )

        # Only constructed (and only holds EMA state) when enabled.
        self.reweighter = (
            AdaptiveReweighter(w0=w0, **(reweighter_kwargs or {}))
            if self.adaptive_weights
            else None
        )

        self.targets = AdaptiveTargets(use=self.adaptive_targets)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.align_tracker = keras.metrics.Mean(name="l_align")
        self.var_tracker = keras.metrics.Mean(name="l_var")
        self.cov_tracker = keras.metrics.Mean(name="l_cov")
        # Realized weights, so adaptive_weights runs are diagnosable from the
        # metrics history (did the reweighter actually move, and where to).
        self.w_sim_tracker = keras.metrics.Mean(name="w_sim")
        self.w_var_tracker = keras.metrics.Mean(name="w_var")
        self.w_cov_tracker = keras.metrics.Mean(name="w_cov")

    @property
    def metrics(self):
        """Metrics Keras should reset at the start of each epoch."""
        return [
            self.loss_tracker, self.align_tracker, self.var_tracker, self.cov_tracker,
            self.w_sim_tracker, self.w_var_tracker, self.w_cov_tracker,
        ]

    def compile(self, optimizer: keras.optimizers.Optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data) -> dict:
        """One training step on a batch of paired views: data = (x1, x2), each [B, H, W, 3]."""
        if isinstance(data, (tuple, list)) and len(data) == 2:
            x1, x2 = data
        else:
            x1, x2 = data[0], data[1]

        frac = tf.cast(self.curr_step, tf.float32) / tf.cast(self.total_steps, tf.float32)

        with tf.GradientTape() as tape:
            h1 = self.encoder(x1, training=True)
            h2 = self.encoder(x2, training=True)
            z1 = self.projector(h1, training=True)
            z2 = self.projector(h2, training=True)

            gamma = self.targets.gamma(frac) if self.adaptive_targets else tf.constant(1.0, tf.float32)
            nu    = self.targets.nu(frac)    if self.adaptive_targets else tf.constant(0.0, tf.float32)

            # Raw (unweighted) components. Weighting happens below, so
            # adaptive_weights can see the raw magnitudes and the probe
            # embedding before weights are chosen.
            _, parts = vicreg_total(
                z1, z2,
                w=VICRegWeights(sim=1.0, var=1.0, cov=1.0),
                gamma=gamma,
                nu=nu,
            )

            # The only place adaptive_weights and baseline differ.
            weights = (
                self.reweighter(parts, z_probe=z1)
                if self.adaptive_weights
                else self.schedules.weights(frac)
            )

            total = (
                weights["sim"] * parts["l_align"]
                + weights["var"] * parts["l_var"]
                + weights["cov"] * parts["l_cov"]
            )

        vars = self.encoder.trainable_variables + self.projector.trainable_variables
        grads = tape.gradient(total, vars)
        self.optimizer.apply_gradients(zip(grads, vars))

        self.loss_tracker.update_state(total)
        self.align_tracker.update_state(parts["l_align"])
        self.var_tracker.update_state(parts["l_var"])
        self.cov_tracker.update_state(parts["l_cov"])
        self.w_sim_tracker.update_state(weights["sim"])
        self.w_var_tracker.update_state(weights["var"])
        self.w_cov_tracker.update_state(weights["cov"])

        self.curr_step.assign_add(1)

        return {
            "loss": self.loss_tracker.result(),
            "l_align": self.align_tracker.result(),
            "l_var": self.var_tracker.result(),
            "l_cov": self.cov_tracker.result(),
            "w_sim": self.w_sim_tracker.result(),
            "w_var": self.w_var_tracker.result(),
            "w_cov": self.w_cov_tracker.result(),
        }

    def get_config(self):
        """
        Minimal JSON-serializable snapshot of construction-time settings.

        Only weights are actually saved/restored (via `save_weights` /
        `ModelCheckpoint(save_weights_only=True)`), so Keras never uses this
        to rebuild the model; it only needs something serializable here to
        stop warning about missing `get_config`. Excludes the encoder,
        projector, and optimizer, since those aren't JSON-serializable and
        are restored from weights instead.
        """
        cfg = {
            "class_name": self.__class__.__name__,
            "adaptive_weights": bool(getattr(self, "adaptive_weights", False)),
            "adaptive_targets": bool(getattr(self, "adaptive_targets", False)),
            "use_schedules": bool(getattr(self, "use_schedules", False)),
        }

        try:
            w0 = getattr(self, "w0")
            cfg["w0"] = {"sim": float(w0.sim), "var": float(w0.var), "cov": float(w0.cov)}
        except Exception:
            cfg["w0"] = {"sim": 25.0, "var": 25.0, "cov": 1.0}

        total_steps = getattr(self, "total_steps", None)
        if total_steps is not None:
            try:
                cfg["total_steps"] = int(total_steps)
            except Exception:
                pass

        base_lr = None
        base_wd = None
        try:
            base_lr = float(getattr(self, "schedules").base_lr)
            base_wd = float(getattr(self, "schedules").base_wd)
            cfg["warmup_frac"] = float(getattr(self, "schedules").warmup_frac)
            cfg["min_scale"]   = float(getattr(self, "schedules").min_scale)
        except Exception:
            pass

        if base_lr is None:
            base_lr = float(getattr(self, "base_lr", 0.0))
        if base_wd is None:
            base_wd = float(getattr(self, "base_wd", 0.0))

        cfg["base_lr"] = base_lr
        cfg["base_wd"] = base_wd

        return cfg

    @classmethod
    def from_config(cls, config):
        """
        Not supported: `encoder`/`projector` are required constructor args
        and are not JSON-serializable, so a trainer can't be rebuilt from
        `get_config()` alone. Build them explicitly, construct
        `VICRegTrainer(encoder=..., projector=..., ...)`, then call
        `load_weights(...)`. This method exists only to satisfy Keras'
        serialization contract and avoid warnings during weight saving.
        """
        raise NotImplementedError(
            "VICRegTrainer cannot be constructed from config alone. "
            "Build encoder/projector explicitly, create VICRegTrainer(encoder=..., projector=..., ...), "
            "then call `load_weights(...)` if needed."
        )
