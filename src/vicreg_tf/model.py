"""Encoder and projector builders, and the VICReg training loop.

`VICRegTrainer` runs the two-view forward pass, computes the loss terms and optimizes the encoder and
projector. Two independent options sit on top of plain VICReg:

- `adaptive_weights`: per-step sim/var/cov weights from `schedules.AdaptiveReweighter`. This is what the
  README calls Adaptive VICReg. When off, the weights stay at `w0`.
- `adaptive_targets`: the variance floor `gamma` and correlation target `nu` follow
  `schedules.AdaptiveTargets` instead of the constants 1.0 and 0.0.

With both off this is baseline VICReg.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras

from .losses import VICRegWeights, vicreg_total
from .schedules import AdaptiveReweighter, AdaptiveTargets, WeightSchedules


def _conv_block(x, filters, k=3, s=1, name=None):
    """Conv -> BatchNorm -> ReLU."""
    def layer_name(suffix):
        return None if name is None else f"{name}_{suffix}"

    x = keras.layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False, name=layer_name("conv"))(x)
    x = keras.layers.BatchNormalization(name=layer_name("bn"))(x)
    return keras.layers.ReLU(name=layer_name("relu"))(x)


def build_encoder(image_size: int, feat_dim: int = 2048) -> keras.Model:
    """Small CNN encoder: three stages of two conv blocks, global average pooling, then a linear layer to `feat_dim`.

    Outputs feature vectors, not logits.
    """
    inp = keras.Input(shape=(image_size, image_size, 3), name="enc_in")

    x = _conv_block(inp, 64, 3, 1, name="s1a")
    x = _conv_block(x, 64, 3, 1, name="s1b")

    x = _conv_block(x, 128, 3, 2, name="s2a")
    x = _conv_block(x, 128, 3, 1, name="s2b")

    x = _conv_block(x, 256, 3, 2, name="s3a")
    x = _conv_block(x, 256, 3, 1, name="s3b")

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    feat = keras.layers.Dense(feat_dim, use_bias=False, name="enc_out")(x)

    return keras.Model(inp, feat, name="encoder")


def _mlp(feat_in: int, proj_out: int, proj_layers: int, name_prefix: str) -> keras.Sequential:
    """MLP with 1, 2 or 3 dense layers (other values give 3). No BatchNorm, so it behaves the same at any batch size."""
    n_hidden = {1: 0, 2: 1}.get(proj_layers, 2)
    layers = [
        keras.layers.Dense(feat_in, activation="relu", name=f"{name_prefix}_dense_h{i + 1}") for i in range(n_hidden)
    ]
    layers.append(keras.layers.Dense(proj_out, activation=None, name=f"{name_prefix}_dense_out"))
    return keras.Sequential(layers, name=f"{name_prefix}_mlp")


def build_projector(feat_in: int, proj_out: int, proj_layers: int) -> keras.Model:
    """Projector MLP as a Keras Model. Layer names carry a unique suffix so rebuilding in one process doesn't clash."""
    uniq = hex(id(object()))[-6:]
    inp = keras.Input(shape=(feat_in,), name=f"proj_in_{uniq}")
    mlp = _mlp(feat_in, proj_out, proj_layers, name_prefix=f"proj_{feat_in}_{proj_out}_{proj_layers}_{uniq}")
    out = mlp(inp)
    return keras.Model(inp, out, name=f"proj_{uniq}")


class VICRegTrainer(keras.Model):
    """Keras model that trains an encoder and projector with (Adaptive) VICReg.

    Args:
        encoder: Backbone.
        projector: MLP head mapping encoder features to the embedding space.
        w0: Base (sim, var, cov) weights.
        adaptive_weights: Take per-step weights from `AdaptiveReweighter`. Otherwise use `w0`.
        adaptive_targets: Vary gamma and nu with `AdaptiveTargets`. Otherwise use gamma=1.0, nu=0.0.
        use_schedules: Whether the LR/WD cosine schedules are in use.
        steps_per_epoch: Steps per epoch.
        epochs: Total training epochs.
        base_lr: Base learning rate.
        base_wd: Base weight decay.
        reweighter_kwargs: Overrides passed to `AdaptiveReweighter`.
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

        # Constant weights, used when adaptive_weights is off.
        self.schedules = WeightSchedules(
            w0=w0,
            use=self.use_schedules,
            base_lr=float(base_lr),
            base_wd=float(base_wd),
            total_steps=self.total_steps,
            warmup_frac=0.0,
            min_scale=0.0,
        )

        self.reweighter = (
            AdaptiveReweighter(w0=w0, **(reweighter_kwargs or {})) if self.adaptive_weights else None
        )

        self.targets = AdaptiveTargets(use=self.adaptive_targets)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.align_tracker = keras.metrics.Mean(name="l_align")
        self.var_tracker = keras.metrics.Mean(name="l_var")
        self.cov_tracker = keras.metrics.Mean(name="l_cov")
        # The weights actually used, so an adaptive run can be inspected from its metrics history.
        self.w_sim_tracker = keras.metrics.Mean(name="w_sim")
        self.w_var_tracker = keras.metrics.Mean(name="w_var")
        self.w_cov_tracker = keras.metrics.Mean(name="w_cov")

    @property
    def metrics(self):
        """Metrics that Keras resets at the start of each epoch."""
        return [
            self.loss_tracker, self.align_tracker, self.var_tracker, self.cov_tracker,
            self.w_sim_tracker, self.w_var_tracker, self.w_cov_tracker,
        ]

    def compile(self, optimizer: keras.optimizers.Optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data) -> dict:
        """One step on a batch of paired views, `data = (x1, x2)`, each [B, H, W, 3]."""
        x1, x2 = data[0], data[1]

        frac = tf.cast(self.curr_step, tf.float32) / tf.cast(self.total_steps, tf.float32)

        with tf.GradientTape() as tape:
            h1 = self.encoder(x1, training=True)
            h2 = self.encoder(x2, training=True)
            z1 = self.projector(h1, training=True)
            z2 = self.projector(h2, training=True)

            gamma = self.targets.gamma(frac)
            nu = self.targets.nu(frac)

            # Unweighted terms first: the reweighter needs the raw magnitudes to pick the weights.
            _, parts = vicreg_total(
                z1, z2,
                w=VICRegWeights(sim=1.0, var=1.0, cov=1.0),
                gamma=gamma,
                nu=nu,
            )

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

        variables = self.encoder.trainable_variables + self.projector.trainable_variables
        grads = tape.gradient(total, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

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
        """Serializable summary of the construction settings.

        Only weights are ever saved and loaded, so this is not used to rebuild the model. It exists so Keras
        doesn't warn about a missing `get_config`. The encoder, projector and optimizer are left out; they
        are restored from weights.
        """
        return {
            "class_name": self.__class__.__name__,
            "adaptive_weights": self.adaptive_weights,
            "adaptive_targets": self.adaptive_targets,
            "use_schedules": self.use_schedules,
            "w0": {"sim": float(self.w0.sim), "var": float(self.w0.var), "cov": float(self.w0.cov)},
            "total_steps": self.total_steps,
            "warmup_frac": float(self.schedules.warmup_frac),
            "min_scale": float(self.schedules.min_scale),
            "base_lr": float(self.schedules.base_lr),
            "base_wd": float(self.schedules.base_wd),
        }

    @classmethod
    def from_config(cls, config):
        """Not supported: the encoder and projector aren't serializable.

        Build them, construct `VICRegTrainer(encoder=..., projector=..., ...)`, then call `load_weights`.
        """
        raise NotImplementedError(
            "VICRegTrainer cannot be built from a config alone. Build the encoder and projector, construct "
            "VICRegTrainer(encoder=..., projector=..., ...), then call load_weights(...)."
        )
