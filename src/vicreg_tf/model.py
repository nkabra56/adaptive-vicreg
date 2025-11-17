"""
Models for VICReg: encoder, projector, and the trainer wrapper.

The trainer is a subclassed Keras.Model with a custom train_step. It implements
`get_config()` and `from_config()` to keep Keras 3 quiet when saving, even
though we typically save **weights only** for this project.
"""

from __future__ import annotations
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .losses import VICRegWeights, vicreg_total
from .schedules import AdaptiveTargets, WeightSchedules


def _named_conv_block(x: tf.Tensor, filters: int, conv_name: str, bn_name: str) -> tf.Tensor:
    """Conv2D -> BatchNorm -> ReLU with explicit layer names."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=conv_name)(x)
    x = layers.BatchNormalization(name=bn_name)(x)
    x = layers.ReLU()(x)
    return x


def build_encoder(image_size: int, feat_dim: int = 2048) -> keras.Model:
    """
    Tiny CIFAR-friendly encoder with stable layer names.

    Args:
        image_size: Input height/width.
        feat_dim: Final feature width before the projector.

    Returns:
        Keras Model mapping images -> feature vectors.
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
    feat = layers.Lambda(lambda t: t, name="feat")(f)  # identity to expose name
    return keras.Model(inp, feat, name="encoder")


def build_projector(in_dim: int, out_dim: int, num_layers: int) -> keras.Model:
    """
    MLP projector with fixed, evaluation-friendly names.

    Args:
        in_dim: Input width (encoder feature dim).
        out_dim: Projection width.
        num_layers: 1, 2, or 3 layers.

    Returns:
        Keras Model mapping features -> projections.
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


class VICRegTrainer(keras.Model):
    """
    Wrap encoder + projector and implement a custom train_step.

    Notes
    -----
    - We only save **weights** in this project. `get_config` exists to avoid
      Keras warnings and to record training hyperparameters in checkpoints.
    - `encoder` and `projector` are held as submodules and are not serialized
      by config; their weights are part of `save_weights`.
    """
    
    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model,
        w0: VICRegWeights = VICRegWeights(sim=25.0, var=25.0, cov=1.0),
        adaptive: bool = False,
        use_schedules: bool = False,
        steps_per_epoch: int = 1000,
        epochs: int = 100,
        base_lr: float = 0.1,
        base_wd: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projector = projector
        self.adaptive = bool(adaptive)
        self.use_schedules = bool(use_schedules)
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_steps = int(steps_per_epoch) * int(epochs)

        # --- schedules / targets
        self.schedules = WeightSchedules(
            w0=w0,
            use=self.use_schedules,
            base_lr=float(base_lr),
            base_wd=float(base_wd),
            total_steps=self.total_steps,
        )
        self.targets = AdaptiveTargets(use=self.adaptive)

        # book-keeping
        self.curr_step = tf.Variable(0, dtype=tf.int64, trainable=False)

        # metrics that I expose to the Keras logs
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.align_tracker = keras.metrics.Mean(name="l_align")
        self.var_tracker = keras.metrics.Mean(name="l_var")
        self.cov_tracker = keras.metrics.Mean(name="l_cov")


    # ---- Keras bookkeeping ----
    @property
    def metrics(self):
        # Keras 3 will reset and log these automatically each epoch/step.
        return [self.loss_tracker, self.align_tracker, self.var_tracker, self.cov_tracker]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict describing training hyperparameters.

        This avoids the Keras warning about non-serializable __init__ args and
        helps future runs verify that the trainer was built with the expected
        settings. Submodels are intentionally omitted from the config.
        """
        return {
            "w0": {"sim": self.w0.sim, "var": self.w0.var, "cov": self.w0.cov},
            "adaptive": self.adaptive,
            "use_schedules": isinstance(self.schedules, WeightSchedules),
            "total_steps": int(self.total_steps),
            "bn_freeze_steps": int(self.bn_freeze_steps),
            "base_lr": float(self.schedules.base_lr),
            "base_wd": float(self.schedules.base_wd),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VICRegTrainer":
        """
        Recreate with placeholder submodules.

        The encoder and projector must be set by the caller before use. This is
        sufficient for our use case because we always rebuild modules and then
        call `load_weights`.
        """
        # Build minimal placeholders; caller should replace them.
        dummy_in = keras.Input(shape=(32, 32, 3))
        enc = keras.Model(dummy_in, dummy_in, name="encoder_placeholder")
        prj = keras.Model(keras.Input(shape=(32, 32, 3)), keras.Input(shape=(32, 32, 3)), name="proj_placeholder")
        w = VICRegWeights(**config.get("w0", {"sim": 25.0, "var": 25.0, "cov": 1.0}))
        return cls(
            encoder=enc,
            projector=prj,
            w0=w,
            adaptive=config.get("adaptive", False),
            use_schedules=config.get("use_schedules", False),
            steps_per_epoch=max(1, config.get("total_steps", 1)),
            epochs=1,
            base_lr=config.get("base_lr", 0.001),
            base_wd=config.get("base_wd", 0.0),
            bn_freeze_steps=config.get("bn_freeze_steps", 0),
        )

    # ---- Forward + train step ----
    def call(self, inputs, training=None):
        """
        Forward pass for two-view batches.

        Args:
            inputs: Tuple (x1, x2) of augmented image batches.
            training: Whether to run in training mode.

        Returns:
            Tuple (z1, z2) of projected features.
        """
        x1, x2 = inputs
        # Optionally "freeze" BN updates for a warm start
        bn_train = (self.curr_step >= self.bn_freeze_steps)
        f1 = self.encoder(x1, training=bn_train if training is None else training)
        f2 = self.encoder(x2, training=bn_train if training is None else training)
        z1 = self.projector(f1, training=training)
        z2 = self.projector(f2, training=training)
        return z1, z2

    def train_step(self, data):
        # Expect two augmented views (x1, x2)
        (x1, x2) = data if isinstance(data, (tuple, list)) else (data, data)

        with tf.GradientTape() as tape:
            h1 = self.encoder(x1, training=True)
            h2 = self.encoder(x2, training=True)
            z1 = self.projector(h1, training=True)
            z2 = self.projector(h2, training=True)

            # progress fraction in [0,1] as a Tensor (works in graph mode)
            frac = tf.cast(self.curr_step, tf.float32) / tf.cast(self.total_steps, tf.float32)

            # weights + adaptive targets (gamma/nu may be Tensors)
            w = self.schedules.weights(frac)                  # dict with sim/var/cov
            gamma = self.targets.gamma(frac) if self.adaptive else tf.constant(1.0, tf.float32)
            nu    = self.targets.nu(frac)    if self.adaptive else tf.constant(0.0, tf.float32)

            total, parts = vicreg_total(z1, z2, w, gamma=gamma, nu=nu)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # update trackers
        self.loss_tracker.update_state(total)
        self.align_tracker.update_state(parts["l_align"])
        self.var_tracker.update_state(parts["l_var"])
        self.cov_tracker.update_state(parts["l_cov"])

        # step++
        self.curr_step.assign_add(1)

        return {
            "loss": self.loss_tracker.result(),
            "l_align": self.align_tracker.result(),
            "l_var": self.var_tracker.result(),
            "l_cov": self.cov_tracker.result(),
        }
