"""
Model builders and trainer for VICReg / Adaptive VICReg.

Purpose
-------
I keep all model-related code here: the encoder backbone, the projector MLP,
and a subclassed Keras `Model` (VICRegTrainer) that runs forward + loss +
metrics in `train_step`. This file is intentionally self-contained so my
training/eval scripts can import the exact same builders.

What I expose
-------------
- build_encoder(image_size, feat_dim)  -> tf.keras.Model producing a vector named "feat"
- build_projector(feat_dim, proj_out, proj_layers) -> tf.keras.Model ("proj" head)
- VICRegTrainer(encoder, projector, ...) -> subclassed model with VICReg loss

Design notes
------------
- The encoder outputs a *named* tensor "feat". My downstream scripts rely on
  this name when loading encoder-only weights and extracting features.
- The projector is an MLP with 1–3 layers; the final layer is linear (no act).
- `VICRegTrainer.train_step` computes z1/z2 from two augmented views, computes
  VICReg total loss (and parts), updates metrics, and returns a logs dict so
  Keras shows "loss", "l_align", "l_var", "l_cov" per step/epoch.
- I avoid `float(tensor)` so this code works in both eager and graph modes.

Example (building only)
-----------------------
>>> enc = build_encoder(image_size=32, feat_dim=2048)
>>> proj = build_projector(feat_dim=2048, proj_out=4096, proj_layers=3)

Example (trainer)
-----------------
>>> trainer = VICRegTrainer(
...     encoder=enc,
...     projector=proj,
...     w0=VICRegWeights(sim=25.0, var=25.0, cov=1.5),
...     adaptive=True,               # enable gamma/nu scheduling (internal simple schedule)
...     use_schedules=True,          # cosmetic flag; LR/WD scaled in my callback
...     steps_per_epoch=390,
...     epochs=100,
...     base_lr=0.01,
...     base_wd=1e-6,
... )
>>> trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

Author: Nishant Kabra
Date: 11/17/2025
"""
from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

# I import losses + weights container from my vicreg_tf package.
from .losses import vicreg_total, VICRegWeights


# -----------------------------------------------------------------------------
# Encoder builder
# -----------------------------------------------------------------------------
def build_encoder(image_size: int, feat_dim: int) -> tf.keras.Model:
    """
    Build the image encoder backbone that outputs a feature vector named "feat".

    Why I build it this way
    -----------------------
    I want a simple, reliable CNN that:
      1) handles small CIFAR crops cleanly,
      2) ends with a GlobalAveragePooling (spatial -> vector),
      3) produces a Dense(feat_dim, name="feat") vector that downstream
         code can load by name for linear/knn evaluation.

    Args
    ----
    image_size : int
        Square input size (e.g., 32 for CIFAR).
    feat_dim : int
        Width of the final feature vector.

    Returns
    -------
    tf.keras.Model
        Keras model with input shape [None, image_size, image_size, 3] and an
        output tensor named "feat".
    """
    # Input is a standard HWC image; I keep it float-friendly through the pipeline.
    inp = L.Input(shape=(image_size, image_size, 3), name="image")

    # A small conv tower (CIFAR-friendly). I keep channels modest to fit GPUs.
    # I use Conv->BN->ReLU blocks and occasional stride-2 to downsample.
    x = L.Conv2D(64, 3, padding="same", use_bias=False)(inp)  # first conv
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    x = L.Conv2D(128, 3, padding="same", strides=2, use_bias=False)(x)  # downsample
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    x = L.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    x = L.Conv2D(256, 3, padding="same", strides=2, use_bias=False)(x)  # downsample
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    x = L.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # Global average pooling converts [H,W,C] -> [C]
    x = L.GlobalAveragePooling2D(name="gap")(x)

    # Final feature vector; I deliberately name it "feat" because my evaluation
    # scripts (linear/kNN) load this tensor by name after loading weights.
    feat = L.Dense(feat_dim, activation=None, name="feat")(x)

    # I return a clean Model with a single output 'feat'.
    return keras.Model(inp, feat, name="encoder")


# -----------------------------------------------------------------------------
# Projector builder
# -----------------------------------------------------------------------------
def _mlp_block(units: int) -> keras.Sequential:
    """A tiny helper to create "Dense -> BN -> ReLU" as a block for the MLP."""
    return keras.Sequential(
        [
            L.Dense(units, use_bias=False),
            L.BatchNormalization(),
            L.ReLU(),
        ],
        name=f"mlp_{units}",
    )


def build_projector(feat_dim: int, proj_out: int, proj_layers: int) -> keras.Model:
    """
    Build the projection head (MLP) for VICReg.

    I keep the exact computation pattern (proj_layers blocks, final linear),
    but I assign unique names to every layer so Keras never complains about
    duplicate names like "mlp_2048".

    Args:
        feat_dim: Width of encoder features fed into the projector.
        proj_out: Hidden & output width of the MLP.
        proj_layers: Number of MLP layers (typical VICReg uses 3).

    Returns:
        Keras Model mapping R^{feat_dim} -> R^{proj_out}.
    """
    # Input is a 1D feature vector from the encoder
    inp = keras.Input(shape=(feat_dim,), name="proj_in")

    x = inp
    # For proj_layers > 1, build (proj_layers-1) hidden blocks:
    # Dense (no bias) -> BatchNorm -> ReLU
    # Each layer gets a unique name with an index suffix.
    num_hidden = max(0, proj_layers - 1)
    for i in range(num_hidden):
        x = keras.layers.Dense(
            proj_out,
            use_bias=False,
            name=f"proj_dense_{i}"
        )(x)
        x = keras.layers.BatchNormalization(name=f"proj_bn_{i}")(x)
        x = keras.layers.ReLU(name=f"proj_relu_{i}")(x)

    # Final linear layer to proj_out (standard VICReg projector ending)
    out = keras.layers.Dense(proj_out, name="proj_out")(x)

    # Model name "proj" is fine; all inner layers now have unique names.
    return keras.Model(inp, out, name="proj")


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class VICRegTrainer(keras.Model):
    """
    Keras Model wrapper that implements VICReg training in `train_step`.

    What this class does for me
    ---------------------------
    - Accepts two augmented views (x1, x2),
    - runs encoder+projector to produce z1, z2,
    - computes VICReg total loss (and the three components),
    - updates metrics so Keras history contains loss/align/var/cov,
    - returns a logs dict so callbacks can see those values by name.

    Parameters
    ----------
    encoder : tf.keras.Model
        Backbone CNN that maps images -> feature vector "feat".
    projector : tf.keras.Model
        MLP head that maps "feat" -> projection "proj_out".
    w0 : VICRegWeights
        Base weights for (sim, var, cov) in the total loss.
    adaptive : bool
        If True, I schedule gamma (variance floor) and nu (off-diag target)
        with a simple time-based schedule. If False, I use constants:
        gamma=1.0, nu=0.0
    use_schedules : bool
        Cosmetic flag carried for completeness; LR/WD are actually handled by
        my training callback (CosineScheduleCallback) outside of this class.
    steps_per_epoch : int
        Steps in each epoch; used to compute overall progress fraction.
    epochs : int
        Total epochs; used to compute overall progress fraction.
    base_lr : float
        Remembered only for logs/serialization; not used directly inside.
    base_wd : float
        Remembered only for logs/serialization; not used directly inside.
    """

    def __init__(
        self,
        encoder: tf.keras.Model,
        projector: tf.keras.Model,
        w0: VICRegWeights,
        adaptive: bool,
        use_schedules: bool,
        steps_per_epoch: int,
        epochs: int,
        base_lr: float,
        base_wd: float,
    ) -> None:
        super().__init__(name="vicreg_trainer")
        self.encoder = encoder
        self.projector = projector

        # I store base VICReg weights (sim/var/cov) for the loss calculation.
        self.w0 = w0

        # Flags + run geometry. I keep total steps to compute progress fraction.
        self.adaptive = bool(adaptive)
        self.use_schedules = bool(use_schedules)
        self.steps_per_epoch = int(steps_per_epoch)
        self.epochs = int(epochs)
        self.total_steps = max(1, self.steps_per_epoch * self.epochs)

        # I keep LR/WD for completeness; actual scaling happens in a callback.
        self.base_lr = float(base_lr)
        self.base_wd = float(base_wd)

        # I set up Keras Metric objects so they appear in `model.history.history`.
        # The names match what my plotting utilities expect ("loss", "l_*").
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.align_tracker = keras.metrics.Mean(name="l_align")
        self.var_tracker = keras.metrics.Mean(name="l_var")
        self.cov_tracker = keras.metrics.Mean(name="l_cov")

    @property
    def metrics(self):
        """
        Keras inspects this to know which stateful metrics I track/reset.
        Returning these ensures they show up in logs and are reset per epoch.
        """
        return [self.loss_tracker, self.align_tracker, self.var_tracker, self.cov_tracker]

    # ------------------------ helpers: schedules for gamma/nu -------------------
    def _progress_frac(self) -> tf.Tensor:
        """
        Compute a [0,1] progress fraction based on optimizer iterations.

        I avoid Python floats here. I convert to a tf.float32 tensor so this
        works in graph mode and can be used inside `train_step`.
        """
        # `self.optimizer.iterations` is the canonical step counter in Keras.
        step = tf.cast(self.optimizer.iterations, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        # I clip in [0,1] to avoid any numerical drift after training ends.
        return tf.clip_by_value(step / tf.maximum(1.0, total), 0.0, 1.0)

    def _gamma_nu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return (gamma, nu) as tensors.

        If `adaptive` is True, I run a very light schedule:
        - gamma(t) = 1.0  (kept constant by default; easy to tweak later)
        - nu(t)    = 0.0  (kept constant by default)

        The key point is: I return TF tensors, not Python floats, so the
        downstream math in `losses.py` stays graph-friendly.
        """
        if not self.adaptive:
            # Non-adaptive: I return simple constants as tensors.
            return tf.constant(1.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

        # Adaptive: I *could* vary these with time. Right now I keep the
        # defaults but in a TF graph-friendly way.
        t = self._progress_frac()  # I compute this to keep the door open for future schedules
        _ = t  # unused; placeholder for future schedule tweaks
        gamma = tf.constant(1.0, dtype=tf.float32)
        nu = tf.constant(0.0, dtype=tf.float32)
        return gamma, nu

    # ------------------------------- train_step --------------------------------
    def train_step(self, data):
        """
        One gradient update on a batch of two augmented views.

        Expected input
        --------------
        `data` should be a tuple/list `(x1, x2)` where each is a batch of images
        in [0,1] float32 (my `tf.data` pipeline produces that).

        What I do
        ---------
        - Forward pass both views through encoder + projector -> z1, z2
        - Compute VICReg total loss and individual parts
        - Apply gradients
        - Update Keras metrics and return a logs dict for Keras/Callbacks
        """
        # Unpack the two views; many pipelines yield a tuple, so I handle both.
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            x1, x2 = data[0], data[1]
        else:
            # If only one tensor is provided, I duplicate it (defensive).
            x1 = x2 = data

        # Make sure everything is float32 to avoid dtype surprises downstream.
        x1 = tf.cast(x1, tf.float32)
        x2 = tf.cast(x2, tf.float32)

        # I compute scheduled targets (gamma/nu) as TENSORS (graph-friendly).
        gamma, nu = self._gamma_nu()

        with tf.GradientTape() as tape:
            # Forward view 1
            f1 = self.encoder(x1, training=True)           # feature vector
            z1 = self.projector(f1, training=True)         # projection

            # Forward view 2
            f2 = self.encoder(x2, training=True)
            z2 = self.projector(f2, training=True)

            # Compute VICReg loss + parts using my imported loss helper.
            total, parts = vicreg_total(z1, z2, self.w0, gamma=gamma, nu=nu)

        # Standard Keras gradient application.
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update my trackers so Keras logs the scalars every step/epoch.
        self.loss_tracker.update_state(total)
        self.align_tracker.update_state(parts["l_align"])
        self.var_tracker.update_state(parts["l_var"])
        self.cov_tracker.update_state(parts["l_cov"])

        # I return a dict so the values show up in the progress bar and in history.
        return {
            "loss": self.loss_tracker.result(),
            "l_align": self.align_tracker.result(),
            "l_var": self.var_tracker.result(),
            "l_cov": self.cov_tracker.result(),
        }

    # ------------------------------- test_step ---------------------------------
    def test_step(self, data):
        """
        Optional eval step that mirrors `train_step` without applying gradients.

        I keep this so I can call `model.evaluate(...)` if I ever hook up a
        validation stream of two-view batches.
        """
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            x1, x2 = data[0], data[1]
        else:
            x1 = x2 = data

        x1 = tf.cast(x1, tf.float32)
        x2 = tf.cast(x2, tf.float32)

        gamma, nu = self._gamma_nu()

        f1 = self.encoder(x1, training=False)
        z1 = self.projector(f1, training=False)
        f2 = self.encoder(x2, training=False)
        z2 = self.projector(f2, training=False)

        total, parts = vicreg_total(z1, z2, self.w0, gamma=gamma, nu=nu)

        self.loss_tracker.update_state(total)
        self.align_tracker.update_state(parts["l_align"])
        self.var_tracker.update_state(parts["l_var"])
        self.cov_tracker.update_state(parts["l_cov"])

        return {
            "loss": self.loss_tracker.result(),
            "l_align": self.align_tracker.result(),
            "l_var": self.var_tracker.result(),
            "l_cov": self.cov_tracker.result(),
        }

    # ----------------------------- serialization -------------------------------
    def get_config(self) -> Dict:
        """
        Minimal config dict so Keras does not complain during save/clone.

        I only store simple JSON-serializable values; the actual submodels
        (encoder/projector) are weights-saved separately by my scripts.
        """
        return {
            "name": self.name,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "total_steps": self.total_steps,
            "adaptive": self.adaptive,
            "use_schedules": self.use_schedules,
            "base_lr": self.base_lr,
            "base_wd": self.base_wd,
            "w0": {"sim": float(self.w0.sim), "var": float(self.w0.var), "cov": float(self.w0.cov)},
        }
