"""
Model and trainer definitions for VICReg / Adaptive-VICReg.

What lives here (my words)
--------------------------
1) `build_encoder(image_size, feat_dim)`:
   A lightweight CNN encoder that outputs a `feat_dim`-wide feature vector.
   I keep it simple, CIFAR-friendly, and deterministic.

2) `build_projector(feat_in, proj_out, proj_layers)`:
   A small MLP head (1-3 layers) that maps encoder features to projection space.

3) `VICRegTrainer(keras.Model)`:
   A custom Keras model that:
     • runs the two-view forward pass,
     • computes VICReg losses (alignment, variance, covariance),
     • optionally *adapts* the variance floor `gamma` and corr target `nu`
       over training progress (this is my **adaptive VICReg** implementation),
     • applies schedules and records scalar metrics for logging.

Baseline vs Adaptive
--------------------
- Baseline VICReg is the exact same code path with:
    gamma = 1.0, nu = 0.0            (hard-coded constants)
  I get baseline by simply *omitting* `--adaptive` in the training script.

- Adaptive VICReg (my method) switches `gamma` and `nu` to *time-varying*
  signals driven by schedulers in `vicreg_tf.schedules.AdaptiveTargets`.
  The switch is controlled by a boolean feature-flag passed from the CLI.

Author: Nishant Kabra
Date: 11/18/2025
"""
from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras

# Loss function (and weights container) live in vicreg_tf.losses.
from .losses import vicreg_total, VICRegWeights

# Schedules and adaptive targets are constructed by the trainer.
from .schedules import WeightSchedules, AdaptiveTargets


# =============================================================================
# Encoder & Projector builders
# =============================================================================
def _conv_block(x, filters, k=3, s=1, name=None):
    """A small helper: Conv -> BN -> ReLU block suited for CIFAR-sized inputs."""
    x = keras.layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False, name=None if name is None else f"{name}_conv")(x)
    x = keras.layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = keras.layers.ReLU(name=None if name is None else f"{name}_relu")(x)
    return x


def build_encoder(image_size: int, feat_dim: int = 2048) -> keras.Model:
    """
    Build my CIFAR-friendly encoder.

    I keep this deterministic and lean: a few Conv blocks, downsampling, GAP,
    then a final Dense to `feat_dim`. This outputs a *feature vector* (no logits).

    Parameters
    ----------
    image_size : int
        Spatial size of inputs (e.g., 32 for CIFAR-10).
    feat_dim : int
        Output width of the feature vector.

    Returns
    -------
    keras.Model
        The encoder backbone producing a `[None, feat_dim]` feature vector.
    """
    inp = keras.Input(shape=(image_size, image_size, 3), name="enc_in")

    # A tiny CNN: two convs per stage; stride=2 to downsample at stages 2 & 3.
    x = _conv_block(inp, 64, 3, 1, name="s1a")
    x = _conv_block(x,   64, 3, 1, name="s1b")

    x = _conv_block(x,  128, 3, 2, name="s2a")
    x = _conv_block(x,  128, 3, 1, name="s2b")

    x = _conv_block(x,  256, 3, 2, name="s3a")
    x = _conv_block(x,  256, 3, 1, name="s3b")

    # Global Average Pooling collapses HxW into a single feature per channel.
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # Final Dense projects to the requested feat_dim.
    feat = keras.layers.Dense(feat_dim, use_bias=False, name="enc_out")(x)

    return keras.Model(inp, feat, name="encoder")


def _mlp(feat_in: int, proj_out: int, proj_layers: int, name_prefix: str) -> keras.Sequential:
    """
    Construct a small MLP with 1-3 layers for the projector.

    I deliberately avoid BatchNorm here to keep behavior simple and compatible
    with variable batch sizes on smaller GPUs.

    Parameters
    ----------
    feat_in : int
        Input feature width from the encoder.
    proj_out : int
        Output width of the projection vector.
    proj_layers : int
        1, 2, or 3 layers.
    name_prefix : str
        Prefix used to make layer names unique across multiple builds.

    Returns
    -------
    keras.Sequential
        A small MLP mapping `feat_in` -> `proj_out`.
    """
    layers = []
    if proj_layers == 1:
        layers += [keras.layers.Dense(proj_out, activation=None, name=f"{name_prefix}_dense_out")]
    elif proj_layers == 2:
        layers += [
            keras.layers.Dense(feat_in, activation="relu", name=f"{name_prefix}_dense_h1"),
            keras.layers.Dense(proj_out, activation=None, name=f"{name_prefix}_dense_out"),
        ]
    else:  # 3 or more -> clamp to 3
        layers += [
            keras.layers.Dense(feat_in, activation="relu", name=f"{name_prefix}_dense_h1"),
            keras.layers.Dense(feat_in, activation="relu", name=f"{name_prefix}_dense_h2"),
            keras.layers.Dense(proj_out, activation=None, name=f"{name_prefix}_dense_out"),
        ]
    return keras.Sequential(layers, name=f"{name_prefix}_mlp")


def build_projector(feat_in: int, proj_out: int, proj_layers: int) -> keras.Model:
    """
    Wrap the MLP into a Keras Model for clarity.

    I add a unique name prefix to avoid name clashes if the projector is rebuilt.
    """
    # Unique suffix helps avoid "layer name used twice" if graph is rebuilt in place.
    uniq = hex(id(object()))[-6:]
    inp = keras.Input(shape=(feat_in,), name=f"proj_in_{uniq}")
    mlp = _mlp(feat_in, proj_out, proj_layers, name_prefix=f"proj_{feat_in}_{proj_out}_{proj_layers}_{uniq}")
    out = mlp(inp)
    return keras.Model(inp, out, name=f"proj_{uniq}")


# =============================================================================
# Trainer: VICReg (baseline) + Adaptive targets (feature-flag on)
# =============================================================================
class VICRegTrainer(keras.Model):
    """
    Keras Model wrapper that implements (Adaptive) VICReg training.

    Responsibilities (step-by-step)
    -------------------------------
    • Forward pass: run (x1, x2) through encoder->projector to get z1, z2.
    • Loss: call `vicreg_total(z1, z2, weights, gamma, nu)`.
      - `gamma` and `nu` come from my **adaptive** schedulers *iff* `adaptive=True`.
      - Otherwise (`adaptive=False`) I use *constants* (gamma=1.0, nu=0.0) for baseline.
    • Optimize: compute gradients on encoder+projector and apply opt step.
    • Metrics: track and return loss terms so callbacks can log them.

    Parameters
    ----------
    encoder : keras.Model
        My backbone.
    projector : keras.Model
        My MLP head for projection space.
    w0 : VICRegWeights
        The base weights for (sim, var, cov) components of VICReg.
    adaptive : bool
        If True, enable time-varying gamma/nu via `AdaptiveTargets` (my method).
        If False, run exact baseline VICReg using constants gamma=1.0, nu=0.0.
    use_schedules : bool
        Whether to scale *optimizer* lr/wd via external cosine schedules. Loss
        weights (w0) are currently kept constant; you could extend them too.
    steps_per_epoch : int
        Steps in an epoch (for progress calculation).
    epochs : int
        Number of epochs (for global step and reporting).
    base_lr : float
        For logging and for schedules.
    base_wd : float
        For logging and for schedules.
    """
    def __init__(
        self,
        *,
        encoder: keras.Model,
        projector: keras.Model,
        w0: VICRegWeights,
        adaptive: bool,
        use_schedules: bool,
        steps_per_epoch: int,
        epochs: int,
        base_lr: float,
        base_wd: float,
    ):
        super().__init__(name="vicreg_trainer")
        self.encoder = encoder
        self.projector = projector

        # A copy of the user-provided base weights (sim/var/cov).
        self.w0 = w0

        # Feature flags controlling *targets* (gamma/nu) and external LR/WD schedules.
        self.adaptive = bool(adaptive)
        self.use_schedules = bool(use_schedules)

        # Bookkeeping for step-based progress calculation.
        self.steps_per_epoch = int(steps_per_epoch)
        self.epochs = int(epochs)
        self.total_steps = int(self.steps_per_epoch * self.epochs)
        self.curr_step = tf.Variable(0, dtype=tf.int64, trainable=False)

        # --------------------------
        # (A) Loss weight schedules.
        # --------------------------
        # I use constant VICReg weights by default (weights() returns w0).
        # You could wire these to cosine too (e.g., down-weight cov later).
        self.schedules = WeightSchedules(
            w0=w0,
            use=self.use_schedules,
            base_lr=float(base_lr),
            base_wd=float(base_wd),
            total_steps=self.total_steps,
            warmup_frac=0.0,
            min_scale=0.0,
        )

        # -------------------------------------------------------
        # (B) Adaptive targets (my method) for gamma and nu.
        # -------------------------------------------------------
        # If `self.adaptive == True`, gamma/nu are time-varying. Otherwise,
        # I will not read them from here in `train_step`; I will use constants.
        self.targets = AdaptiveTargets(
            use=self.adaptive,  # <── key feature-flag controlling adaptive behavior
            # Schedulers are set to identity-like defaults (scale in [min,1]).
            # You can tune warmup_frac/min_scale externally if desired.
        )

        # Metric trackers (I keep simple Means so callbacks can read scalar logs).
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.align_tracker = keras.metrics.Mean(name="l_align")
        self.var_tracker = keras.metrics.Mean(name="l_var")
        self.cov_tracker = keras.metrics.Mean(name="l_cov")

    # Keras calls this to collect metrics it should reset each epoch.
    @property
    def metrics(self):
        return [self.loss_tracker, self.align_tracker, self.var_tracker, self.cov_tracker]

    def compile(self, optimizer: keras.optimizers.Optimizer, **kwargs):
        """Standard Keras compile; I only need an optimizer."""
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data) -> dict:
        """
        One training step on a batch of *paired* views.

        Expected `data` structure from my pipeline:
        -------------------------------------------
        data = (x1, x2) with shapes [B, H, W, 3] each (augmented independently).
        """
        # Accept both tuple/list and dict-like structures; keep it robust.
        if isinstance(data, (tuple, list)) and len(data) == 2:
            x1, x2 = data
        else:
            # Last resort: try to index; this keeps errors informative.
            x1, x2 = data[0], data[1]

        # Progress fraction in [0,1] based on global step.
        frac = tf.cast(self.curr_step, tf.float32) / tf.cast(self.total_steps, tf.float32)

        # ---------------------------
        # (1) Forward pass (no loss).
        # ---------------------------
        with tf.GradientTape() as tape:
            # Forward both views through encoder and projector (shared weights).
            h1 = self.encoder(x1, training=True)
            h2 = self.encoder(x2, training=True)
            z1 = self.projector(h1, training=True)
            z2 = self.projector(h2, training=True)

            # -------------------------------
            # (2) Adaptive targets (my method)
            # -------------------------------
            # This is the *only* place where adaptive/baseline differ.
            # • adaptive == True  -> time-varying gamma/nu from self.targets
            # • adaptive == False -> constants gamma=1.0, nu=0.0 (baseline VICReg)
            gamma = self.targets.gamma(frac) if self.adaptive else tf.constant(1.0, tf.float32)
            nu    = self.targets.nu(frac)    if self.adaptive else tf.constant(0.0, tf.float32)

            # ---------------------------
            # (3) VICReg loss computation
            # ---------------------------
            # `vicreg_total` returns total loss and a dict with the components.
            # I pass *constant* weights (self.schedules.weights(frac) returns w0).
            # You can wire weight scaling later by modifying WeightSchedules.weights.
            total, parts = vicreg_total(
                z1, z2,
                w=self.schedules.weights(frac),
                gamma=gamma,
                nu=nu,
            )

        # ---------------------------
        # (4) Apply gradients (opt step)
        # ---------------------------
        vars = self.encoder.trainable_variables + self.projector.trainable_variables
        grads = tape.gradient(total, vars)
        self.optimizer.apply_gradients(zip(grads, vars))

        # ---------------------------
        # (5) Update scalar trackers
        # ---------------------------
        self.loss_tracker.update_state(total)
        self.align_tracker.update_state(parts["l_align"])
        self.var_tracker.update_state(parts["l_var"])
        self.cov_tracker.update_state(parts["l_cov"])

        # Advance the global step counter (used for progress fraction).
        self.curr_step.assign_add(1)

        # Keras displays/returns this mapping in logs and callbacks.
        return {
            "loss": self.loss_tracker.result(),
            "l_align": self.align_tracker.result(),
            "l_var": self.var_tracker.result(),
            "l_cov": self.cov_tracker.result(),
        }
    def get_config(self):
        """
        Return a JSON-serializable snapshot of my construction-time settings.

        Why I implement this
        --------------------
        Keras warns when a subclassed Model has no `get_config()` because it
        cannot serialize the constructor args. I'm only saving **weights**
        (via `save_weights` / ModelCheckpoint with `save_weights_only=True`),
        so Keras will not *use* this config to rebuild the model; it just needs
        something serializable to stop warning.

        What I include
        --------------
        Only plain Python types (bool/int/float/str/dict) – never Trackables
        like the actual `encoder` / `projector` models or Optimizers.

        Returns
        -------
        dict
            A minimal, JSON-serializable dictionary describing run-time knobs.
        """
        # Base hyperparameters: keep them JSON-serializable
        cfg = {
            "class_name": self.__class__.__name__,          # helpful breadcrumb
            "adaptive": bool(getattr(self, "adaptive", False)),
            "use_schedules": bool(getattr(self, "use_schedules", False)),
        }

        # Loss weights (convert dataclass to plain dict of floats)
        try:
            w0 = getattr(self, "w0")
            cfg["w0"] = {"sim": float(w0.sim), "var": float(w0.var), "cov": float(w0.cov)}
        except Exception:
            cfg["w0"] = {"sim": 25.0, "var": 25.0, "cov": 1.0}

        # Training duration in steps; safer than storing steps_per_epoch/epochs separately
        total_steps = getattr(self, "total_steps", None)
        if total_steps is not None:
            try:
                cfg["total_steps"] = int(total_steps)
            except Exception:
                pass

        # Scheduler bases (if present). Fall back to attributes set in __init__.
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

        # NOTE: I deliberately do NOT include actual models/optimizers/callbacks,
        # because they are not JSON-serializable and are restored from weights.
        return cfg

    @classmethod
    def from_config(cls, config):
        """
        Rebuild a trainer from a config dict.

        Important
        ---------
        My training workflow restores **weights** into an already-constructed
        trainer that you build via your `build_encoder` / `build_projector`
        helpers. Because `encoder` and `projector` are required and not
        serializable into JSON, automatic reconstruction from this config is
        intentionally **not** supported.

        This method exists only to satisfy Keras' serialization contract and
        prevent warnings during weight saving. If someone tries to call it,
        I raise a clear error explaining the supported path.
        """
        raise NotImplementedError(
            "VICRegTrainer cannot be constructed from config alone. "
            "Build encoder/projector explicitly, create VICRegTrainer(encoder=..., projector=..., ...), "
            "then call `load_weights(...)` if needed."
        )

