"""Keras callbacks used by the pretraining scripts."""

from __future__ import annotations

import json
import os
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from .schedules import cosine_scaler

_DEFAULT_LOSS_KEYS = {
    "total": "loss",
    "align": "l_align",
    "var": "l_var",
    "cov": "l_cov",
    "w_sim": "w_sim",
    "w_var": "w_var",
    "w_cov": "w_cov",
}


def _set_hparam(optimizer: keras.optimizers.Optimizer, name: str, value: float) -> None:
    """Set an optimizer hyperparameter that is a variable (learning_rate) or a plain float (weight_decay)."""
    current = getattr(optimizer, name)
    if hasattr(current, "assign"):
        current.assign(value)
    else:
        setattr(optimizer, name, value)


class CosineScheduleCallback(keras.callbacks.Callback):
    """Scale the optimizer's learning rate, and weight decay if `base_wd` is given, on a per-step cosine schedule.

    Stepping per batch instead of per epoch avoids jumps and doesn't depend on the dataset length.

    Args:
        optimizer: Optimizer to update.
        total_steps: steps_per_epoch * epochs.
        base_lr: Peak learning rate.
        base_wd: Peak weight decay, or None to leave weight decay alone.
        verbose: If nonzero, print the current lr (and wd) at the start of each epoch.
        warmup_frac: Fraction of `total_steps` spent ramping linearly up from zero before the cosine decay.
        start_step: Step to begin at, so a resumed run continues the schedule where it stopped.
        ramp_steps: For the first this many steps of this run, scale the scheduled value from 10% up to 100%.
            This is the post-resume warmup, and it scales the schedule instead of fighting it.
    """

    def __init__(
        self,
        optimizer: keras.optimizers.Optimizer,
        total_steps: int,
        base_lr: float,
        base_wd: Optional[float],
        verbose: int = 1,
        warmup_frac: float = 0.0,
        start_step: int = 0,
        ramp_steps: int = 0,
    ) -> None:
        super().__init__()
        self.opt = optimizer
        self.total_steps = int(total_steps)
        self.base_lr = float(base_lr)
        self.base_wd = None if base_wd is None else float(base_wd)
        self.verbose = int(verbose)
        self.warmup_frac = float(warmup_frac)
        self._start = int(start_step)
        self._step = self._start
        self.ramp_steps = int(max(0, ramp_steps))

        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.verbose:
            print(f"[schedules] total_steps={self.total_steps} base_lr={self.base_lr} base_wd={self.base_wd}")

    def _scales_weight_decay(self) -> bool:
        return self.base_wd is not None and hasattr(self.opt, "weight_decay")

    def on_train_batch_begin(self, batch: int, logs=None):
        step = min(self._step, self.total_steps)
        scale = float(cosine_scaler(step=step, total_steps=self.total_steps, warmup_frac=self.warmup_frac))
        elapsed = self._step - self._start
        if elapsed < self.ramp_steps:
            scale *= 0.1 + 0.9 * (elapsed + 1) / float(self.ramp_steps)

        _set_hparam(self.opt, "learning_rate", self.base_lr * scale)
        if self._scales_weight_decay():
            _set_hparam(self.opt, "weight_decay", self.base_wd * scale)

        self._step += 1

    def on_epoch_begin(self, epoch: int, logs=None):
        if not self.verbose:
            return

        msg = f"[schedules] epoch {epoch + 1:03d} | lr={float(self.opt.learning_rate):.6f}"
        if self._scales_weight_decay():
            msg += f" wd={float(self.opt.weight_decay):.6f}"
        print(msg)


class WarmupLR(keras.callbacks.Callback):
    """Ramp the learning rate from 10% to 100% of `base_lr` over the first `warmup_steps` steps of training.

    Steps are counted from where this run starts, so a resumed run warms up too.
    """

    def __init__(self, base_lr: float, warmup_steps: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(0, warmup_steps))
        self._first_step = None

    def on_train_batch_begin(self, batch, logs=None):
        if self.warmup_steps <= 0:
            return
        step = int(self.model.curr_step)
        if self._first_step is None:
            self._first_step = step
        elapsed = step - self._first_step
        if elapsed < self.warmup_steps:
            scale = 0.1 + 0.9 * (elapsed + 1) / float(self.warmup_steps)
            _set_hparam(self.model.optimizer, "learning_rate", self.base_lr * scale)


class LossExplosionGuard(keras.callbacks.Callback):
    """Multiply the learning rate by `factor` (floor 1e-6) whenever a batch loss exceeds `threshold`."""

    def __init__(self, threshold: float = 1e8, factor: float = 0.1):
        super().__init__()
        self.threshold = float(threshold)
        self.factor = float(factor)

    def on_train_batch_end(self, batch, logs=None):
        if not logs:
            return
        loss = float(logs.get("loss", 0.0))
        if loss > self.threshold:
            lr = float(self.model.optimizer.learning_rate)
            new_lr = max(lr * self.factor, 1e-6)
            _set_hparam(self.model.optimizer, "learning_rate", new_lr)
            print(f"[loss-guard] loss={loss:.3e} > {self.threshold:.1e} -> lr {lr:.2e} -> {new_lr:.2e}")


class VicRegMetricsLogger(keras.callbacks.Callback):
    """Append one JSON record per epoch to `<run_dir>/metrics/history.jsonl` for `report_metrics.py`.

    Each record has the loss terms and realized loss weights from the Keras logs, plus
    `stats/avg_std` and `stats/avg_offdiag_corr_sq` computed on a fixed probe batch.

    Args:
        run_dir: Run directory; the log goes in its `metrics/` subfolder.
        encoder: Backbone used to embed the probe batch.
        projector: Projector head, used when `compute_on="projector"`.
        sample_images: Fixed single-view probe batch. If None, only losses are logged.
        compute_on: Where to compute embedding stats, "projector" or "encoder".
        loss_keys: Maps record names to Keras log keys. Defaults to all loss terms and weights.
        record_every: Write a record every N epochs.
    """

    def __init__(
        self,
        run_dir: str,
        encoder: keras.Model,
        projector: Optional[keras.Model],
        sample_images: Optional[tf.Tensor],
        compute_on: str = "projector",
        loss_keys: Optional[dict] = None,
        record_every: int = 1,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.encoder = encoder
        self.projector = projector
        self.sample_images = sample_images
        self.compute_on = compute_on
        self.loss_keys = loss_keys or dict(_DEFAULT_LOSS_KEYS)
        self.record_every = int(record_every)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.history_path = os.path.join(self.metrics_dir, "history.jsonl")

    def _get_embeddings(self, x: tf.Tensor) -> tf.Tensor:
        z = self.encoder(x, training=False)
        if self.compute_on == "projector" and self.projector is not None:
            z = self.projector(z, training=False)
        return z

    @staticmethod
    def _tf_avg_std(z: tf.Tensor) -> tf.Tensor:
        """Mean per-dimension std across the batch. A value near zero means the embeddings collapsed."""
        z2 = tf.reshape(z, [tf.shape(z)[0], -1])
        std = tf.math.reduce_std(z2, axis=0)
        return tf.reduce_mean(std)

    @staticmethod
    def _tf_avg_offdiag_corr_sq(z: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        """Mean squared off-diagonal entry of the correlation matrix."""
        z2 = tf.reshape(z, [tf.shape(z)[0], -1])
        n = tf.shape(z2)[0]
        d = tf.shape(z2)[1]
        mean = tf.reduce_mean(z2, axis=0, keepdims=True)
        zc = z2 - mean
        std = tf.math.reduce_std(zc, axis=0, keepdims=True)
        std = tf.where(std < eps, tf.ones_like(std), std)
        zn = zc / std
        corr = tf.matmul(zn, zn, transpose_a=True) / tf.cast(n, zn.dtype)
        eye = tf.eye(d, dtype=tf.bool)
        off = tf.boolean_mask(corr, ~eye)
        return tf.reduce_mean(tf.square(off))

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.record_every != 0:
            return

        rec = {"epoch": int(epoch + 1)}

        for pretty, key in self.loss_keys.items():
            if logs.get(key) is not None:
                try:
                    rec[f"loss/{pretty}"] = float(logs[key])
                except Exception:
                    pass

        if self.sample_images is not None:
            x = tf.cast(tf.convert_to_tensor(self.sample_images), tf.float32)
            z = self._get_embeddings(x)
            if z.dtype not in (tf.float32, tf.float64):
                z = tf.cast(z, tf.float32)
            try:
                rec["stats/avg_std"] = float(self._tf_avg_std(z))
                rec["stats/avg_offdiag_corr_sq"] = float(self._tf_avg_offdiag_corr_sq(z))
            except Exception:
                # Keep the row but mark the stats as missing.
                rec["stats/avg_std"] = float("nan")
                rec["stats/avg_offdiag_corr_sq"] = float("nan")

        with open(self.history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
