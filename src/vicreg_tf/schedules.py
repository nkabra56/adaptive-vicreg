"""Cosine LR/WD schedules, the optional gamma/nu target schedule, and the adaptive loss-weight reweighter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import tensorflow as tf

from .losses import VICRegWeights

Scalar = Union[float, tf.Tensor]


class CosineWarmup:
    """Linear warmup over `warmup_frac`, then cosine decay to `min_scale` at frac=1.

    Takes a float or a tensor and returns the same kind, so it works in eager code and inside `tf.function`.
    """

    def __init__(self, warmup_frac: float = 0.0, min_scale: float = 0.0):
        self.warmup_frac = float(max(0.0, min(1.0, warmup_frac)))
        self.min_scale = float(max(0.0, min(1.0, min_scale)))

    def __call__(self, frac: Scalar) -> Scalar:
        if tf.is_tensor(frac):
            f = tf.clip_by_value(tf.cast(frac, tf.float32), 0.0, 1.0)
            wf = tf.cast(self.warmup_frac, tf.float32)

            def _cosine_part():
                t = (f - wf) / tf.maximum(1e-9, 1.0 - wf)
                return 0.5 * (1.0 + tf.cos(tf.constant(math.pi, tf.float32) * t))

            warm = tf.where(f < wf, f / tf.maximum(wf, 1e-9), _cosine_part())
            return tf.maximum(tf.cast(self.min_scale, tf.float32), warm)

        f = float(max(0.0, min(1.0, float(frac))))
        if f < self.warmup_frac and self.warmup_frac > 0.0:
            warm = f / self.warmup_frac
        else:
            t = (f - self.warmup_frac) / max(1e-9, 1.0 - self.warmup_frac)
            warm = 0.5 * (1.0 + math.cos(math.pi * t))
        return max(self.min_scale, warm)


def cosine_scaler(
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    t: Optional[Scalar] = None,
    warmup_frac: float = 0.0,
    min_scale: float = 0.0,
) -> Scalar:
    """Warmup-then-cosine scale in [min_scale, 1].

    Pass the progress fraction `t` directly, or `step` and `total_steps` to have it computed.
    """
    sched = CosineWarmup(warmup_frac=warmup_frac, min_scale=min_scale)
    if t is not None:
        return sched(t)
    if step is None or total_steps is None:
        raise ValueError("Provide either t or (step, total_steps).")
    if tf.is_tensor(step):
        frac = tf.cast(step, tf.float32) / tf.cast(total_steps, tf.float32)
    else:
        frac = float(step) / float(total_steps)
    return sched(frac)


@dataclass
class WeightSchedules:
    """LR/WD cosine schedules plus the constant loss weights `w0`.

    `weights()` always returns `w0`. Loss reweighting is done by `AdaptiveReweighter`.
    """
    w0: VICRegWeights
    use: bool
    base_lr: float
    base_wd: float
    total_steps: int
    warmup_frac: float = 0.0
    min_scale: float = 0.0

    def __post_init__(self):
        if not isinstance(self.w0, VICRegWeights):
            self.w0 = VICRegWeights(**self.w0)

    def _scaled(self, base: float, step: Union[int, tf.Tensor]) -> Scalar:
        if not self.use:
            return base
        scale = cosine_scaler(
            step=step, total_steps=self.total_steps, warmup_frac=self.warmup_frac, min_scale=self.min_scale
        )
        return tf.cast(base, tf.float32) * scale if tf.is_tensor(scale) else base * float(scale)

    def lr_at(self, step: Union[int, tf.Tensor]) -> Scalar:
        return self._scaled(self.base_lr, step)

    def wd_at(self, step: Union[int, tf.Tensor]) -> Scalar:
        return self._scaled(self.base_wd, step)

    def weights(self, frac: Scalar) -> dict:
        return {"sim": self.w0.sim, "var": self.w0.var, "cov": self.w0.cov}


@dataclass
class AdaptiveTargets:
    """Optional time-varying variance floor (gamma) and correlation target (nu).

    This is independent of `AdaptiveReweighter`: it changes what the loss aims for, not how the terms
    are weighted. With `use=False` it returns the plain VICReg constants gamma=1.0 and nu=0.0.

    By default gamma stays at 1.0 and nu decays from 1.0 to 0.0 along a cosine.
    """
    use: bool
    gamma_sched: CosineWarmup = field(default_factory=lambda: CosineWarmup(warmup_frac=0.0, min_scale=1.0))
    nu_sched: CosineWarmup = field(default_factory=lambda: CosineWarmup(warmup_frac=0.0, min_scale=0.0))
    start: float = 1.0
    end: float = 1.0

    def gamma(self, frac: Scalar) -> Scalar:
        """Interpolate from `start` to `end` following `gamma_sched`."""
        if not self.use:
            return tf.constant(1.0, tf.float32)
        base = self.gamma_sched(frac)
        if tf.is_tensor(base):
            start, end = tf.cast(self.start, tf.float32), tf.cast(self.end, tf.float32)
            return start + (end - start) * base
        return self.start + (self.end - self.start) * float(base)

    def nu(self, frac: Scalar) -> Scalar:
        if not self.use:
            return tf.constant(0.0, tf.float32)
        base = self.nu_sched(frac)
        if tf.is_tensor(base):
            # Numerically equal to `base`, but simplifying it moves a reported float32 loss by 1 ulp.
            # Kept as written so seeded runs stay bit-identical to earlier ones.
            return tf.cast(0.0, tf.float32) + tf.cast(1.0, tf.float32) * base
        return float(base)


class AdaptiveReweighter:
    """Per-step sim/var/cov loss weights for Adaptive VICReg.

    Two signals are combined:

    * Loss-trend balancing. Each raw loss term gets a fast EMA (`decay`) and a slow EMA
      (`trend_decay`, meant to move several times slower than `decay`). The multiplier is
      fast / slow, clipped to `mag_clip` (var uses `var_mag_clip` if given). This asks "is this
      term getting worse lately, relative to its own recent history," not "is this term bigger
      than the other two right now."
      The old design (mean of all three EMAs, divided by this term's EMA) compared terms against
      each other: whichever was biggest got punished, even if it was only biggest because its
      competitors had converged toward zero. That caused two real failures: a collapsed embedding
      makes `l_var` huge and gets its weight cut exactly when it needs more (finding 11), and once
      that path was patched, a healthy-but-still-converging `l_align` became the new biggest term
      and got the same treatment (finding 15). Comparing a term only to its own slower-moving
      average sidesteps the zero-sum dynamic entirely: a term that is shrinking (fast EMA below
      slow EMA, the healthy-convergence case) never gets a multiplier below `mag_lo`, which
      defaults to 1.0, so no term is ever pushed below its base weight by this signal. A term that
      is trending up (fast EMA above slow EMA, whether from collapse or from being starved of
      weight) gets boosted instead, for as long as the trend lasts. Once a term's loss plateaus,
      fast and slow converge and the multiplier relaxes back to 1.0, even if that term's raw loss
      is still the largest of the three. Size alone is no longer a reason to move the weight.
    * Embedding-health boost (var and cov only). EMAs of the probe embedding's mean per-dimension
      std and mean squared off-diagonal correlation raise the var weight when std falls below
      `std_target` and the cov weight when correlation rises above `corr_target`, by at most
      `boost_clip`. Unrelated to the trend signal above, kept as-is.

    EMA inputs are passed through `tf.stop_gradient`, so gradients flow through weight * loss but not
    through the weight computation.

    Call once per step inside the GradientTape with the raw loss parts and a probe embedding (for
    example `z1`). Returns {"sim": ..., "var": ..., "cov": ...}.
    """

    def __init__(
        self,
        w0: VICRegWeights,
        decay: float = 0.98,
        trend_decay: float = 0.995,
        mag_clip: tuple[float, float] = (1.0, 5.0),
        var_mag_clip: Optional[tuple[float, float]] = None,
        std_target: float = 1.0,
        corr_target: float = 0.0,
        k_std: float = 2.0,
        k_cov: float = 2.0,
        boost_clip: tuple[float, float] = (1.0, 4.0),
    ):
        if not isinstance(w0, VICRegWeights):
            w0 = VICRegWeights(**w0)
        self.w0 = w0
        self.decay = float(decay)
        self.trend_decay = float(trend_decay)
        self.mag_lo, self.mag_hi = mag_clip
        self.var_mag_lo, self.var_mag_hi = var_mag_clip if var_mag_clip is not None else (self.mag_lo, self.mag_hi)
        self.std_target = float(std_target)
        self.corr_target = float(corr_target)
        self.k_std = float(k_std)
        self.k_cov = float(k_cov)
        self.boost_lo, self.boost_hi = boost_clip

        self.ema_align = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_align")
        self.ema_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_var")
        self.ema_cov = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_cov")
        self.ema_align_slow = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_align_slow")
        self.ema_var_slow = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_var_slow")
        self.ema_cov_slow = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_cov_slow")
        self.ema_std = tf.Variable(float(std_target), trainable=False, dtype=tf.float32, name="ema_std")
        self.ema_corr = tf.Variable(float(corr_target), trainable=False, dtype=tf.float32, name="ema_corr")
        self.step_count = tf.Variable(0, trainable=False, dtype=tf.int64, name="reweighter_step")

    def _update_ema(self, var: tf.Variable, x: tf.Tensor, decay: float) -> tf.Tensor:
        """Update `var` with the given decay and return its bias-corrected value."""
        decay_t = tf.cast(decay, tf.float32)
        var.assign(decay_t * var + (1.0 - decay_t) * tf.cast(x, tf.float32))
        t = tf.cast(self.step_count + 1, tf.float32)
        return var / (1.0 - tf.pow(decay_t, t))

    def __call__(self, raw_losses: dict, z_probe: tf.Tensor) -> dict:
        eps = tf.constant(1e-8, tf.float32)

        raw_align = tf.stop_gradient(raw_losses["l_align"])
        raw_var = tf.stop_gradient(raw_losses["l_var"])
        raw_cov = tf.stop_gradient(raw_losses["l_cov"])

        ema_align = self._update_ema(self.ema_align, raw_align, self.decay)
        ema_var = self._update_ema(self.ema_var, raw_var, self.decay)
        ema_cov = self._update_ema(self.ema_cov, raw_cov, self.decay)

        ema_align_slow = self._update_ema(self.ema_align_slow, raw_align, self.trend_decay)
        ema_var_slow = self._update_ema(self.ema_var_slow, raw_var, self.trend_decay)
        ema_cov_slow = self._update_ema(self.ema_cov_slow, raw_cov, self.trend_decay)

        trend_align = ema_align / (ema_align_slow + eps)
        trend_var = ema_var / (ema_var_slow + eps)
        trend_cov = ema_cov / (ema_cov_slow + eps)

        mag_sim = tf.clip_by_value(trend_align, self.mag_lo, self.mag_hi)
        mag_var = tf.clip_by_value(trend_var, self.var_mag_lo, self.var_mag_hi)
        mag_cov = tf.clip_by_value(trend_cov, self.mag_lo, self.mag_hi)

        z = tf.convert_to_tensor(z_probe)
        if z.shape.rank is not None and z.shape.rank > 2:
            z = tf.reshape(z, [tf.shape(z)[0], -1])
        avg_std = tf.reduce_mean(tf.math.reduce_std(z, axis=0))

        zc = z - tf.reduce_mean(z, axis=0, keepdims=True)
        zstd = tf.math.reduce_std(zc, axis=0, keepdims=True)
        zstd = tf.where(zstd < tf.cast(1e-12, z.dtype), tf.ones_like(zstd), zstd)
        zn = zc / zstd
        n = tf.cast(tf.shape(z)[0], zn.dtype)
        corr = tf.matmul(zn, zn, transpose_a=True) / tf.maximum(n, tf.cast(1.0, zn.dtype))
        d = tf.shape(corr)[0]
        off = tf.boolean_mask(corr, ~tf.eye(d, dtype=tf.bool))
        avg_offdiag = tf.reduce_mean(tf.square(off))

        ema_std = self._update_ema(self.ema_std, tf.stop_gradient(avg_std), self.decay)
        ema_corr = self._update_ema(self.ema_corr, tf.stop_gradient(avg_offdiag), self.decay)

        std_deficit = tf.nn.relu(self.std_target - ema_std) / max(self.std_target, 1e-8)
        corr_excess = tf.nn.relu(ema_corr - self.corr_target)

        boost_var = tf.clip_by_value(1.0 + self.k_std * std_deficit, self.boost_lo, self.boost_hi)
        boost_cov = tf.clip_by_value(1.0 + self.k_cov * corr_excess, self.boost_lo, self.boost_hi)

        self.step_count.assign_add(1)

        return {
            "sim": tf.cast(self.w0.sim, tf.float32) * mag_sim,
            "var": tf.cast(self.w0.var, tf.float32) * mag_var * boost_var,
            "cov": tf.cast(self.w0.cov, tf.float32) * mag_cov * boost_cov,
        }
