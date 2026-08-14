"""
Cosine LR/WD schedules, the optional gamma/nu target schedule, and the
adaptive loss-weight reweighter.

TensorFlow is an optional import here so these utilities can be unit tested
and reasoned about independently of a TF installation where possible; the
tensor code paths require it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional

try:
    import tensorflow as tf
    _TF = True
except Exception:
    tf = None  # type: ignore
    _TF = False


def _to_float(x: Union[float, int, "tf.Tensor"]) -> float:
    """Convert to a Python float. Only safe to call in eager mode, never on a graph tensor."""
    if _TF and tf.is_tensor(x):
        return float(x.numpy())
    return float(x)


class CosineWarmup:
    """
    Linear warmup over `warmup_frac`, then cosine decay to `min_scale` by frac=1.

    Accepts a Python float or a TF tensor for `frac` and returns the same type,
    so it is safe to call from both eager code and inside `tf.function`.
    """
    def __init__(self, warmup_frac: float = 0.0, min_scale: float = 0.0):
        self.warmup_frac = float(max(0.0, min(1.0, warmup_frac)))
        self.min_scale   = float(max(0.0, min(1.0, min_scale)))

    def __call__(self, frac: Union[float, "tf.Tensor"]) -> Union[float, "tf.Tensor"]:
        if _TF and tf.is_tensor(frac):
            f = tf.clip_by_value(tf.cast(frac, tf.float32), 0.0, 1.0)
            wf = tf.cast(self.warmup_frac, tf.float32)

            def _cosine_part():
                t = (f - wf) / tf.maximum(1e-9, 1.0 - wf)
                return 0.5 * (1.0 + tf.cos(tf.constant(3.141592653589793, tf.float32) * t))

            warm = tf.where(f < wf, f / tf.maximum(wf, 1e-9), _cosine_part())
            return tf.maximum(tf.cast(self.min_scale, tf.float32), warm)

        f = float(max(0.0, min(1.0, _to_float(frac))))
        if f < self.warmup_frac and self.warmup_frac > 0.0:
            warm = f / self.warmup_frac
        else:
            t = (f - self.warmup_frac) / max(1e-9, 1.0 - self.warmup_frac)
            import math
            warm = 0.5 * (1.0 + math.cos(math.pi * t))
        return max(self.min_scale, warm)


def cosine_scaler(
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    t: Optional[Union[float, "tf.Tensor"]] = None,
    warmup_frac: float = 0.0,
    min_scale: float = 0.0,
):
    """
    Scalar in [min_scale, 1] following linear warmup then cosine decay.

    Pass `t` directly as a fraction in [0, 1], or `step`/`total_steps` to have
    the fraction computed for you.
    """
    sched = CosineWarmup(warmup_frac=warmup_frac, min_scale=min_scale)
    if t is not None:
        return sched(t)
    if step is None or total_steps is None:
        raise ValueError("Provide either t or (step, total_steps).")
    if _TF and tf.is_tensor(step):
        frac = tf.cast(step, tf.float32) / tf.cast(total_steps, tf.float32)
    else:
        frac = float(step) / float(total_steps)
    return sched(frac)


def cosine_schedule(*, step=None, total_steps=None, t=None, warmup_frac: float = 0.0, min_scale: float = 0.0):
    """Alias for `cosine_scaler`, kept for backward compatibility."""
    return cosine_scaler(step=step, total_steps=total_steps, t=t, warmup_frac=warmup_frac, min_scale=min_scale)


@dataclass
class WeightSchedules:
    """
    Holds LR/WD schedules and the (currently constant) VICReg component weights.

    `weights()` always returns `w0`; loss-term weighting is instead handled by
    `AdaptiveReweighter` when `adaptive_weights=True` on the trainer.
    """
    w0: "VICRegWeights"
    use: bool
    base_lr: float
    base_wd: float
    total_steps: int
    warmup_frac: float = 0.0
    min_scale: float = 0.0

    def __post_init__(self):
        from .losses import VICRegWeights
        if not isinstance(self.w0, VICRegWeights):
            self.w0 = VICRegWeights(**self.w0)  # type: ignore[arg-type]
        print(f"[schedules] total_steps={self.total_steps} base_lr={self.base_lr} base_wd={self.base_wd}")

    def lr_at(self, step: Union[int, "tf.Tensor"]) -> Union[float, "tf.Tensor"]:
        if not self.use:
            return self.base_lr
        scale = cosine_scaler(step=step, total_steps=self.total_steps, warmup_frac=self.warmup_frac, min_scale=self.min_scale)
        return (tf.cast(self.base_lr, tf.float32) * scale) if _TF and tf.is_tensor(scale) else self.base_lr * float(scale)

    def wd_at(self, step: Union[int, "tf.Tensor"]) -> Union[float, "tf.Tensor"]:
        if not self.use:
            return self.base_wd
        scale = cosine_scaler(step=step, total_steps=self.total_steps, warmup_frac=self.warmup_frac, min_scale=self.min_scale)
        return (tf.cast(self.base_wd, tf.float32) * scale) if _TF and tf.is_tensor(scale) else self.base_wd * float(scale)

    def weights(self, frac: Union[float, "tf.Tensor"]) -> dict:
        """Return the constant VICReg component weights (sim/var/cov)."""
        return {"sim": self.w0.sim, "var": self.w0.var, "cov": self.w0.cov}


@dataclass
class AdaptiveTargets:
    """
    Optional time-varying variance floor (gamma) and redundancy target (nu).

    This is independent of `AdaptiveReweighter`: it changes what the loss
    targets, not how the three components are weighted against each other.
    When `use=False`, `gamma()`/`nu()` return the baseline constants
    (1.0 and 0.0), matching plain VICReg exactly.
    """
    use: bool
    gamma_sched: CosineWarmup = field(default_factory=lambda: CosineWarmup(warmup_frac=0.0, min_scale=1.0))
    nu_sched: CosineWarmup    = field(default_factory=lambda: CosineWarmup(warmup_frac=0.0, min_scale=0.0))
    start: float = 1.0  # starting gamma, for interpolating [start, end]
    end:   float = 1.0  # ending gamma; default 1.0 keeps gamma constant

    def gamma(self, frac: Union[float, "tf.Tensor"]):
        if not self.use:
            return tf.constant(1.0, tf.float32) if _TF else 1.0
        base = self.gamma_sched(frac)
        if _TF and tf.is_tensor(base):
            return tf.cast(self.start, tf.float32) + (tf.cast(self.end, tf.float32) - tf.cast(self.start, tf.float32)) * base
        return self.start + (self.end - self.start) * float(base)

    def nu(self, frac: Union[float, "tf.Tensor"]):
        if not self.use:
            return tf.constant(0.0, tf.float32) if _TF else 0.0
        base = self.nu_sched(frac)
        if _TF and tf.is_tensor(base):
            return tf.cast(0.0, tf.float32) + tf.cast(1.0, tf.float32) * base
        return float(base)


class AdaptiveReweighter:
    """
    Per-step VICReg component weights (sim/var/cov). This is the mechanism the
    README calls "Adaptive VICReg". It combines two signals:

    Loss-magnitude balancing: tracks a bias-corrected EMA of each raw loss
    term (l_align/l_var/l_cov). A term whose EMA is large relative to the
    other two gets down-weighted, and vice versa, so no term dominates the
    gradient purely because it lives on a bigger numeric scale.

    Embedding-health reactive boost (var/cov only): tracks EMAs of the probe
    embedding's average per-dimension std and average squared off-diagonal
    correlation. If std drifts below `std_target` (collapse risk) or
    correlation drifts above `corr_target` (redundancy risk), the var/cov
    weight is boosted multiplicatively until the statistic recovers.

    Both EMAs are updated from `tf.stop_gradient`-ed values, so the weights
    behave as step-varying scalars rather than a path gradients flow through:
    the encoder is trained by the (weight * raw_loss) product, not by how the
    weight itself was computed.

    Call once per step, inside the GradientTape, with the raw (unweighted)
    loss parts and a probe embedding (e.g. z1), to get the weight dict
    {"sim": ..., "var": ..., "cov": ...}.
    """
    def __init__(
        self,
        w0: "VICRegWeights",
        decay: float = 0.98,
        mag_clip: tuple = (0.2, 5.0),
        std_target: float = 1.0,
        corr_target: float = 0.0,
        k_std: float = 2.0,
        k_cov: float = 2.0,
        boost_clip: tuple = (1.0, 4.0),
    ):
        if not _TF:
            raise RuntimeError("AdaptiveReweighter requires TensorFlow.")
        from .losses import VICRegWeights
        if not isinstance(w0, VICRegWeights):
            w0 = VICRegWeights(**w0)
        self.w0 = w0
        self.decay = float(decay)
        self.mag_lo, self.mag_hi = mag_clip
        self.std_target = float(std_target)
        self.corr_target = float(corr_target)
        self.k_std = float(k_std)
        self.k_cov = float(k_cov)
        self.boost_lo, self.boost_hi = boost_clip

        self.ema_align = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_align")
        self.ema_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_var")
        self.ema_cov = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ema_l_cov")
        self.ema_std = tf.Variable(float(std_target), trainable=False, dtype=tf.float32, name="ema_std")
        self.ema_corr = tf.Variable(float(corr_target), trainable=False, dtype=tf.float32, name="ema_corr")
        self.step_count = tf.Variable(0, trainable=False, dtype=tf.int64, name="reweighter_step")

    def _update_ema(self, var: tf.Variable, x: tf.Tensor) -> tf.Tensor:
        decay = tf.cast(self.decay, tf.float32)
        var.assign(decay * var + (1.0 - decay) * tf.cast(x, tf.float32))
        t = tf.cast(self.step_count + 1, tf.float32)
        return var / (1.0 - tf.pow(decay, t))

    def __call__(self, raw_losses: dict, z_probe: "tf.Tensor") -> dict:
        eps = tf.constant(1e-8, tf.float32)

        # Loss-magnitude balancing.
        ema_align = self._update_ema(self.ema_align, tf.stop_gradient(raw_losses["l_align"]))
        ema_var = self._update_ema(self.ema_var, tf.stop_gradient(raw_losses["l_var"]))
        ema_cov = self._update_ema(self.ema_cov, tf.stop_gradient(raw_losses["l_cov"]))

        mean_ema = (ema_align + ema_var + ema_cov) / 3.0
        mag_sim = tf.clip_by_value(mean_ema / (ema_align + eps), self.mag_lo, self.mag_hi)
        mag_var = tf.clip_by_value(mean_ema / (ema_var + eps), self.mag_lo, self.mag_hi)
        mag_cov = tf.clip_by_value(mean_ema / (ema_cov + eps), self.mag_lo, self.mag_hi)

        # Embedding-health reactive boost.
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

        ema_std = self._update_ema(self.ema_std, tf.stop_gradient(avg_std))
        ema_corr = self._update_ema(self.ema_corr, tf.stop_gradient(avg_offdiag))

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
