"""
Cosine schedules (LR/WD) and simple adaptive targets (gamma/nu) that I can
evaluate in eager/graph without tripping over float() on SymbolicTensors.

Design goals (my words)
----------------------
- Keep LR/WD schedules as *pure functions* that can be consumed by a Keras
  callback (see `CosineScheduleCallback` in train_vicreg.py).
- Provide an *adaptive* target producer for `gamma` (variance floor) and `nu`
  (corr target). This is where my **adaptive VICReg** parameters come from when
  the feature-flag is enabled in the trainer.

Author: Nishant Kabra
Date: 11/18/2025
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional

# I make TF an optional import: these utilities work with floats or tensors.
try:
    import tensorflow as tf
    _TF = True
except Exception:
    tf = None  # type: ignore
    _TF = False


def _to_float(x: Union[float, int, "tf.Tensor"]) -> float:
    """
    Detach to Python float if this is a Tensor and we're in eager mode.
    I never call this on graph SymbolicTensors.
    """
    if _TF and tf.is_tensor(x):
        return float(x.numpy())  # eager-only; OK for my usage
    return float(x)


# ============================================================================
# Core cosine warmup (returns same *type* as its input: float -> float; tf -> tf)
# ============================================================================
class CosineWarmup:
    """
    Cosine schedule with linear warmup over an initial fraction of [0, 1].

    I expect `frac` in [0, 1]. If you pass a Tensor, I stay in TF ops so this
    is safe inside `tf.function`.
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

        # Python path (floats)
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
    Scalar in [min_scale, 1] following linear-warmup + cosine decay.

    - If `t` is provided, it is the fraction in [0,1].
    - Else, I compute it from `step/total_steps`.
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


# Backwards-compat helper re-exported in __init__.py
def cosine_schedule(*, step=None, total_steps=None, t=None, warmup_frac: float = 0.0, min_scale: float = 0.0):
    return cosine_scaler(step=step, total_steps=total_steps, t=t, warmup_frac=warmup_frac, min_scale=min_scale)


# ============================================================================
# Weight schedules + Adaptive targets (used by the trainer)
# ============================================================================
@dataclass
class WeightSchedules:
    """
    Holds LR/WD schedules and returns VICReg weights scaled over training.

    NOTE (current behavior):
    ------------------------
    I keep loss weights constant by default. `weights()` returns `w0` so the
    baseline and adaptive variants differ *only* in gamma/nu targets—not
    in how the three VICReg components are weighted.
    """
    w0: "VICRegWeights"
    use: bool
    base_lr: float
    base_wd: float
    total_steps: int
    warmup_frac: float = 0.0
    min_scale: float = 0.0

    def __post_init__(self):
        # Avoid circular import at module import-time.
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
        """
        Return constant VICReg component weights (sim/var/cov).
        You can wire your own schedule here later if you want.
        """
        return {"sim": self.w0.sim, "var": self.w0.var, "cov": self.w0.cov}


@dataclass
class AdaptiveTargets:
    """
    Time-varying variance floor (gamma) and redundancy target (nu).

    How the trainer uses me
    -----------------------
    • If `use == True`, trainer calls `gamma(frac)` and `nu(frac)` using current
      progress fraction in [0,1]. These return TF scalars (inside `tf.function`)
      or floats (in eager) that can be broadcast against the loss tensors.

    • If `use == False`, the trainer *does not* call me; instead it uses the
      baseline constants gamma=1.0 and nu=0.0. This guarantees exact baseline.
    """
    use: bool
    gamma_sched: CosineWarmup = field(default_factory=lambda: CosineWarmup(warmup_frac=0.0, min_scale=1.0))
    nu_sched: CosineWarmup    = field(default_factory=lambda: CosineWarmup(warmup_frac=0.0, min_scale=0.0))
    start: float = 1.0  # starting gamma if you want to interpolate [start -> end]
    end:   float = 1.0  # end gamma (default 1.0 keeps gamma constant)

    def gamma(self, frac: Union[float, "tf.Tensor"]):
        if not self.use:
            return tf.constant(1.0, tf.float32) if _TF else 1.0
        base = self.gamma_sched(frac)  # float or tf.Tensor in [min_scale, 1]
        if _TF and tf.is_tensor(base):
            return tf.cast(self.start, tf.float32) + (tf.cast(self.end, tf.float32) - tf.cast(self.start, tf.float32)) * base
        return self.start + (self.end - self.start) * float(base)

    def nu(self, frac: Union[float, "tf.Tensor"]):
        if not self.use:
            return tf.constant(0.0, tf.float32) if _TF else 0.0
        base = self.nu_sched(frac)
        if _TF and tf.is_tensor(base):
            # For now I map [0,1] -> [0,1] linearly; adjust if you want a nonzero end.
            return tf.cast(0.0, tf.float32) + tf.cast(1.0, tf.float32) * base
        return float(base)
