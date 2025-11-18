"""
Cosine schedules (LR/WD) and simple adaptive targets (gamma/nu).

Purpose
-------
I keep all learning-rate / weight-decay cosine utilities and my "adaptive
targets" (variance floor `gamma` and off-diagonal target `nu`) here. Everything
is written to be safe in both eager and graph execution: I never call
`float(tensor)` inside graph paths, and I return the same *type* (float vs
Tensor) that I’m given.

How I use this module
---------------------
- During training I attach a callback that calls `cosine_scaler(step=..., total_steps=...)`
  each batch to scale optimizer LR and WD.
- Inside the trainer I instantiate `AdaptiveTargets` when I want time-varying
  `gamma` and `nu`. With my defaults below, `gamma` ramps quickly from 0.9 -> 1.0
  during the first 10% of training, and `nu` stays near 0 (classical VICReg).
- `WeightSchedules` is a small convenience wrapper that prints base LR/WD and
  can return scheduled LR/WD (or just constants if schedules are off). I also
  expose `weights(frac)` where I gently ramp the covariance weight up over time
  (helps reduce redundancy more as features stabilize).

Author: Nishant Kabra
Date: 11/18/2025
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
    """
    Convert to a Python float, but only detach a Tensor in *eager* mode.

    Why I wrote it this way
    -----------------------
    Calling `float(tensor)` inside a graph function will crash, so when a TF
    tensor is passed in, I first convert it to a NumPy value and then to float
    only if we are definitely in eager. Everywhere else, I keep the math in TF.

    Args:
        x: A Python float/int or a TF tensor (only eager tensors are allowed).

    Returns:
        A plain Python float.
    """
    if _TF and tf.is_tensor(x):
        # Safe only in eager; never call in the graph path.
        return float(x.numpy())
    return float(x)


# -----------------------------------------------------------------------------
# Core cosine warmup (returns same type as its input: float -> float, tf -> tf)
# -----------------------------------------------------------------------------
class CosineWarmup:
    """
    Linear warmup followed by cosine decay on a unit interval.

    Contract
    --------
    - If you pass a Python float `frac` in [0, 1], I return a Python float.
    - If you pass a Tensor `frac` (float32), I return a Tensor float32.
    - I clamp the input to [0, 1] in both branches.

    Args:
        warmup_frac: Fraction of the [0,1] schedule to spend linearly warming up.
        min_scale: Minimum multiplier after cosine decay (>= 0).

    Example
    -------
    >>> s = CosineWarmup(warmup_frac=0.1, min_scale=0.0)
    >>> s(0.0), s(0.05), s(0.5), s(1.0)
    (0.0, 0.5, ~0.5, 0.0)
    """
    def __init__(self, warmup_frac: float = 0.0, min_scale: float = 0.0):
        # I sanitize inputs to valid ranges once here.
        self.warmup_frac = float(max(0.0, min(1.0, warmup_frac)))
        self.min_scale   = float(max(0.0, min(1.0, min_scale)))

    def __call__(self, frac: Union[float, "tf.Tensor"]) -> Union[float, "tf.Tensor"]:
        """
        Evaluate the schedule at a given progress fraction.

        Args:
            frac: Progress in [0, 1] as a float or Tensor.

        Returns:
            Scale in [min_scale, 1.0] with warmup then cosine decay.
        """
        if _TF and tf.is_tensor(frac):
            # --- Tensor path (graph-safe) ------------------------------------
            f = tf.clip_by_value(tf.cast(frac, tf.float32), 0.0, 1.0)
            wf = tf.cast(self.warmup_frac, tf.float32)

            # Linear warmup while f < warmup_frac, else cosine part
            def _cosine_part():
                t = (f - wf) / tf.maximum(1e-9, 1.0 - wf)
                # I use an explicit pi constant to avoid depending on tf.math.pi
                return 0.5 * (1.0 + tf.cos(tf.constant(3.141592653589793, tf.float32) * t))

            warm = tf.where(f < wf, f / tf.maximum(wf, 1e-9), _cosine_part())
            return tf.maximum(tf.cast(self.min_scale, tf.float32), warm)

        # --- Python path (floats) --------------------------------------------
        f = float(max(0.0, min(1.0, _to_float(frac))))
        if f < self.warmup_frac and self.warmup_frac > 0.0:
            # Linear ascent from 0 to 1 over warmup interval
            warm = f / self.warmup_frac
        else:
            # Cosine decay from 1 down to min_scale over the rest
            t = (f - self.warmup_frac) / max(1e-9, 1.0 - self.warmup_frac)
            import math
            warm = 0.5 * (1.0 + math.cos(math.pi * t))
        return max(self.min_scale, warm)


def cosine_scaler(step: Optional[int] = None,
                  total_steps: Optional[int] = None,
                  t: Optional[Union[float, "tf.Tensor"]] = None,
                  warmup_frac: float = 0.0,
                  min_scale: float = 0.0):
    """
    Cosine schedule driver used throughout my scripts.

    Contract
    --------
    - Provide either a direct fraction `t` in [0, 1], or the pair
      `(step, total_steps)`, from which I compute the fraction internally.
    - I return a float if the inputs are floats, or a Tensor if inputs are TF.

    Args:
        step: Current global step (int or Tensor). Optional if `t` provided.
        total_steps: Total number of steps (int or Tensor). Optional if `t` provided.
        t: Direct fraction in [0, 1]. If set, `step`/`total_steps` are ignored.
        warmup_frac: Warmup fraction for the internal schedule.
        min_scale: Lower bound of the schedule.

    Returns:
        A scalar scale factor in [min_scale, 1].

    Example
    -------
    >>> cosine_scaler(step=0, total_steps=100)
    1.0
    >>> cosine_scaler(step=100, total_steps=100)
    0.0
    >>> cosine_scaler(t=0.5, warmup_frac=0.1)
    ~0.5
    """
    sched = CosineWarmup(warmup_frac=warmup_frac, min_scale=min_scale)

    if t is not None:
        # Direct fraction path: convenient when I already track progress.
        return sched(t)

    if step is None or total_steps is None:
        # I require a complete pair to compute a fraction, otherwise it is ambiguous.
        raise ValueError("Provide either t or (step, total_steps).")

    if _TF and tf.is_tensor(step):
        # Graph-safe fraction computation
        frac = tf.cast(step, tf.float32) / tf.cast(total_steps, tf.float32)
    else:
        # Pure Python fallback
        frac = float(step) / float(total_steps)
    return sched(frac)


def cosine_schedule(*, step=None, total_steps=None, t=None,
                    warmup_frac: float = 0.0, min_scale: float = 0.0):
    """
    Backwards-compat shim: identical to `cosine_scaler`.

    I keep this alias because some of my older scripts import
    `cosine_schedule` by name from `vicreg_tf.schedules`.
    """
    return cosine_scaler(step=step, total_steps=total_steps, t=t,
                         warmup_frac=warmup_frac, min_scale=min_scale)


# -----------------------------------------------------------------------------
# Weight schedules + adaptive targets
# -----------------------------------------------------------------------------
@dataclass
class WeightSchedules:
    """
    Container for VICReg base weights and LR/WD schedule helpers.

    I primarily use this to print my base LR/WD once and to offer
    `lr_at(step)` / `wd_at(step)` helpers. The `weights(frac)` method returns
    the VICReg loss weights for a given progress fraction. I keep `sim` and
    `var` constant and *gently ramp up* the covariance weight from 0.5× to 1.5×
    its base value across training. This puts more emphasis on redundancy
    reduction once features have stabilized, which is a common tweak.

    Fields
    ------
    w0 : VICRegWeights
        Base weights for (sim, var, cov). I coerce from dict if needed.
    use : bool
        If False, `lr_at` and `wd_at` just return the base values; the weight
        ramp still returns a simple constant equal to w0.
    base_lr : float
        Learning rate before scheduling.
    base_wd : float
        Weight decay before scheduling.
    total_steps : int
        Total number of steps in this run (epochs * steps/epoch).
    warmup_frac : float
        Warmup fraction for the cosine scaler (defaults to 0).
    min_scale : float
        Min scale for the cosine scaler (defaults to 0).
    """
    w0: "VICRegWeights"
    use: bool
    base_lr: float
    base_wd: float
    total_steps: int
    warmup_frac: float = 0.0
    min_scale: float = 0.0

    def __post_init__(self):
        """
        Validate and normalize inputs after dataclass initialization.

        I import VICRegWeights lazily to avoid circular imports during package
        initialization, and I coerce dictionaries to the proper dataclass.
        """
        from .losses import VICRegWeights  # late import to avoid circular typing
        if not isinstance(self.w0, VICRegWeights):
            # Accept a dict-like and build the dataclass
            self.w0 = VICRegWeights(**self.w0)  # type: ignore[arg-type]

        # A simple banner so the console shows my schedule baseline.
        print(f"[schedules] total_steps={self.total_steps} base_lr={self.base_lr} base_wd={self.base_wd}")

    def lr_at(self, step: Union[int, "tf.Tensor"]) -> Union[float, "tf.Tensor"]:
        """
        Return scheduled learning rate at `step` or the base LR if schedules are disabled.
        """
        if not self.use:
            return self.base_lr
        scale = cosine_scaler(step=step, total_steps=self.total_steps,
                              warmup_frac=self.warmup_frac, min_scale=self.min_scale)
        # Keep return type consistent with `scale`
        return (tf.cast(self.base_lr, tf.float32) * scale) if _TF and tf.is_tensor(scale) else self.base_lr * float(scale)

    def wd_at(self, step: Union[int, "tf.Tensor"]) -> Union[float, "tf.Tensor"]:
        """
        Return scheduled weight decay at `step` or the base WD if schedules are disabled.
        """
        if not self.use:
            return self.base_wd
        scale = cosine_scaler(step=step, total_steps=self.total_steps,
                              warmup_frac=self.warmup_frac, min_scale=self.min_scale)
        return (tf.cast(self.base_wd, tf.float32) * scale) if _TF and tf.is_tensor(scale) else self.base_wd * float(scale)

    def weights(self, frac: Union[float, "tf.Tensor"]) -> dict:
        """
        Return the VICReg loss weights at progress `frac` in [0, 1].

        Current behavior
        ----------------
        - `sim` := w0.sim  (constant)
        - `var` := w0.var  (constant)
        - `cov` := w0.cov * (0.5 + 1.0*frac)  -> ramps 0.5× at t=0 to 1.5× at t=1

        Rationale: I want to down-weight the covariance penalty early (when
        features are noisy), and then emphasize redundancy reduction later.

        Args:
            frac: Progress fraction in [0, 1].

        Returns:
            A dict with keys 'sim', 'var', 'cov'.
        """
        if _TF and tf.is_tensor(frac):
            # Tensor path
            frac = tf.clip_by_value(tf.cast(frac, tf.float32), 0.0, 1.0)
            cov_scale = 0.5 + 1.0 * frac
            return {"sim": self.w0.sim, "var": self.w0.var, "cov": self.w0.cov * cov_scale}
        # Python path
        f = max(0.0, min(1.0, _to_float(frac)))
        cov_scale = 0.5 + 1.0 * f
        return {"sim": self.w0.sim, "var": self.w0.var, "cov": self.w0.cov * cov_scale}


@dataclass
class AdaptiveTargets:
    """
    Adaptive variance floor (`gamma`) and redundancy target (`nu`).

    What I return
    -------------
    - `gamma(frac)`: ramps from ~0.9 -> 1.0 over the first 10% of training,
      then stays near 1.0. This makes the variance floor gentle at the start.
    - `nu(frac)`: stays close to 0.0 (classical VICReg), but I keep it as a
      callable so I can make it time-varying later without touching call sites.

    Fields
    ------
    use : bool
        Whether to use adaptive schedules. If False, I return constants.
    gamma_sched : CosineWarmup
        Schedule that maps frac->[scale] before mapping to [start,end].
    nu_sched : CosineWarmup
        Schedule that maps frac->[scale]; I map it to 0.0 by default.
    start : float
        Start value for gamma.
    end : float
        End value for gamma.
    """
    use: bool
    # I warm up gamma quickly: 10% warmup, min_scale=1.0 (keeps scale >= 1.0),
    # then I map [0,1] -> [start,end] below.
    gamma_sched: CosineWarmup = field(default_factory=lambda: CosineWarmup(warmup_frac=0.10, min_scale=1.0))
    nu_sched: CosineWarmup    = field(default_factory=lambda: CosineWarmup(warmup_frac=0.0,  min_scale=0.0))
    start: float = 0.9
    end:   float = 1.0

    def gamma(self, frac: Union[float, "tf.Tensor"]):
        """
        Return the variance floor at progress `frac`.

        Behavior:
            - If `use` is False, I return constant 1.0.
            - Else, I follow `gamma_sched(frac)` and map it from [0,1] into
              [start, end] (default: 0.9 -> 1.0).
        """
        if not self.use:
            return tf.constant(1.0, tf.float32) if _TF else 1.0
        base = self.gamma_sched(frac)  # float or tf.Tensor
        if _TF and tf.is_tensor(base):
            # Map [0,1] -> [start,end] in TF
            return tf.cast(self.start, tf.float32) + (tf.cast(self.end, tf.float32) - tf.cast(self.start, tf.float32)) * base
        # Python path
        return self.start + (self.end - self.start) * float(base)

    def nu(self, frac: Union[float, "tf.Tensor"]):
        """
        Return the off-diagonal correlation target at progress `frac`.

        Behavior:
            - If `use` is False, I return constant 0.0.
            - Else, I evaluate `nu_sched(frac)` and clamp it to 0.0 for now.
              I keep the shape/type plumbing so I can change this later.
        """
        if not self.use:
            return tf.constant(0.0, tf.float32) if _TF else 0.0
        base = self.nu_sched(frac)
        if _TF and tf.is_tensor(base):
            # Currently returns 0.0 (placeholder for future non-zero schedules).
            return tf.cast(0.0, tf.float32) + tf.cast(1.0, tf.float32) * 0.0
        # Python path; same placeholder behavior
        return 0.0
