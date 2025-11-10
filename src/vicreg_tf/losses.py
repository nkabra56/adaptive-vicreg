"""
Loss functions for VICReg and Adaptive VICReg (TensorFlow / Keras 3).

This module provides two Keras layers that compute the VICReg-style loss terms:
  1) VICRegLoss: baseline with fixed variance floor gamma and off-diagonal covariance penalty
  2) AdaptiveVICRegLoss: adaptive variance target (EMA + median) and scale-invariant covariance

Both layers return (total_loss, logs) where logs is a dict of sub-term scalars
that the training loop can record.

Author: Nishant Kabra
Date: 11/8/2025
"""
from typing import Dict, Tuple, Union, Sequence
import tensorflow as tf

TensorLike = tf.Tensor
Pair = Union[Sequence[TensorLike], Tuple[TensorLike, TensorLike]]


def _center(z: TensorLike) -> TensorLike:
    """Zero-center features along the batch dimension."""
    return z - tf.reduce_mean(z, axis=0, keepdims=True)


def _covariance(z: TensorLike, eps: float = 1e-12) -> TensorLike:
    """Sample covariance matrix [D, D] for batch features [B, D]."""
    zc = _center(z)
    b = tf.shape(zc)[0]
    b_f = tf.cast(tf.maximum(b - 1, 1), tf.float32)  # safe denominator
    return tf.matmul(zc, zc, transpose_a=True) / b_f + eps * tf.eye(tf.shape(zc)[1])


def _offdiag_penalty(C: TensorLike) -> TensorLike:
    """Sum of squared off-diagonal entries of covariance."""
    d = tf.shape(C)[0]
    diag = tf.linalg.diag(tf.linalg.diag_part(C))
    off = C - diag
    d_f = tf.cast(d, tf.float32)  # normalize by D for scale stability
    return tf.reduce_sum(tf.square(off)) / d_f


def _trace_normalized_cov_penalty(C: TensorLike) -> TensorLike:
    """Scale-invariant covariance penalty: || C/tr(C) - (1/D)I ||_F^2."""
    d = tf.shape(C)[0]
    d_f = tf.cast(d, tf.float32)
    trace = tf.linalg.trace(C) + 1e-12
    Cn = C / trace
    I_scaled = tf.eye(d) / d_f
    return tf.reduce_sum(tf.square(Cn - I_scaled))


def _batch_std(z: TensorLike) -> TensorLike:
    """Per-dimension standard deviation over the batch."""
    return tf.math.reduce_std(z, axis=0)


def _extract_feature_dim(input_shape) -> int:
    """
    Robustly extract feature dim D from Keras input_shape variants:
      - [(None, D), (None, D)]
      - [(B, D), (B, D)]
      - (B, D)
      - TensorShape(...)
      - when the last entry itself is a 1-tuple like (D,)
    """
    # Normalize to pick the first shape
    s0 = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape

    # Convert to list form
    if isinstance(s0, tf.TensorShape):
        parts = s0.as_list()
    elif isinstance(s0, (list, tuple)):
        parts = list(s0)
    else:
        parts = tf.TensorShape(s0).as_list()

    if not parts:
        raise ValueError("AdaptiveVICRegLoss: could not parse input shape.")

    d_dim = parts[-1]  # expect D or (D,)
    # Resolve tf.compat.v1.Dimension
    if hasattr(d_dim, "value"):
        d_dim = d_dim.value
    # If last dim is itself a tuple/list like (D,) grab its last scalar
    if isinstance(d_dim, (list, tuple)):
        if not d_dim:
            raise ValueError("AdaptiveVICRegLoss: empty last-dimension tuple.")
        d_dim = d_dim[-1]
        if hasattr(d_dim, "value"):
            d_dim = d_dim.value

    if d_dim is None:
        raise ValueError("AdaptiveVICRegLoss: feature dimension must be known at build time.")
    if not isinstance(d_dim, (int, float)):
        raise TypeError(f"AdaptiveVICRegLoss: unexpected feature dim type: {type(d_dim)} ({d_dim})")
    return int(d_dim)


class VICRegLoss(tf.keras.layers.Layer):
    """Baseline VICReg loss: total = λ * align + μ * var + ν * cov."""
    def __init__(
        self,
        lambda0: float = 25.0,
        mu0: float = 25.0,
        nu0: float = 1.0,
        gamma: float = 1.0,
        name: str = "vicreg_loss",
        dtype: tf.dtypes.DType | str | None = None,
        trainable: bool = False,
    ):
        super().__init__(name=name, dtype=dtype, trainable=trainable)
        self.lambda0 = float(lambda0)
        self.mu0 = float(mu0)
        self.nu0 = float(nu0)
        self.gamma = float(gamma)

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg.update(dict(lambda0=self.lambda0, mu0=self.mu0, nu0=self.nu0, gamma=self.gamma))
        return cfg

    def call(
        self,
        *inputs: Pair,
        lambda_scale: TensorLike | float = 1.0,
        nu_scale: TensorLike | float = 1.0,
        training: bool | None = None,
    ):
        # Accept either (z1, z2) or ([z1, z2])
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            z1, z2 = inputs[0]
        elif len(inputs) == 2:
            z1, z2 = inputs
        else:
            raise ValueError("VICRegLoss expects (z1, z2) or [z1, z2] as inputs.")

        z1 = tf.cast(z1, self.compute_dtype)
        z2 = tf.cast(z2, self.compute_dtype)

        # Alignment
        align = tf.reduce_mean(tf.reduce_sum(tf.square(z1 - z2), axis=1))

        # Variance hinge
        s1 = _batch_std(z1)
        s2 = _batch_std(z2)
        v1 = tf.reduce_mean(tf.nn.relu(self.gamma - s1))
        v2 = tf.reduce_mean(tf.nn.relu(self.gamma - s2))
        var = 0.5 * (v1 + v2)

        # Covariance redundancy
        C1 = _covariance(z1)
        C2 = _covariance(z2)
        cov = 0.5 * (_offdiag_penalty(C1) + _offdiag_penalty(C2))

        # Allow schedules as tensors
        ls = tf.cast(lambda_scale, tf.float32)
        ns = tf.cast(nu_scale, tf.float32)
        total = (self.lambda0 * ls) * align + (self.mu0) * var + (self.nu0 * ns) * cov

        logs = {"align": align, "var": var, "cov": cov}
        return total, logs


class AdaptiveVICRegLoss(tf.keras.layers.Layer):
    """
    Adaptive VICReg:
      - AVT: γ_t from EMA of batch std vector via median, clipped to [gamma_min, gamma_max]
      - SICov: || C/tr(C) - (1/D)I ||_F^2
    total = λ * align + μ * var(γ_t) + ν * sicov
    """
    def __init__(
        self,
        lambda0: float = 25.0,
        mu0: float = 25.0,
        nu0: float = 1.0,
        ema_beta: float = 0.99,
        gamma_min: float = 0.05,
        gamma_max: float = 1.0,
        name: str = "adaptive_vicreg_loss",
        dtype: tf.dtypes.DType | str | None = None,
        trainable: bool = False,
    ):
        super().__init__(name=name, dtype=dtype, trainable=trainable)
        self.lambda0 = float(lambda0)
        self.mu0 = float(mu0)
        self.nu0 = float(nu0)
        self.ema_beta = float(ema_beta)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self._ema_std = None  # created in build()

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg.update(
            dict(
                lambda0=self.lambda0,
                mu0=self.mu0,
                nu0=self.nu0,
                ema_beta=self.ema_beta,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
            )
        )
        return cfg

    def build(self, input_shape):
        """
        Create EMA container once feature dim D is known.
        Handles: [(None, D), (None, D)] or [(B, D), (B, D)] or (B, D) and variants.
        """
        d = _extract_feature_dim(input_shape)
        self._ema_std = self.add_weight(
            name="ema_std",
            shape=(d,),
            dtype=tf.float32,
            initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)

    def call(
        self,
        *inputs: Pair,
        lambda_scale: TensorLike | float = 1.0,
        nu_scale: TensorLike | float = 1.0,
        training: bool | None = None,
    ):
        # Accept either (z1, z2) or ([z1, z2])
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            z1, z2 = inputs[0]
        elif len(inputs) == 2:
            z1, z2 = inputs
        else:
            raise ValueError("AdaptiveVICRegLoss expects (z1, z2) or [z1, z2].")

        z1 = tf.cast(z1, self.compute_dtype)
        z2 = tf.cast(z2, self.compute_dtype)

        # Alignment
        align = tf.reduce_mean(tf.reduce_sum(tf.square(z1 - z2), axis=1))

        # Adaptive variance target gamma_t
        s1 = _batch_std(z1)
        s2 = _batch_std(z2)
        s = 0.5 * (s1 + s2)

        # EMA update (graph-friendly)
        beta = tf.cast(self.ema_beta, tf.float32)
        new_ema = beta * self._ema_std + (1.0 - beta) * tf.cast(s, tf.float32)
        self._ema_std.assign(new_ema)

        # Median for robustness; clip to [gamma_min, gamma_max]
        gamma_t = tfp_median(self._ema_std)
        gamma_t = tf.clip_by_value(gamma_t, self.gamma_min, self.gamma_max)

        v1 = tf.reduce_mean(tf.nn.relu(gamma_t - s1))
        v2 = tf.reduce_mean(tf.nn.relu(gamma_t - s2))
        var = 0.5 * (v1 + v2)

        # Scale-invariant covariance
        C1 = _covariance(z1)
        C2 = _covariance(z2)
        sicov = 0.5 * (_trace_normalized_cov_penalty(C1) + _trace_normalized_cov_penalty(C2))

        # Allow schedules as tensors
        ls = tf.cast(lambda_scale, tf.float32)
        ns = tf.cast(nu_scale, tf.float32)
        total = (self.lambda0 * ls) * align + (self.mu0) * var + (self.nu0 * ns) * sicov

        logs = {"align": align, "var": var, "cov": sicov, "gamma_t": gamma_t}
        return total, logs


def tfp_median(x: TensorLike) -> TensorLike:
    """Median of a 1-D tensor via sorting (no TFP dependency)."""
    x = tf.cast(x, tf.float32)
    d = tf.shape(x)[0]
    xs = tf.sort(x)
    even = tf.equal(d % 2, 0)
    mid_hi = d // 2
    mid_lo = mid_hi - 1

    def _even():
        a = tf.gather(xs, mid_lo)
        b = tf.gather(xs, mid_hi)
        return 0.5 * (a + b)

    def _odd():
        return tf.gather(xs, mid_hi)

    return tf.where(even, _even(), _odd())
