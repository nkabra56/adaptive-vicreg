"""
VICReg loss components (invariance, variance, covariance) and a wrapper that
combines them into a weighted total.

`gamma` and `nu` accept either a Python float or a TF tensor, since the
adaptive-targets mechanism in `schedules.AdaptiveTargets` produces time-varying
tensors for them inside `train_step`. This module doesn't care which mechanism
produced them; it just uses whatever it's given.
"""
from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf


@dataclass
class VICRegWeights:
    """Weights for the three VICReg loss components."""
    sim: float = 25.0  # invariance (alignment) weight
    var: float = 25.0  # variance-floor weight
    cov: float = 1.0   # covariance (redundancy) weight


def _symm_offdiag(mat: tf.Tensor) -> tf.Tensor:
    """Return the off-diagonal entries of a square matrix as a 1D tensor."""
    d = tf.shape(mat)[0]
    mask = tf.logical_not(tf.eye(d, dtype=tf.bool))
    return tf.boolean_mask(mat, mask)


def invariance_loss(z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
    """Mean squared error between the two views' embeddings."""
    z1 = tf.convert_to_tensor(z1)
    z2 = tf.convert_to_tensor(z2)
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z: tf.Tensor, gamma) -> tf.Tensor:
    """
    Penalize per-dimension std below a floor `gamma`: mean(relu(gamma - std)).

    `gamma` may be a float or a TF tensor; avoid `float(gamma)` here since a
    tensor gamma would fail outside eager mode.
    """
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])

    std = tf.math.reduce_std(z, axis=0)
    gamma_t = tf.cast(tf.convert_to_tensor(gamma), std.dtype)
    return tf.reduce_mean(tf.nn.relu(gamma_t - std))


def covariance_loss(z: tf.Tensor, nu) -> tf.Tensor:
    """
    Penalize off-diagonal correlation away from a target `nu`:
    mean((offdiag(corr) - nu)^2).

    Standardizes each dimension before computing correlations, and guards
    near-zero std (a collapsed dimension) to avoid dividing by zero.
    """
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])

    n = tf.cast(tf.shape(z)[0], z.dtype)

    zc = z - tf.reduce_mean(z, axis=0, keepdims=True)
    std = tf.math.reduce_std(zc, axis=0, keepdims=True)
    std = tf.where(std < tf.cast(1e-12, z.dtype), tf.ones_like(std), std)
    zn = zc / std

    corr = tf.matmul(zn, zn, transpose_a=True) / tf.maximum(n, tf.cast(1.0, z.dtype))

    off = _symm_offdiag(corr)
    nu_t = tf.cast(tf.convert_to_tensor(nu), off.dtype)
    return tf.reduce_mean(tf.square(off - nu_t))


def vicreg_total(
    z1: tf.Tensor,
    z2: tf.Tensor,
    w: VICRegWeights | dict,
    gamma = 1.0,
    nu = 0.0,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Compute the weighted VICReg total loss and its unweighted components.

    Args:
        z1, z2: Projection batches from the two augmented views (same shape).
        w: Weights for (sim, var, cov), as a `VICRegWeights` or dict.
        gamma: Variance floor (baseline 1.0). May be a tensor under adaptive targets.
        nu: Off-diagonal correlation target (baseline 0.0). May be a tensor.

    Returns:
        total: The weighted sum of the three components.
        parts: The unweighted components, keyed "l_align", "l_var", "l_cov".
    """
    if isinstance(w, dict):
        sim_w = w["sim"]
        var_w = w["var"]
        cov_w = w["cov"]
    else:
        sim_w = w.sim
        var_w = w.var
        cov_w = w.cov

    sim_w = tf.cast(tf.convert_to_tensor(sim_w), tf.float32)
    var_w = tf.cast(tf.convert_to_tensor(var_w), tf.float32)
    cov_w = tf.cast(tf.convert_to_tensor(cov_w), tf.float32)

    align = invariance_loss(z1, z2)
    var   = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov   = covariance_loss(z1, nu)  + covariance_loss(z2, nu)

    total = sim_w * align + var_w * var + cov_w * cov
    parts = {"l_align": align, "l_var": var, "l_cov": cov}
    return total, parts
