"""VICReg loss terms (invariance, variance, covariance) and their weighted sum."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class VICRegWeights:
    """Loss weights: `sim` for invariance, `var` for variance, `cov` for covariance."""
    sim: float = 25.0
    var: float = 25.0
    cov: float = 1.0


def _offdiag(mat: tf.Tensor) -> tf.Tensor:
    """Off-diagonal entries of a square matrix as a 1D tensor."""
    d = tf.shape(mat)[0]
    mask = tf.logical_not(tf.eye(d, dtype=tf.bool))
    return tf.boolean_mask(mat, mask)


def invariance_loss(z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
    """Mean squared error between the embeddings of the two views."""
    z1 = tf.convert_to_tensor(z1)
    z2 = tf.convert_to_tensor(z2)
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z: tf.Tensor, gamma) -> tf.Tensor:
    """mean(relu(gamma - std)) over feature dimensions, so each dimension keeps a std of at least `gamma`.

    `gamma` may be a float or a tensor.
    """
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])

    std = tf.math.reduce_std(z, axis=0)
    # No float(gamma) here: it can be a symbolic tensor inside train_step.
    gamma_t = tf.cast(tf.convert_to_tensor(gamma), std.dtype)
    return tf.reduce_mean(tf.nn.relu(gamma_t - std))


def covariance_loss(z: tf.Tensor, nu) -> tf.Tensor:
    """mean((offdiag(corr) - nu)^2), where `corr` is the feature correlation matrix.

    A constant dimension (std ~ 0) is left unscaled to avoid dividing by zero.
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

    off = _offdiag(corr)
    nu_t = tf.cast(tf.convert_to_tensor(nu), off.dtype)
    return tf.reduce_mean(tf.square(off - nu_t))


def vicreg_total(
    z1: tf.Tensor,
    z2: tf.Tensor,
    w: VICRegWeights | dict,
    gamma=1.0,
    nu=0.0,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Weighted VICReg loss and its unweighted components.

    Args:
        z1, z2: Embeddings of the two views, same shape.
        w: Weights for the (sim, var, cov) terms, as a `VICRegWeights` or a dict.
        gamma: Variance floor. Float or tensor.
        nu: Target for off-diagonal correlations. Float or tensor.

    Returns:
        The weighted total, and the unweighted terms keyed "l_align", "l_var", "l_cov".
    """
    if isinstance(w, dict):
        sim_w, var_w, cov_w = w["sim"], w["var"], w["cov"]
    else:
        sim_w, var_w, cov_w = w.sim, w.var, w.cov

    sim_w = tf.cast(tf.convert_to_tensor(sim_w), tf.float32)
    var_w = tf.cast(tf.convert_to_tensor(var_w), tf.float32)
    cov_w = tf.cast(tf.convert_to_tensor(cov_w), tf.float32)

    align = invariance_loss(z1, z2)
    var = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov = covariance_loss(z1, nu) + covariance_loss(z2, nu)

    total = sim_w * align + var_w * var + cov_w * cov
    parts = {"l_align": align, "l_var": var, "l_cov": cov}
    return total, parts
