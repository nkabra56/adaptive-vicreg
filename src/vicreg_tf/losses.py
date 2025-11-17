"""
VICReg loss components and a convenience wrapper that returns the total
and the individual parts (alignment / variance / covariance).

Author: Nishant Kabra
Date: 11/08/2025
"""
from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Weights container
# -----------------------------------------------------------------------------
@dataclass
class VICRegWeights:
    """Simple container for the three VICReg loss weights."""
    sim: float = 25.0
    var: float = 25.0
    cov: float = 1.0


# -----------------------------------------------------------------------------
# Helper: symmetric off-diagonal extraction for a [D x D] matrix
# -----------------------------------------------------------------------------
def _symm_offdiag(mat: tf.Tensor) -> tf.Tensor:
    """Return the off-diagonal entries of a square matrix as a 1-D tensor."""
    d = tf.shape(mat)[0]
    mask = tf.logical_not(tf.eye(d, dtype=tf.bool))
    return tf.boolean_mask(mat, mask)


# -----------------------------------------------------------------------------
# Loss components
# -----------------------------------------------------------------------------
def invariance_loss(z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
    """
    Alignment term: mean squared error between the two views' embeddings.
    """
    z1 = tf.convert_to_tensor(z1)
    z2 = tf.convert_to_tensor(z2)
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z: tf.Tensor, gamma) -> tf.Tensor:
    """
    Variance term: encourage per-dimension std to be at least `gamma`.

    Note: `gamma` may be a Python float or a Tensor. We avoid `float(gamma)`
    so this works under both eager and graph execution.
    """
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])

    # per-dimension std over the batch
    std = tf.math.reduce_std(z, axis=0)

    gamma_t = tf.cast(tf.convert_to_tensor(gamma), std.dtype)
    return tf.reduce_mean(tf.nn.relu(gamma_t - std))


def covariance_loss(z: tf.Tensor, nu) -> tf.Tensor:
    """
    Covariance (redundancy) term: penalize off-diagonal correlation away from `nu`.

    We standardize each embedding dimension, compute its correlation matrix,
    take off-diagonals, and drive them toward `nu` (often 0.0).
    """
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])

    n = tf.cast(tf.shape(z)[0], z.dtype)
    # zero-mean per dim
    zc = z - tf.reduce_mean(z, axis=0, keepdims=True)
    # unit-variance guard
    std = tf.math.reduce_std(zc, axis=0, keepdims=True)
    std = tf.where(std < tf.cast(1e-12, z.dtype), tf.ones_like(std), std)
    zn = zc / std

    # correlation ≈ (zn^T zn)/n
    corr = tf.matmul(zn, zn, transpose_a=True) / tf.maximum(n, tf.cast(1.0, z.dtype))

    off = _symm_offdiag(corr)
    nu_t = tf.cast(tf.convert_to_tensor(nu), off.dtype)
    return tf.reduce_mean(tf.square(off - nu_t))


# -----------------------------------------------------------------------------
# Total VICReg loss
# -----------------------------------------------------------------------------
def vicreg_total(
    z1: tf.Tensor,
    z2: tf.Tensor,
    w: VICRegWeights | dict,
    gamma = 1.0,
    nu = 0.0,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Compute total VICReg loss and return (total, parts_dict).

    Parameters
    ----------
    z1, z2 : tf.Tensor
        Two embedding batches from different views.
    w : VICRegWeights | dict
        Weights for (sim, var, cov). Accepts dataclass or dict.
    gamma : float | tf.Tensor
        Variance floor. Can be scheduled and arrive as a Tensor.
    nu : float | tf.Tensor
        Off-diagonal correlation target. Often 0.0; may be Tensor if scheduled.
    """
    # Convert to a consistent view of weights (tensors are fine; floats also fine).
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

    # Parts
    align = invariance_loss(z1, z2)
    var   = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov   = covariance_loss(z1, nu)  + covariance_loss(z2, nu)

    total = sim_w * align + var_w * var + cov_w * cov
    parts = {"l_align": align, "l_var": var, "l_cov": cov}
    return total, parts
