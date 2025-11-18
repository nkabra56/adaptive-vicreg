"""
VICReg loss components and a convenience wrapper that returns the total
and the individual parts (alignment / variance / covariance).

What I ensure here (my words)
-----------------------------
- All ops are TF-friendly (no `float(tensor)` that breaks inside graph).
- `gamma` and `nu` may be *Python floats* **or** *TF tensors*.
  This is crucial because my **adaptive VICReg** produces time-varying
  `gamma/nu` tensors inside `train_step`. Baseline passes plain floats.

Baseline vs Adaptive (how it plugs in)
--------------------------------------
The *trainer* decides whether `gamma`/`nu` are:
  • constants (gamma=1.0, nu=0.0)  -> exact baseline VICReg
  • scheduled tensors from `AdaptiveTargets` -> adaptive VICReg
This file doesn't care *who* produced them; it just uses what it's given.

Author: Nishant Kabra
Date: 11/18/2025
"""
from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf


# -----------------------------------------------------------------------------
# Weights container for readability when wiring the three VICReg components
# -----------------------------------------------------------------------------
@dataclass
class VICRegWeights:
    """Simple container for the three VICReg loss weights."""
    sim: float = 25.0  # weight for alignment (invariance) term
    var: float = 25.0  # weight for variance floor term
    cov: float = 1.0   # weight for covariance (redundancy) term


# -----------------------------------------------------------------------------
# Helper: symmetric off-diagonal extraction for a [D x D] matrix
# -----------------------------------------------------------------------------
def _symm_offdiag(mat: tf.Tensor) -> tf.Tensor:
    """
    Return the off-diagonal entries of a square matrix as a 1-D tensor.

    I use a boolean mask against the identity to grab off-diagonal elements.
    """
    d = tf.shape(mat)[0]
    mask = tf.logical_not(tf.eye(d, dtype=tf.bool))
    return tf.boolean_mask(mat, mask)


# -----------------------------------------------------------------------------
# VICReg components (I keep them pure TF; no eager-only Python casts)
# -----------------------------------------------------------------------------
def invariance_loss(z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
    """
    Alignment term: mean squared error between the two views' embeddings.

    Parameters
    ----------
    z1, z2 : tf.Tensor
        Projection vectors for the two augmented views ([B, D] each).

    Returns
    -------
    tf.Tensor
        Scalar loss promoting view-invariance (MSE).
    """
    z1 = tf.convert_to_tensor(z1)
    z2 = tf.convert_to_tensor(z2)
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z: tf.Tensor, gamma) -> tf.Tensor:
    """
    Variance term: encourage per-dimension std to be at least `gamma`.

    Why I implement it like this
    ----------------------------
    - I *must not* do `float(gamma)` here because `gamma` can be a TF tensor
      produced by my adaptive scheduler. I convert/cast to the same dtype as
      `std` instead, which is safe in eager and in graph.

    Parameters
    ----------
    z : tf.Tensor
        A [B, D] projection batch (B = batch size, D = embedding dim).
        If rank>2 (e.g., someone passed features), I reshape to [B, -1].
    gamma : float or tf.Tensor
        Variance floor (often 1.0 for baseline VICReg). When using adaptive
        targets, this comes from `AdaptiveTargets.gamma(frac)` as a TF scalar.

    Returns
    -------
    tf.Tensor
        Scalar loss: mean over dims of relu(gamma - std).
    """
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])  # [B, D]

    # per-dimension std over the batch
    std = tf.math.reduce_std(z, axis=0)  # [D]

    # Cast gamma to std's dtype safely (supports tensor or float inputs).
    gamma_t = tf.cast(tf.convert_to_tensor(gamma), std.dtype)
    return tf.reduce_mean(tf.nn.relu(gamma_t - std))


def covariance_loss(z: tf.Tensor, nu) -> tf.Tensor:
    """
    Covariance (redundancy) term: penalize off-diagonal correlation away from `nu`.

    Implementation notes (important for stability)
    ----------------------------------------------
    - I standardize each dimension before computing correlations to avoid scale
      issues.
    - I guard division by near-zero std with `where(std < eps, 1, std)`.
    - I compute the correlation matrix ~ (zn^T zn)/N and then take off-diagonals.

    Parameters
    ----------
    z : tf.Tensor
        A [B, D] projection batch (or [B, ...] which I flatten).
    nu : float or tf.Tensor
        Target correlation for off-diagonals (0.0 in baseline VICReg). With
        adaptive targets this may be a TF scalar changing over time.

    Returns
    -------
    tf.Tensor
        Scalar loss = mean((offdiag(corr) - nu)^2).
    """
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])  # [B, D]

    n = tf.cast(tf.shape(z)[0], z.dtype)

    # Zero-mean per dimension
    zc = z - tf.reduce_mean(z, axis=0, keepdims=True)

    # Unit-variance guard (avoid exploding corr when a dim collapses)
    std = tf.math.reduce_std(zc, axis=0, keepdims=True)  # [1, D]
    std = tf.where(std < tf.cast(1e-12, z.dtype), tf.ones_like(std), std)
    zn = zc / std  # [B, D]

    # Correlation ≈ (zn^T zn)/n
    corr = tf.matmul(zn, zn, transpose_a=True) / tf.maximum(n, tf.cast(1.0, z.dtype))

    off = _symm_offdiag(corr)                         # [D*(D-1)]
    nu_t = tf.cast(tf.convert_to_tensor(nu), off.dtype)
    return tf.reduce_mean(tf.square(off - nu_t))


# -----------------------------------------------------------------------------
# Total VICReg loss wrapper
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
        Two projection batches from different views (same shape).
    w : VICRegWeights | dict
        Weights for (sim, var, cov). Accepts dataclass or dict.
    gamma : float | tf.Tensor
        Variance floor (baseline typically 1.0). May be a tensor if *adaptive*.
    nu : float | tf.Tensor
        Off-diagonal correlation target (baseline typically 0.0). May be tensor.

    Returns
    -------
    total : tf.Tensor
        The weighted sum of VICReg components.
    parts : dict[str, tf.Tensor]
        Unweighted component scalars with keys: l_align, l_var, l_cov.
    """
    # Resolve weights consistently (both dict and dataclass work).
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

    # Components (each returns a scalar TF tensor)
    align = invariance_loss(z1, z2)
    var   = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov   = covariance_loss(z1, nu)  + covariance_loss(z2, nu)

    total = sim_w * align + var_w * var + cov_w * cov
    parts = {"l_align": align, "l_var": var, "l_cov": cov}
    return total, parts
