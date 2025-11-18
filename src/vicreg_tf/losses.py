"""
VICReg loss components and a convenience wrapper that returns the total
and the individual parts (alignment / variance / covariance).

What this module does
---------------------
I implement the three VICReg losses exactly as I use them in training:
- **Alignment (invariance)**: pulls paired views together.
- **Variance**: keeps per-dimension standard deviation above a floor `gamma`.
- **Covariance (redundancy)**: pushes off-diagonal correlation toward a target `nu`
  (usually zero), discouraging redundant features.

I also expose a small `VICRegWeights` dataclass and a `vicreg_total(...)` helper
that combines the three terms and returns both the total loss and a dict of parts
that I can log as Keras metrics.

How I use it
------------
- In my trainer's `train_step`, I call:
    `total, parts = vicreg_total(z1, z2, w, gamma=gamma, nu=nu)`
  and then add the parts to the metric logs.
- I pass `gamma` and `nu` as Python floats or Tensors (for schedules). I **avoid**
  forcing them to Python floats inside the loss, so everything works under eager
  and graph execution.

Example
-------
>>> w = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
>>> total, parts = vicreg_total(z1, z2, w, gamma=1.0, nu=0.0)
>>> float(parts["l_align"]), float(parts["l_var"]), float(parts["l_cov"])

Author: Nishant Kabra
Date: 11/17/2025
"""
from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf


# -----------------------------------------------------------------------------
# Weights container
# -----------------------------------------------------------------------------
@dataclass
class VICRegWeights:
    """
    Simple container for the three VICReg loss weights.

    Why I keep it like this
    -----------------------
    I prefer a tiny dataclass over a free-floating tuple/dict so I can pass the
    bundle around explicitly and still access each weight by name.

    Attributes
    ----------
    sim : float
        Weight for the alignment (invariance) term. Typical default is 25.0.
    var : float
        Weight for the variance floor term. Typical default is 25.0.
    cov : float
        Weight for the covariance (redundancy) term. Typical default is 1.0.
    """
    sim: float = 25.0
    var: float = 25.0
    cov: float = 1.0


# -----------------------------------------------------------------------------
# Helper: symmetric off-diagonal extraction for a [D x D] matrix
# -----------------------------------------------------------------------------
def _symm_offdiag(mat: tf.Tensor) -> tf.Tensor:
    """
    Return the off-diagonal entries of a square matrix as a 1-D tensor.

    Purpose
    -------
    In VICReg's covariance term I only want the off-diagonal correlations.
    This helper extracts them in a vectorized way so I can take a simple mean.

    Parameters
    ----------
    mat : tf.Tensor
        A square matrix of shape [D, D].

    Returns
    -------
    tf.Tensor
        A rank-1 tensor containing all off-diagonal elements, shape [D*(D-1)].

    Notes
    -----
    - I build a boolean mask where the diagonal is False and everything else is True,
      then I use `tf.boolean_mask` to gather the off-diagonal values.
    """
    # Get the square size D at runtime to support dynamic shapes.
    d = tf.shape(mat)[0]
    # Build a D x D identity in bool, then invert it to mask only off-diagonals.
    mask = tf.logical_not(tf.eye(d, dtype=tf.bool))
    # Gather all entries where mask is True (i.e., off-diagonals only).
    return tf.boolean_mask(mat, mask)


# -----------------------------------------------------------------------------
# Loss components
# -----------------------------------------------------------------------------
def invariance_loss(z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
    """
    Alignment (invariance) term: mean squared error between the two views.

    Parameters
    ----------
    z1, z2 : tf.Tensor
        Two batches of embeddings, typically shape [N, D] (but extra dims are fine).

    Returns
    -------
    tf.Tensor
        A scalar tensor with the MSE between paired embeddings.

    Notes
    -----
    - I rely on broadcasting-safe subtraction followed by a reduction to the mean.
    """
    # Ensure both inputs are tensors (helps with autograph and mixed types).
    z1 = tf.convert_to_tensor(z1)
    z2 = tf.convert_to_tensor(z2)
    # Standard MSE between paired embeddings encourages invariance to augmentation.
    return tf.reduce_mean(tf.square(z1 - z2))


def variance_loss(z: tf.Tensor, gamma) -> tf.Tensor:
    """
    Variance term: encourage per-dimension std to be at least `gamma`.

    Parameters
    ----------
    z : tf.Tensor
        A batch of embeddings. I accept [N, D] or higher rank; I flatten to [N, D].
    gamma : float | tf.Tensor
        The variance floor per dimension. This may be a Python float or a Tensor
        (for schedules). I **do not** coerce it to a Python float, so it works
        under graph mode too.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the average ReLU(gamma - std) across dims.
        This is zero when std >= gamma and positive when std falls short.

    Notes
    -----
    - I compute the batch std per dimension and apply a hinge at `gamma`.
    - I cast `gamma` to the same dtype as `std` to avoid type warnings.
    """
    # Convert input to tensor and flatten non-batch dims to a single feature axis.
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])

    # Compute per-dimension standard deviation over the batch.
    std = tf.math.reduce_std(z, axis=0)

    # Make sure gamma is a tensor of the same dtype as `std` (avoid float()).
    gamma_t = tf.cast(tf.convert_to_tensor(gamma), std.dtype)

    # Penalize only when std < gamma (ReLU makes the penalty non-negative).
    return tf.reduce_mean(tf.nn.relu(gamma_t - std))


def covariance_loss(z: tf.Tensor, nu) -> tf.Tensor:
    """
    Covariance (redundancy) term: penalize off-diagonal correlation away from `nu`.

    Parameters
    ----------
    z : tf.Tensor
        A batch of embeddings. I accept [N, D] or higher rank; I flatten to [N, D].
    nu : float | tf.Tensor
        Target value for off-diagonal correlation. I commonly use 0.0.
        This can also be a Tensor if I schedule it.

    Returns
    -------
    tf.Tensor
        A scalar tensor: mean of squared deviation of off-diagonal correlations
        from the target `nu`.

    Notes
    -----
    - I standardize each dimension to zero-mean and unit-variance (guarding the
      small-variance case), compute the correlation matrix, pull off-diagonals,
      and drive them toward `nu`.
    """
    # Convert input to tensor and flatten non-batch dims to a single feature axis.
    z = tf.convert_to_tensor(z)
    if z.shape.rank is not None and z.shape.rank > 2:
        z = tf.reshape(z, [tf.shape(z)[0], -1])

    # Batch size as tensor (keeps dtype consistent for division).
    n = tf.cast(tf.shape(z)[0], z.dtype)

    # Subtract mean per dimension to center features.
    zc = z - tf.reduce_mean(z, axis=0, keepdims=True)

    # Compute per-dimension std and guard against near-zero values to avoid division blowups.
    std = tf.math.reduce_std(zc, axis=0, keepdims=True)
    std = tf.where(std < tf.cast(1e-12, z.dtype), tf.ones_like(std), std)

    # Standardize each dimension to unit variance.
    zn = zc / std

    # Correlation matrix ≈ (zn^T zn) / n for standardized features.
    corr = tf.matmul(zn, zn, transpose_a=True) / tf.maximum(n, tf.cast(1.0, z.dtype))

    # Extract all off-diagonal entries.
    off = _symm_offdiag(corr)

    # Cast target to the same dtype as correlation entries and compute squared error.
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
    Compute total VICReg loss and return `(total, parts_dict)`.

    Parameters
    ----------
    z1, z2 : tf.Tensor
        Two embedding batches from different augmented views (shapes broadcastable).
    w : VICRegWeights | dict
        Weights for (sim, var, cov). I accept either a `VICRegWeights` instance
        or a dict with keys {'sim','var','cov'}.
    gamma : float | tf.Tensor, optional
        Variance floor (per dimension). Can be a Tensor when scheduled.
    nu : float | tf.Tensor, optional
        Off-diagonal correlation target. Often 0.0; may be a Tensor when scheduled.

    Returns
    -------
    (tf.Tensor, dict)
        - `total`: scalar tensor with the full VICReg loss.
        - `parts`: dict with keys:
            * 'l_align' : alignment term
            * 'l_var'   : variance term (sum over z1 and z2)
            * 'l_cov'   : covariance term (sum over z1 and z2)

    Notes
    -----
    - I cast the weights to tensors (float32) so that the multiplication and
      gradient flow remain consistent in TF graphs.
    - I deliberately avoid `float(gamma)` and `float(nu)` so both eager and
      autograph/graph mode are safe when these come from schedules.
    """
    # Convert weights container into three scalars (support dict or dataclass).
    if isinstance(w, dict):
        sim_w = w["sim"]
        var_w = w["var"]
        cov_w = w["cov"]
    else:
        sim_w = w.sim
        var_w = w.var
        cov_w = w.cov

    # Cast weights to tensors for safe math in TF graphs.
    sim_w = tf.cast(tf.convert_to_tensor(sim_w), tf.float32)
    var_w = tf.cast(tf.convert_to_tensor(var_w), tf.float32)
    cov_w = tf.cast(tf.convert_to_tensor(cov_w), tf.float32)

    # --- Individual parts ---
    # Alignment encourages invariance between the two augmented views.
    align = invariance_loss(z1, z2)
    # Variance keeps per-dimension std above the floor for both views.
    var   = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    # Covariance penalizes redundancy (off-diagonal correlation) for both views.
    cov   = covariance_loss(z1, nu)  + covariance_loss(z2, nu)

    # Weighted sum of all parts gives the total VICReg objective.
    total = sim_w * align + var_w * var + cov_w * cov

    # I also return the parts so the trainer can log them as metrics.
    parts = {"l_align": align, "l_var": var, "l_cov": cov}
    return total, parts
