"""
Utility helpers used across the project.

Contents:
- set_seed():  basic reproducibility for Python + TensorFlow RNG.
- enable_mixed_precision(): one-liner to turn on mixed_float16 policy.
- EMA:  tiny exponential moving average over a 1D vector; used to track std.
- median_1d(): dependency-free median (via sort) for a 1D tensor.

Design goals:
- Zero external deps; keep tools tiny, explicit, and easily testable.

Author: Nishant Kabra
Date: 11/8/2025
"""

from __future__ import annotations
import random
import tensorflow as tf

def set_seed(s: int = 42) -> None:
    """
    Set Python and TensorFlow RNG seeds for reproducibility.

    Note:
    - This is "good enough" for course projects/ablations.
    - Perfect determinism across GPUs/CuDNN kernels requires extra flags.

    Args:
        s: seed value to apply to Python and TF.
    """
    random.seed(s)            # Python RNG (affects random.* calls)
    tf.random.set_seed(s)     # TensorFlow RNG (ops that respect TF seeds)

def enable_mixed_precision(enabled: bool) -> None:
    """
    Enable mixed precision (float16 compute, float32 variables) if requested.

    Pros:
    - Reduces memory footprint and often speeds up training on recent GPUs.

    Caveats:
    - Numerically sensitive ops still run in float32 automatically.
    - Ensure your GPU supports Tensor Cores (Volta+) for best results.

    Args:
        enabled: if True, activates policy; otherwise does nothing.
    """
    if enabled:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

class EMA(tf.Module):
    """
    Exponential Moving Average (EMA) for a 1D vector.

    Usage:
        ema = EMA((256,), beta=0.99)
        ema.update(new_values)      # one TF op, in-graph friendly
        current = ema.value         # tf.Variable with the current EMA

    Internals:
    - `_value` is a non-trainable tf.Variable living under this Module.
    - We cast inputs to float32 to be robust to mixed precision.
    """
    def __init__(self, shape, beta: float = 0.99, name: str = "ema"):
        super().__init__(name=name)
        self.beta = tf.constant(beta, dtype=tf.float32)                     # decay factor
        self._value = tf.Variable(tf.zeros(shape, dtype=tf.float32),        # initial vector
                                  trainable=False)

    @property
    def value(self) -> tf.Tensor:
        """Return the current EMA vector (tf.Variable)."""
        return self._value

    @tf.function
    def update(self, x: tf.Tensor) -> None:
        """
        Perform: value = beta * value + (1 - beta) * x

        Args:
            x: new sample (must broadcast to the EMA shape).
        """
        x = tf.cast(x, tf.float32)                                          # ensure fp32 math
        self._value.assign(self.beta * self._value + (1.0 - self.beta) * x) # in-place update

def median_1d(x: tf.Tensor) -> tf.Tensor:
    """
    Deterministic 1D median via sort (no TFP dependency).

    Args:
        x: any shape; we flatten to 1D and cast to float32.

    Returns:
        scalar float32 median.
    """
    x = tf.reshape(tf.cast(x, tf.float32), [-1])  # flatten to 1D
    x_sorted = tf.sort(x)                         # ascending sort
    n = tf.shape(x_sorted)[0]
    mid = n // 2                                  # middle index
    is_odd = tf.equal(n % 2, 1)
    # If odd:    return middle element
    # If even:   return average of two middle elements
    return tf.cond(
        is_odd,
        lambda: x_sorted[mid],
        lambda: 0.5 * (x_sorted[mid - 1] + x_sorted[mid])
    )
