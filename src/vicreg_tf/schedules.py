"""
A tiny, dependency-free cosine schedule for scalar weights.

We use it to ramp up `lambda` (alignment) and `nu` (decorrelation).

Author: Nishant Kabra
Date: 11/8/2025
"""

from __future__ import annotations
import tensorflow as tf

def cosine_scaler(step: tf.Tensor, total_steps: tf.Tensor) -> tf.Tensor:
    """
    Smoothly increase from 0 -> 1 over `total_steps`.

    s(t) = 0.5 * (1 - cos(pi * t / T)),  t in [0, T]

    Args:
        step: current optimizer step (int or float tensor).
        total_steps: number of steps across all epochs.

    Returns:
        scalar in [0, 1]
    """
    step = tf.cast(step, tf.float32)
    total_steps = tf.cast(tf.maximum(total_steps, 1), tf.float32)  # avoid div-by-zero if unknown
    return 0.5 * (1.0 - tf.cos(tf.constant(3.1415926535) * step / total_steps))
