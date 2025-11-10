"""
Public API exports for vicreg_tf.

This module collects commonly used symbols so downstream code can import
from src.vicreg_tf without reaching into submodules.

Author: Nishant Kabra
Date: 11/8/2025
"""

# Re-export augmentation helpers
from .augment import (
    make_two_views,
    augment_one_view,
    _standardize_channels as standardize,  # keep legacy name for callers
)

# Re-export core training pieces
from .losses import VICRegLoss, AdaptiveVICRegLoss
from .model import build_encoder, VICRegModel

__all__ = [
    "make_two_views",
    "augment_one_view",
    "standardize",
    "VICRegLoss",
    "AdaptiveVICRegLoss",
    "build_encoder",
    "VICRegModel",
]
