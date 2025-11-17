"""
vicreg_tf package initializer.

What lives here
---------------
Re-exports of the most commonly used builders and utilities so that training
and evaluation scripts can import from `vicreg_tf` directly.

Example
-------
from vicreg_tf import build_encoder, build_projector, VICRegTrainer
"""

from .model import build_encoder, build_projector, VICRegTrainer
from .losses import (
    invariance_loss,
    variance_loss,
    covariance_loss,
    vicreg_total,
    VICRegWeights,
)
from .schedules import AdaptiveTargets, WeightSchedules, cosine_scaler, cosine_schedule
from .augment import color_jitter, random_augment, two_view_map
from .data import (
    build_dataset,
    build_cifar10,
    build_cifar100,
    steps_for_dataset,
)
from .utils import (
    set_mixed_precision,
    print_devices,
    enable_memory_growth,
    gpu_probe_ok,
    force_build_for_saving,
    safe_load_trainer_weights,
)

__all__ = [
    # models
    "build_encoder", "build_projector", "VICRegTrainer",
    # losses
    "invariance_loss", "variance_loss", "covariance_loss", "vicreg_total", "VICRegWeights",
    # schedules
    "AdaptiveTargets", "WeightSchedules", "cosine_schedule", "cosine_scaler",
    # data & augment
    "color_jitter", "random_augment", "two_view_map",
    "build_dataset", "build_cifar10", "build_cifar100", "steps_for_dataset",
    # utils
    "set_mixed_precision", "print_devices", "enable_memory_growth", "gpu_probe_ok",
    "force_build_for_saving", "safe_load_trainer_weights",
]
