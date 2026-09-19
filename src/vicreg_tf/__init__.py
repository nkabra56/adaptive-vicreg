"""Public API of the package, so scripts can `from vicreg_tf import ...`."""

from .augment import color_jitter, random_augment, two_view_map
from .callbacks import CosineScheduleCallback, LossExplosionGuard, VicRegMetricsLogger, WarmupLR
from .data import build_cifar10, build_cifar100, build_dataset, steps_for_dataset, take_probe_batch
from .losses import VICRegWeights, covariance_loss, invariance_loss, variance_loss, vicreg_total
from .model import VICRegTrainer, build_encoder, build_projector
from .schedules import AdaptiveReweighter, AdaptiveTargets, WeightSchedules, cosine_scaler
from .utils import (
    add_device_arg,
    enable_memory_growth,
    force_build_for_saving,
    gpu_probe_ok,
    preparse_device,
    print_devices,
    safe_load_trainer_weights,
    select_device,
    set_global_seed,
    set_mixed_precision,
)

__all__ = [
    "AdaptiveReweighter",
    "AdaptiveTargets",
    "CosineScheduleCallback",
    "LossExplosionGuard",
    "VICRegTrainer",
    "VICRegWeights",
    "VicRegMetricsLogger",
    "WarmupLR",
    "WeightSchedules",
    "add_device_arg",
    "build_cifar10",
    "build_cifar100",
    "build_dataset",
    "build_encoder",
    "build_projector",
    "color_jitter",
    "cosine_scaler",
    "covariance_loss",
    "enable_memory_growth",
    "force_build_for_saving",
    "gpu_probe_ok",
    "invariance_loss",
    "preparse_device",
    "print_devices",
    "random_augment",
    "safe_load_trainer_weights",
    "select_device",
    "set_global_seed",
    "set_mixed_precision",
    "steps_for_dataset",
    "take_probe_batch",
    "two_view_map",
    "variance_loss",
    "vicreg_total",
]
