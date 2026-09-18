"""Re-exports the package's public API so scripts can `from vicreg_tf import ...` directly."""

from .model import build_encoder, build_projector, VICRegTrainer
from .losses import (
    invariance_loss,
    variance_loss,
    covariance_loss,
    vicreg_total,
    VICRegWeights,
)
from .schedules import AdaptiveTargets, AdaptiveReweighter, WeightSchedules, cosine_scaler, cosine_schedule
from .augment import color_jitter, random_augment, two_view_map
from .data import (
    build_dataset,
    build_cifar10,
    build_cifar100,
    steps_for_dataset,
)
from .utils import (
    set_global_seed,
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
    "AdaptiveTargets", "AdaptiveReweighter", "WeightSchedules", "cosine_schedule", "cosine_scaler",
    # data & augment
    "color_jitter", "random_augment", "two_view_map",
    "build_dataset", "build_cifar10", "build_cifar100", "steps_for_dataset",
    # utils
    "set_global_seed", "set_mixed_precision", "print_devices", "enable_memory_growth", "gpu_probe_ok",
    "force_build_for_saving", "safe_load_trainer_weights",
]
