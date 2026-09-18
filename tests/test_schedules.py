import numpy as np
import tensorflow as tf

from vicreg_tf.losses import VICRegWeights
from vicreg_tf.schedules import (
    AdaptiveReweighter,
    AdaptiveTargets,
    CosineWarmup,
    WeightSchedules,
    cosine_scaler,
)


# ---------------------------------------------------------------------------
# CosineWarmup / cosine_scaler
# ---------------------------------------------------------------------------
def test_cosine_warmup_endpoints_no_warmup():
    sched = CosineWarmup(warmup_frac=0.0, min_scale=0.0)
    assert abs(sched(0.0) - 1.0) < 1e-6
    assert abs(sched(0.5) - 0.5) < 1e-6
    assert abs(sched(1.0) - 0.0) < 1e-6


def test_cosine_warmup_respects_min_scale_floor():
    sched = CosineWarmup(warmup_frac=0.0, min_scale=0.2)
    assert abs(sched(1.0) - 0.2) < 1e-6


def test_cosine_warmup_linear_ramp_during_warmup():
    sched = CosineWarmup(warmup_frac=0.1, min_scale=0.0)
    assert abs(sched(0.05) - 0.5) < 1e-6  # halfway through warmup -> half scale


def test_cosine_warmup_tensor_path_matches_float_path():
    sched = CosineWarmup(warmup_frac=0.1, min_scale=0.1)
    for f in (0.0, 0.05, 0.3, 0.7, 1.0):
        float_val = sched(f)
        tensor_val = float(sched(tf.constant(f, tf.float32)))
        assert abs(float_val - tensor_val) < 1e-5


def test_cosine_scaler_from_step_and_total_steps():
    val = cosine_scaler(step=50, total_steps=100)
    assert abs(val - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# WeightSchedules: constant weights regardless of `use`/frac (current design)
# ---------------------------------------------------------------------------
def test_weight_schedules_weights_are_constant_w0():
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    sched = WeightSchedules(w0=w0, use=True, base_lr=0.01, base_wd=1e-6, total_steps=1000)
    for frac in (0.0, 0.3, 1.0):
        w = sched.weights(frac)
        assert w == {"sim": 25.0, "var": 25.0, "cov": 1.0}


def test_weight_schedules_lr_scales_with_use_flag():
    w0 = VICRegWeights()
    sched_off = WeightSchedules(w0=w0, use=False, base_lr=0.1, base_wd=0.0, total_steps=100)
    sched_on = WeightSchedules(w0=w0, use=True, base_lr=0.1, base_wd=0.0, total_steps=100)
    assert sched_off.lr_at(50) == 0.1  # unaffected by step when use=False
    assert abs(sched_on.lr_at(50) - 0.05) < 1e-4  # ~half lr at 50% progress


# ---------------------------------------------------------------------------
# AdaptiveTargets: the separate, optional gamma/nu target schedule
# ---------------------------------------------------------------------------
def test_adaptive_targets_disabled_returns_baseline_constants():
    targets = AdaptiveTargets(use=False)
    assert float(targets.gamma(0.5)) == 1.0
    assert float(targets.nu(0.5)) == 0.0


def test_adaptive_targets_enabled_gamma_constant_by_default():
    # Default start=end=1.0 -> gamma is a documented no-op even when enabled.
    targets = AdaptiveTargets(use=True)
    for frac in (0.0, 0.5, 1.0):
        assert abs(float(targets.gamma(frac)) - 1.0) < 1e-6


def test_adaptive_targets_enabled_nu_ramps_from_one_to_zero():
    targets = AdaptiveTargets(use=True)
    assert abs(float(targets.nu(0.0)) - 1.0) < 1e-3
    assert abs(float(targets.nu(1.0)) - 0.0) < 1e-3
    # Monotonically non-increasing across the schedule.
    vals = [float(targets.nu(f)) for f in np.linspace(0, 1, 11)]
    assert all(vals[i] >= vals[i + 1] - 1e-6 for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# AdaptiveReweighter: the new "Adaptive VICReg" weighting mechanism
# ---------------------------------------------------------------------------
def _run_steps(rw, n, **kwargs):
    weights = None
    for _ in range(n):
        weights = rw({
            "l_align": tf.constant(kwargs["l_align"], tf.float32),
            "l_var": tf.constant(kwargs["l_var"], tf.float32),
            "l_cov": tf.constant(kwargs["l_cov"], tf.float32),
        }, z_probe=kwargs["z_probe"])
    return weights


def _healthy_probe(std=1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=std, size=(256, 16)).astype("float32")
    return tf.constant(z)


def _collapsed_probe():
    return tf.zeros([256, 16])  # std == 0 everywhere -> collapse


def _redundant_probe():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(256, 1)).astype("float32")
    return tf.constant(np.tile(base, (1, 16)))  # every dim identical -> corr == 1


def test_reweighter_downweights_dominant_loss_term():
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0, decay=0.9)
    weights = _run_steps(
        rw, 200, l_align=1.0, l_var=10.0, l_cov=1.0, z_probe=_healthy_probe()
    )
    var_ratio = float(weights["var"]) / w0.var
    sim_ratio = float(weights["sim"]) / w0.sim
    cov_ratio = float(weights["cov"]) / w0.cov
    assert var_ratio < sim_ratio
    assert var_ratio < cov_ratio


def test_reweighter_converges_to_w0_when_balanced_and_healthy():
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0, decay=0.9)
    weights = _run_steps(
        rw, 500, l_align=1.0, l_var=1.0, l_cov=1.0, z_probe=_healthy_probe()
    )
    assert abs(float(weights["sim"]) - 25.0) < 1.0
    assert abs(float(weights["var"]) - 25.0) < 1.0
    assert abs(float(weights["cov"]) - 1.0) < 0.1


def test_reweighter_boosts_var_weight_on_collapse():
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0, decay=0.9)
    weights = _run_steps(
        rw, 200, l_align=1.0, l_var=1.0, l_cov=1.0, z_probe=_collapsed_probe()
    )
    assert float(weights["var"]) > w0.var


def test_reweighter_boosts_cov_weight_on_redundancy():
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0, decay=0.9)
    weights = _run_steps(
        rw, 200, l_align=1.0, l_var=1.0, l_cov=1.0, z_probe=_redundant_probe()
    )
    assert float(weights["cov"]) > w0.cov


def test_reweighter_respects_clip_bounds_under_extreme_inputs():
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0, decay=0.9, mag_clip=(0.2, 5.0), boost_clip=(1.0, 4.0))
    weights = _run_steps(
        rw, 500, l_align=0.001, l_var=1000.0, l_cov=0.001, z_probe=_collapsed_probe()
    )
    assert 0.2 * w0.var <= float(weights["var"]) <= 5.0 * w0.var * 4.0
    assert 0.2 * w0.sim <= float(weights["sim"]) <= 5.0 * w0.sim * 4.0


def test_reweighter_works_inside_tf_function():
    # This mirrors how train_step actually calls it (traced once by Keras .fit()).
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0, decay=0.9)

    @tf.function
    def one_step(l_align, l_var, l_cov, z_probe):
        return rw({"l_align": l_align, "l_var": l_var, "l_cov": l_cov}, z_probe=z_probe)

    z = _healthy_probe()
    for _ in range(5):
        weights = one_step(tf.constant(1.0), tf.constant(1.0), tf.constant(1.0), z)
    assert set(weights.keys()) == {"sim", "var", "cov"}
