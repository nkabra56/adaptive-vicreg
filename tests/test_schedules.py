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


def _run_steps(rw, n, **kwargs):
    weights = None
    for _ in range(n):
        weights = rw({
            "l_align": tf.constant(kwargs["l_align"], tf.float32),
            "l_var": tf.constant(kwargs["l_var"], tf.float32),
            "l_cov": tf.constant(kwargs["l_cov"], tf.float32),
        }, z_probe=kwargs["z_probe"])
    return weights


def _ramp(start, end, n):
    return list(np.linspace(start, end, n))


def _run_ramp(rw, l_align, l_var, l_cov, z_probe):
    weights = None
    n = len(l_var)
    for i in range(n):
        a = l_align[i] if isinstance(l_align, list) else l_align
        v = l_var[i] if isinstance(l_var, list) else l_var
        c = l_cov[i] if isinstance(l_cov, list) else l_cov
        weights = rw({
            "l_align": tf.constant(a, tf.float32),
            "l_var": tf.constant(v, tf.float32),
            "l_cov": tf.constant(c, tf.float32),
        }, z_probe=z_probe)
    return weights


def _old_style_multiplier(ema_align, ema_var, ema_cov, which, mag_lo=0.2, mag_hi=5.0, var_mag_lo=1.0):
    """The pre-fix (finding 11/15) magnitude balancing formula, kept here only so the new tests can
    show the exact before/after numbers. Not used by production code."""
    mean_ema = (ema_align + ema_var + ema_cov) / 3.0
    ema = {"sim": ema_align, "var": ema_var, "cov": ema_cov}[which]
    lo = var_mag_lo if which == "var" else mag_lo
    return max(lo, min(mag_hi, mean_ema / (ema + 1e-8)))


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


def test_reweighter_does_not_downweight_a_stable_dominant_term():
    # The old competitor-relative design ("whoever's EMA is biggest gets the smallest multiplier")
    # punished l_var here just for being 10x the others, even though it is perfectly stable, not
    # regressing. That is the mechanism behind findings 11 and 15: a term can be the largest raw
    # loss purely because its competitors converged, not because it needs more pressure. The new
    # trend-relative design only reacts to a term moving relative to its own recent history, so a
    # constant (however large) loss settles at its own baseline multiplier for every term alike.
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0, decay=0.9)
    weights = _run_steps(
        rw, 200, l_align=1.0, l_var=10.0, l_cov=1.0, z_probe=_healthy_probe()
    )
    var_ratio = float(weights["var"]) / w0.var
    sim_ratio = float(weights["sim"]) / w0.sim
    cov_ratio = float(weights["cov"]) / w0.cov
    assert abs(var_ratio - 1.0) < 0.05
    assert abs(sim_ratio - 1.0) < 0.05
    assert abs(cov_ratio - 1.0) < 0.05


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


def test_reweighter_var_multiplier_never_drops_below_w0_even_when_dominant():
    # A collapsed embedding makes l_var the largest raw term. Magnitude balancing must not use
    # that as a reason to lower w_var below w0.var: that is the fixed point finding 11 describes.
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0, decay=0.9)
    weights = _run_steps(
        rw, 200, l_align=0.03, l_var=1.9, l_cov=0.02, z_probe=_collapsed_probe()
    )
    assert float(weights["var"]) >= w0.var - 1e-4


def test_reweighter_var_mag_clip_caps_the_trend_boost():
    # var_mag_clip no longer carries a sub-1.0 floor to restore (the trend ratio can't drop below
    # its own baseline the way the old competitor ratio could), but it still bounds how far the
    # var multiplier can rise during a genuine upward trend. k_std=0 isolates the trend mechanism
    # from the separate embedding-health boost, so the cap here is exact. A tight hi of 1.5 should
    # visibly cap the boost an uncapped (default hi=5.0) run would get from the same ramp.
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    ramp = _ramp(0.1, 10.0, 300)

    probe = _healthy_probe()
    align = [0.03] * 300
    cov = [0.02] * 300

    rw_capped = AdaptiveReweighter(w0=w0, decay=0.9, trend_decay=0.995, var_mag_clip=(1.0, 1.5), k_std=0.0)
    weights_capped = _run_ramp(rw_capped, l_align=align, l_var=ramp, l_cov=cov, z_probe=probe)

    rw_default = AdaptiveReweighter(w0=w0, decay=0.9, trend_decay=0.995, k_std=0.0)
    weights_default = _run_ramp(rw_default, l_align=align, l_var=ramp, l_cov=cov, z_probe=probe)

    assert float(weights_capped["var"]) <= 1.5 * w0.var + 1e-4
    assert float(weights_default["var"]) > float(weights_capped["var"])


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


def test_reweighter_regime_finding11_boosts_var_during_collapse_not_suppresses():
    # Finding 11: embedding collapse makes l_var the largest raw term. The broken reweighter read
    # "largest raw term" as "needs less weight" and cut w_var exactly when the embedding most
    # needed more pressure to re-expand, a fixed point with no way back. Reproduce the regime with
    # a healthy warmup phase (gives l_var a low recent-history baseline) followed by a collapse
    # phase where l_var ramps toward its ceiling of about 2 while l_align/l_cov stay small and the
    # probe embedding collapses (matches the magnitudes finding 11 cites). The new mechanism should
    # boost w_var here because its trend is rising, not shrink it because its size is biggest.
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0)

    healthy_steps, collapse_steps = 300, 300
    l_align = [0.05] * healthy_steps + [0.03] * collapse_steps
    l_var = [0.3] * healthy_steps + list(np.linspace(0.3, 1.9, collapse_steps))
    l_cov = [0.05] * healthy_steps + [0.02] * collapse_steps
    healthy_probe = _healthy_probe()

    weights = None
    for i in range(healthy_steps + collapse_steps):
        z = healthy_probe if i < healthy_steps else _collapsed_probe()
        weights = rw({
            "l_align": tf.constant(l_align[i], tf.float32),
            "l_var": tf.constant(l_var[i], tf.float32),
            "l_cov": tf.constant(l_cov[i], tf.float32),
        }, z_probe=z)

    assert float(weights["var"]) > w0.var
    assert float(weights["sim"]) <= 1.05 * w0.sim
    assert float(weights["cov"]) <= 1.05 * w0.cov


def test_reweighter_regime_finding15_does_not_suppress_align_as_others_converge():
    # Finding 15: once the variance floor was patched, l_var and l_cov converged toward zero through
    # genuine training progress while l_align (never literally "solved," just small by comparison)
    # grew to become the largest raw term. The old mean(EMA)/EMA(term) formula read that as "align is
    # dominant, shrink its weight," pushing w_sim to about 0.34x w0.sim for most of training (see
    # EXPERIMENTS.md finding 15). l_align's rise here is real (it gets worse in absolute terms as the
    # weight that was supposed to push it keeps getting cut, not merely relatively bigger), so the
    # new design should boost it; l_var/l_cov's fall is real progress, so they should stay at
    # baseline rather than get boosted just for being small.
    w0 = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    rw = AdaptiveReweighter(w0=w0)

    n = 600
    l_align = list(np.linspace(0.0003, 1.5, n))
    l_var = list(np.linspace(1.0, 0.01, n))
    l_cov = list(np.linspace(1.0, 0.01, n))
    weights = _run_ramp(rw, l_align=l_align, l_var=l_var, l_cov=l_cov, z_probe=_healthy_probe())

    sim_ratio = float(weights["sim"]) / w0.sim
    var_ratio = float(weights["var"]) / w0.var
    cov_ratio = float(weights["cov"]) / w0.cov

    assert sim_ratio > 1.0
    assert var_ratio >= 1.0 - 1e-4
    assert cov_ratio >= 1.0 - 1e-4

    t = float(rw.step_count.numpy())
    ema_align_old = rw.ema_align.numpy() / (1.0 - rw.decay ** t)
    ema_var_old = rw.ema_var.numpy() / (1.0 - rw.decay ** t)
    ema_cov_old = rw.ema_cov.numpy() / (1.0 - rw.decay ** t)
    old_sim_mult = _old_style_multiplier(ema_align_old, ema_var_old, ema_cov_old, "sim")
    assert old_sim_mult < 1.0  # the failure this regime is reproducing: old design shrinks w_sim here
    assert sim_ratio > old_sim_mult
