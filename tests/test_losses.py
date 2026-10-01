import numpy as np
import tensorflow as tf

from vicreg_tf.losses import (
    VICRegWeights,
    covariance_loss,
    invariance_loss,
    variance_loss,
    vicreg_total,
)


def test_invariance_loss_zero_when_views_identical():
    z = tf.random.normal([64, 32], seed=0)
    assert float(invariance_loss(z, z)) == 0.0


def test_invariance_loss_matches_manual_mse():
    z1 = tf.random.normal([64, 32], seed=1)
    z2 = tf.random.normal([64, 32], seed=2)
    expected = float(tf.reduce_mean(tf.square(z1 - z2)))
    assert abs(float(invariance_loss(z1, z2)) - expected) < 1e-6


def test_variance_loss_zero_when_std_at_or_above_floor():
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(4096, 16)).astype("float32"))  # std ~= 1.0
    loss = variance_loss(z, gamma=0.5)  # floor well below actual std
    assert float(loss) == 0.0


def test_variance_loss_positive_on_collapse():
    z = tf.zeros([64, 16])  # every dim has std == 0 -> full collapse
    loss = variance_loss(z, gamma=1.0)
    assert abs(float(loss) - 1.0) < 1e-6  # relu(gamma - 0) == gamma


def test_variance_loss_accepts_tensor_gamma():
    z = tf.zeros([64, 16])
    loss = variance_loss(z, gamma=tf.constant(0.7, tf.float32))
    assert abs(float(loss) - 0.7) < 1e-6


def test_variance_loss_gamma_hi_disabled_by_default_matches_old_formula():
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(64, 16), scale=5.0).astype("float32"))  # std ~= 5, well past 1.0
    default = float(variance_loss(z, gamma=1.0))
    explicit_zero = float(variance_loss(z, gamma=1.0, gamma_hi=0.0))
    assert default == explicit_zero == 0.0  # old one-sided formula: no penalty for std > gamma


def test_variance_loss_gamma_hi_penalizes_std_above_the_ceiling():
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(4096, 16), scale=5.0).astype("float32"))  # std ~= 5.0
    loss = variance_loss(z, gamma=1.0, gamma_hi=2.0)
    assert float(loss) > 0.0  # std (~5) is past the ceiling (2.0), unlike the one-sided formula


def test_variance_loss_gamma_hi_leaves_the_healthy_middle_alone():
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(4096, 16), scale=1.5).astype("float32"))  # std ~= 1.5
    loss = variance_loss(z, gamma=1.0, gamma_hi=3.0)  # between the floor and the ceiling
    assert float(loss) == 0.0


def test_variance_loss_gamma_hi_gives_a_nonzero_gradient_past_the_ceiling():
    # The motivating case (EXPERIMENTS.md finding 24): the one-sided floor has zero gradient once
    # std >= gamma, so nothing in the loss resists std growing without bound from there. A soft
    # ceiling should restore a nonzero, finite gradient once std grows past it.
    z = tf.Variable(tf.random.normal([64, 16], mean=0.0, stddev=20.0, seed=0))  # std far past any ceiling
    with tf.GradientTape() as tape:
        loss = variance_loss(z, gamma=1.0, gamma_hi=2.0)
    grad = tape.gradient(loss, z)
    assert bool(tf.reduce_all(tf.math.is_finite(grad)))
    assert float(tf.reduce_sum(tf.abs(grad))) > 0.0


def test_variance_loss_default_eps_matches_reduce_std_exactly():
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(64, 16)).astype("float32"))
    default = float(variance_loss(z, gamma=1.0))
    explicit_zero = float(variance_loss(z, gamma=1.0, eps=0.0))
    assert default == explicit_zero


def test_variance_loss_default_gradient_is_not_finite_at_full_collapse():
    # This is finding 10's untested hypothesis: tf.math.reduce_std's gradient is undefined
    # (0/0) at std == 0, unlike reference VICReg's sqrt(var + eps).
    z = tf.Variable(tf.zeros([64, 16]))
    with tf.GradientTape() as tape:
        loss = variance_loss(z, gamma=1.0)
    grad = tape.gradient(loss, z)
    assert not bool(tf.reduce_all(tf.math.is_finite(grad)))


def test_variance_loss_eps_gives_a_finite_gradient_at_full_collapse():
    z = tf.Variable(tf.zeros([64, 16]))  # every dim std == 0, reduce_std's gradient there is undefined
    with tf.GradientTape() as tape:
        loss = variance_loss(z, gamma=1.0, eps=1e-4)
    grad = tape.gradient(loss, z)
    assert bool(tf.reduce_all(tf.math.is_finite(grad)))


def test_variance_loss_eps_barely_changes_the_value_away_from_collapse():
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(4096, 16)).astype("float32"))  # std ~= 1.0, far from zero
    without_eps = float(variance_loss(z, gamma=0.5))
    with_eps = float(variance_loss(z, gamma=0.5, eps=1e-4))
    assert abs(without_eps - with_eps) < 1e-3


def test_covariance_loss_near_zero_for_independent_features():
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(8192, 8)).astype("float32"))
    loss = covariance_loss(z, nu=0.0)
    assert float(loss) < 0.01  # independent columns -> near-zero off-diag corr^2


def test_covariance_loss_penalizes_correlated_features():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(8192, 1)).astype("float32")
    z = tf.constant(np.tile(base, (1, 8)))  # every dim identical -> corr == 1 off-diagonal
    loss = covariance_loss(z, nu=0.0)
    assert float(loss) > 0.9  # (1 - 0)^2 for every off-diagonal entry


def test_covariance_loss_accepts_tensor_nu():
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(8192, 8)).astype("float32"))
    loss_default = float(covariance_loss(z, nu=0.0))
    loss_shifted = float(covariance_loss(z, nu=tf.constant(0.0, tf.float32)))
    assert abs(loss_default - loss_shifted) < 1e-6


def test_vicreg_total_matches_manual_weighted_sum():
    z1 = tf.random.normal([32, 16], seed=3)
    z2 = tf.random.normal([32, 16], seed=4)
    w = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    total, parts = vicreg_total(z1, z2, w=w, gamma=1.0, nu=0.0)
    manual = 25.0 * float(parts["l_align"]) + 25.0 * float(parts["l_var"]) + 1.0 * float(parts["l_cov"])
    assert abs(float(total) - manual) < 1e-4


def test_vicreg_total_parts_are_unweighted():
    z1 = tf.random.normal([32, 16], seed=5)
    z2 = tf.random.normal([32, 16], seed=6)
    # Parts should be identical regardless of the weights passed in.
    _, parts_a = vicreg_total(z1, z2, w=VICRegWeights(sim=25.0, var=25.0, cov=1.0), gamma=1.0, nu=0.0)
    _, parts_b = vicreg_total(z1, z2, w=VICRegWeights(sim=1.0, var=1.0, cov=1.0), gamma=1.0, nu=0.0)
    for key in ("l_align", "l_var", "l_cov"):
        assert abs(float(parts_a[key]) - float(parts_b[key])) < 1e-6


def test_vicreg_total_accepts_dict_weights():
    z1 = tf.random.normal([32, 16], seed=7)
    z2 = tf.random.normal([32, 16], seed=8)
    total_dc, _ = vicreg_total(z1, z2, w=VICRegWeights(sim=2.0, var=3.0, cov=4.0), gamma=1.0, nu=0.0)
    total_dict, _ = vicreg_total(z1, z2, w={"sim": 2.0, "var": 3.0, "cov": 4.0}, gamma=1.0, nu=0.0)
    assert abs(float(total_dc) - float(total_dict)) < 1e-6


def test_vicreg_total_var_gamma_hi_raises_l_var_for_an_exploded_embedding():
    z1 = tf.random.normal([64, 16], mean=0.0, stddev=20.0, seed=9)
    z2 = tf.random.normal([64, 16], mean=0.0, stddev=20.0, seed=10)
    w = VICRegWeights(sim=25.0, var=25.0, cov=1.0)
    _, off = vicreg_total(z1, z2, w=w, gamma=1.0, nu=0.0, var_gamma_hi=0.0)
    _, on = vicreg_total(z1, z2, w=w, gamma=1.0, nu=0.0, var_gamma_hi=2.0)
    assert float(off["l_var"]) == 0.0  # old formula: no penalty for std far above gamma
    assert float(on["l_var"]) > 0.0
