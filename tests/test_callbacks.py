import json
from types import SimpleNamespace

import pytest
import tensorflow as tf
from tensorflow import keras

from vicreg_tf import (
    CosineScheduleCallback,
    LossExplosionGuard,
    VicRegMetricsLogger,
    WarmupLR,
    build_encoder,
)


def _stub_model(lr=0.01):
    return SimpleNamespace(optimizer=keras.optimizers.AdamW(lr), curr_step=tf.Variable(0, dtype=tf.int64))


def test_cosine_callback_follows_the_schedule():
    opt = keras.optimizers.AdamW(learning_rate=0.1, weight_decay=1e-4)
    cb = CosineScheduleCallback(opt, total_steps=10, base_lr=0.1, base_wd=1e-4, verbose=0)
    lrs = []
    for step in range(11):
        cb.on_train_batch_begin(step)
        lrs.append(float(opt.learning_rate))
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[5] == pytest.approx(0.05)
    assert lrs[10] == pytest.approx(0.0, abs=1e-6)
    assert float(opt.weight_decay) == pytest.approx(0.0, abs=1e-9)


def test_cosine_callback_leaves_weight_decay_alone_without_base_wd():
    opt = keras.optimizers.AdamW(learning_rate=0.1, weight_decay=1e-4)
    cb = CosineScheduleCallback(opt, total_steps=10, base_lr=0.1, base_wd=None, verbose=0)
    for step in range(6):
        cb.on_train_batch_begin(step)
    assert float(opt.weight_decay) == pytest.approx(1e-4)


def test_cosine_callback_rejects_nonpositive_total_steps():
    with pytest.raises(ValueError):
        CosineScheduleCallback(keras.optimizers.AdamW(0.1), total_steps=0, base_lr=0.1, base_wd=None)


def test_warmup_ramps_from_ten_percent_to_full_and_then_stops():
    model = _stub_model()
    cb = WarmupLR(base_lr=0.01, warmup_steps=4)
    cb.set_model(model)
    lrs = []
    for _ in range(6):
        cb.on_train_batch_begin(0)
        lrs.append(float(model.optimizer.learning_rate))
        model.curr_step.assign_add(1)
    assert lrs[:4] == pytest.approx([0.01 * (0.1 + 0.9 * (i + 1) / 4) for i in range(4)])
    assert lrs[4] == lrs[5] == pytest.approx(0.01)


def test_loss_guard_cuts_the_learning_rate_and_respects_the_floor():
    model = _stub_model(0.01)
    cb = LossExplosionGuard(threshold=100.0, factor=0.1)
    cb.set_model(model)

    cb.on_train_batch_end(0, {"loss": 1.0})
    assert float(model.optimizer.learning_rate) == pytest.approx(0.01)

    cb.on_train_batch_end(0, {"loss": 500.0})
    assert float(model.optimizer.learning_rate) == pytest.approx(0.001)

    for _ in range(10):
        cb.on_train_batch_end(0, {"loss": 500.0})
    assert float(model.optimizer.learning_rate) == pytest.approx(1e-6)


def test_metrics_logger_writes_losses_and_embedding_stats(tmp_path):
    encoder = build_encoder(8, feat_dim=16)
    probe = tf.random.uniform([16, 8, 8, 3], seed=0)
    cb = VicRegMetricsLogger(str(tmp_path), encoder, None, probe, compute_on="encoder", record_every=2)

    cb.on_epoch_end(0, {"loss": 1.0})
    cb.on_epoch_end(1, {"loss": 1.0, "l_align": 0.1, "w_sim": 25.0})

    lines = (tmp_path / "metrics" / "history.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["epoch"] == 2
    assert rec["loss/total"] == 1.0
    assert rec["loss/align"] == pytest.approx(0.1)
    assert rec["loss/w_sim"] == 25.0
    assert "loss/var" not in rec
    assert rec["stats/avg_std"] > 0
    assert 0.0 <= rec["stats/avg_offdiag_corr_sq"] <= 1.0


def test_metrics_logger_without_a_probe_logs_only_losses(tmp_path):
    cb = VicRegMetricsLogger(str(tmp_path), build_encoder(8, feat_dim=16), None, None)
    cb.on_epoch_end(0, {"loss": 2.0})
    rec = json.loads((tmp_path / "metrics" / "history.jsonl").read_text())
    assert rec == {"epoch": 1, "loss/total": 2.0}
