import json

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from vicreg_tf import (
    VICRegTrainer,
    VICRegWeights,
    build_encoder,
    build_projector,
    force_build_for_saving,
    safe_load_trainer_weights,
)
from vicreg_tf.model import _mlp

METRICS = ("loss", "l_align", "l_var", "l_cov", "w_sim", "w_var", "w_cov")


def _trainer(**flags):
    enc = build_encoder(8, feat_dim=16)
    proj = build_projector(16, 32, 2)
    trainer = VICRegTrainer(
        encoder=enc,
        projector=proj,
        w0=VICRegWeights(),
        use_schedules=False,
        steps_per_epoch=2,
        epochs=1,
        base_lr=1e-3,
        base_wd=1e-6,
        **flags,
    )
    trainer.compile(optimizer=keras.optimizers.AdamW(1e-3))
    force_build_for_saving(trainer, enc, proj, 8)
    return trainer, enc


def _pairs():
    x = np.random.default_rng(0).uniform(size=(8, 8, 8, 3)).astype("float32")
    return tf.data.Dataset.from_tensor_slices((x, x[::-1].copy())).batch(4).repeat()


@pytest.mark.parametrize(
    "flags",
    [{}, {"adaptive_weights": True}, {"adaptive_targets": True}, {"adaptive_weights": True, "adaptive_targets": True}],
)
def test_fit_logs_finite_metrics_for_every_flag_combination(flags):
    trainer, _ = _trainer(**flags)
    history = trainer.fit(_pairs(), epochs=1, steps_per_epoch=2, verbose=0).history
    for key in METRICS:
        assert np.isfinite(history[key]).all(), key
    assert int(trainer.curr_step) == 2

    if flags.get("adaptive_weights"):
        assert history["w_sim"][0] != pytest.approx(25.0)
    else:
        assert history["w_sim"] == pytest.approx([25.0])
        assert history["w_var"] == pytest.approx([25.0])
        assert history["w_cov"] == pytest.approx([1.0])


def test_get_config_is_json_serializable_and_reports_settings():
    trainer, _ = _trainer(adaptive_weights=True)
    cfg = trainer.get_config()
    json.dumps(cfg)
    assert cfg["adaptive_weights"] is True
    assert cfg["adaptive_targets"] is False
    assert cfg["w0"] == {"sim": 25.0, "var": 25.0, "cov": 1.0}
    assert cfg["total_steps"] == 2
    assert cfg["base_lr"] == pytest.approx(1e-3)


def test_from_config_is_not_supported():
    with pytest.raises(NotImplementedError):
        VICRegTrainer.from_config({})


def test_checkpoint_roundtrip_restores_encoder_weights(tmp_path):
    trainer, enc = _trainer()
    path = str(tmp_path / "full.weights.h5")
    trainer.save_weights(path)

    fresh, fresh_enc = _trainer()
    assert not np.array_equal(enc.get_weights()[0], fresh_enc.get_weights()[0])  # random inits differ
    safe_load_trainer_weights(fresh, path)
    for expected, restored in zip(enc.get_weights(), fresh_enc.get_weights()):
        np.testing.assert_array_equal(expected, restored)


def test_safe_load_reports_a_missing_checkpoint(tmp_path):
    trainer, _ = _trainer()
    with pytest.raises(FileNotFoundError):
        safe_load_trainer_weights(trainer, str(tmp_path / "missing.weights.h5"))


@pytest.mark.parametrize("requested,expected", [(1, 1), (2, 2), (3, 3), (5, 3)])
def test_projector_depth(requested, expected):
    mlp = _mlp(16, 32, requested, "p")
    assert len(mlp.layers) == expected
    assert mlp.layers[-1].units == 32
