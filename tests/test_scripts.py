import importlib
import os
import sys

import numpy as np
import pytest

REQUIRED_ARGS = {
    "train_vicreg": [],
    "resume_pretrain": ["--ckpt", "x.weights.h5"],
    "knn_eval": ["--encoder-ckpt", "x", "--out-csv", "o.csv"],
}


@pytest.mark.parametrize("script", REQUIRED_ARGS)
def test_device_flag_is_accepted(monkeypatch, script):
    module = importlib.import_module(script)
    monkeypatch.setattr(sys, "argv", [script, *REQUIRED_ARGS[script], "--device", "cpu"])
    assert module.parse_args().device == "cpu"


@pytest.mark.parametrize("script", REQUIRED_ARGS)
def test_device_defaults_to_auto(monkeypatch, script):
    module = importlib.import_module(script)
    monkeypatch.setattr(sys, "argv", [script, *REQUIRED_ARGS[script]])
    assert module.parse_args().device == "auto"


def test_train_flags_default_to_the_old_behavior(monkeypatch):
    module = importlib.import_module("train_vicreg")
    monkeypatch.setattr(sys, "argv", ["train_vicreg"])
    args = module.parse_args()
    assert (args.warmup_epochs, args.clipnorm, args.stop_epoch) == (0.0, 0.0, None)
    assert (args.w_sim, args.w_var, args.w_cov) == (25.0, 25.0, 1.0)


@pytest.mark.parametrize(
    "extra",
    [["--warmup-epochs", "2"], ["--stop-epoch", "0"], ["--epochs", "10", "--stop-epoch", "11"]],
)
def test_train_rejects_inconsistent_flags(monkeypatch, extra):
    module = importlib.import_module("train_vicreg")
    monkeypatch.setattr(sys, "argv", ["train_vicreg", *extra])
    with pytest.raises(SystemExit):
        module.parse_args()


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))


def test_linear_eval_resolves_files_directories_and_globs(tmp_path):
    eval_linear = importlib.import_module("eval_linear")
    old = tmp_path / "a" / "vicreg_encoder.weights.h5"
    new = tmp_path / "b" / "vicreg_encoder.weights.h5"
    full = tmp_path / "c" / "vicreg_full.weights.h5"
    _touch(old, 100)
    _touch(new, 200)
    _touch(full, 300)

    assert eval_linear._resolve_encoder_ckpt(str(old)) == (str(old), {})
    assert eval_linear._resolve_encoder_ckpt(str(tmp_path / "a")) == (str(old), {})
    assert eval_linear._resolve_encoder_ckpt(str(tmp_path / "*" / "vicreg_encoder.weights.h5")) == (str(new), {})
    path, kwargs = eval_linear._resolve_encoder_ckpt(str(tmp_path / "c"))
    assert path == str(full)
    assert kwargs == {"by_name": True, "skip_mismatch": True}


def test_knn_predict_separates_clean_clusters():
    knn_eval = importlib.import_module("knn_eval")
    rng = np.random.default_rng(0)
    centers = np.eye(3, 8, dtype="float32") * 5
    train_labels = np.repeat(np.arange(3), 30)
    train = centers[train_labels] + rng.normal(scale=0.05, size=(90, 8)).astype("float32")
    test_labels = np.repeat(np.arange(3), 5)
    test = centers[test_labels] + rng.normal(scale=0.05, size=(15, 8)).astype("float32")

    preds = knn_eval._knn_predict(train, train_labels, test, k=5, temperature=0.1, num_classes=3, chunk=4)
    assert knn_eval._top1_acc(test_labels, preds) == 1.0


def test_knn_resolve_accepts_a_run_directory_and_rejects_others(tmp_path):
    knn_eval = importlib.import_module("knn_eval")
    weights = tmp_path / "vicreg_encoder.weights.h5"
    weights.write_bytes(b"x")
    assert knn_eval._resolve_encoder_ckpt(str(tmp_path)) == str(weights)
    with pytest.raises(FileNotFoundError):
        knn_eval._resolve_encoder_ckpt(str(tmp_path / "nope"))
