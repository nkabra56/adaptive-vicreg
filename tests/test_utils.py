import argparse
import os
import random
import sys

import numpy as np
import pytest
import tensorflow as tf

from vicreg_tf import add_device_arg, preparse_device, select_device, set_global_seed


def test_select_device_cpu_needs_no_probe():
    assert select_device("cpu", "test") == "/CPU:0"


def test_preparse_device_hides_gpus_for_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "argv", ["prog", "--device", "cpu", "--unrelated", "x"])
    assert preparse_device() == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_preparse_device_defaults_to_auto_and_leaves_gpus_alone(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "argv", ["prog", "--epochs", "3"])
    assert preparse_device() == "auto"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_device_arg_rejects_unknown_choices():
    parser = argparse.ArgumentParser()
    add_device_arg(parser)
    assert parser.parse_args([]).device == "auto"
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "tpu"])


def test_set_global_seed_repeats_random_streams(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    def draw():
        set_global_seed(5)
        return random.random(), float(np.random.rand()), float(tf.random.uniform([]))

    assert draw() == draw()
