import numpy as np
import pytest
import tensorflow as tf

from vicreg_tf import data


@pytest.mark.parametrize("name", ["cifar10", "CIFAR-10", "cifar100", "cifar-100"])
def test_steps_for_dataset(name):
    assert data.steps_for_dataset(name, 256) == (50_000, 195)


def test_steps_for_dataset_is_at_least_one():
    assert data.steps_for_dataset("cifar10", 10**7)[1] == 1


def test_unsupported_dataset_names_raise():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        data.steps_for_dataset("mnist", 32)
    with pytest.raises(ValueError, match="Unsupported dataset"):
        data.build_dataset("mnist", 32, 32)


def test_two_view_pipeline_yields_paired_float_batches():
    images = np.random.default_rng(0).integers(0, 256, size=(16, 8, 8, 3), dtype=np.uint8)
    view1, view2 = next(iter(data._two_view_pipeline(images, image_size=8, batch_size=4)))
    assert view1.shape == view2.shape == (4, 8, 8, 3)
    assert view1.dtype == tf.float32
    assert 0.0 <= float(tf.reduce_min(view1)) and float(tf.reduce_max(view1)) <= 1.0
    assert not np.array_equal(view1.numpy(), view2.numpy())


def test_take_probe_batch_returns_the_first_view():
    ds = tf.data.Dataset.from_tensor_slices((np.zeros((8, 4)), np.ones((8, 4)))).batch(8)
    probe = data.take_probe_batch(ds, size=3)
    assert probe.shape == (3, 4)
    assert float(tf.reduce_sum(probe)) == 0.0


def test_take_probe_batch_returns_none_for_an_empty_dataset():
    assert data.take_probe_batch(tf.data.Dataset.from_tensor_slices(np.zeros((0, 4)))) is None
