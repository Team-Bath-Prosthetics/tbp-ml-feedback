"""Tests for the shared recording format in tactile.data_format."""

import json

import numpy as np
import pytest

from tactile import data_format
from tactile.data_format import ADC_MAX, load_recording, save_recording


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    return rng.integers(0, ADC_MAX + 1, size=(5, 20, 10, 5), dtype=np.uint16)


def test_save_writes_frames_fs_and_meta(tmp_path, frames):
    path = tmp_path / "rec.npz"
    save_recording(path, frames, fs=30.0, meta={"seed": 1})

    with np.load(path) as data:
        assert set(data.files) == {"frames", "fs", "meta"}
        assert data["frames"].dtype == np.uint16
        np.testing.assert_array_equal(data["frames"], frames)
        assert data["fs"].dtype == np.float32
        assert float(data["fs"]) == 30.0
        assert json.loads(str(data["meta"])) == {"seed": 1}


@pytest.mark.parametrize("value", [-1, ADC_MAX + 1])
def test_out_of_range_readings_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        save_recording(tmp_path / "rec.npz", np.full((2, 3), value))


def test_float_frames_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        save_recording(tmp_path / "rec.npz", np.array([[2047.9]]))


@pytest.mark.parametrize("shape", [(5,), (0, 3)])
def test_bad_shapes_are_rejected(tmp_path, shape):
    with pytest.raises(ValueError):
        save_recording(tmp_path / "rec.npz", np.zeros(shape, dtype=np.uint16))


