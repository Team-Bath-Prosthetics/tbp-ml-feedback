"""Unit tests for the generator; mock the unfinished shared storage writer."""

from unittest.mock import patch

import numpy as np
import pytest

from tactile.generator.dataset_generator import DatasetGenerator


def test_week_one_frames_are_accepted():
    frames = np.zeros((30, 20, 10, 5), dtype=np.uint16)
    assert DatasetGenerator(frame_shape=(20, 10, 5)).validate_frames(frames).shape == frames.shape


def test_repository_default_layout_is_accepted():
    frames = np.zeros((2, 20, 10, 10), dtype=np.uint16)
    assert DatasetGenerator().validate_frames(frames).shape == frames.shape


def test_invalid_shape_is_rejected():
    with pytest.raises(ValueError, match="expected frame shape"):
        DatasetGenerator((20, 10, 5)).validate_frames(np.zeros((2, 20, 10, 10), dtype=np.uint16))


@pytest.mark.parametrize("value", [-1, 4096])
def test_invalid_reading_is_rejected(value):
    with pytest.raises(ValueError, match="sensor readings"):
        DatasetGenerator().validate_frames(np.full((2, 3), value, dtype=np.int32))


def test_float_readings_are_rejected():
    with pytest.raises(TypeError, match="integer"):
        DatasetGenerator().validate_frames(np.zeros((2, 3), dtype=float))


def test_empty_recording_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        DatasetGenerator().validate_frames(np.zeros((0, 20, 10, 5), dtype=np.uint16))


def test_save_delegates_to_shared_writer():
    frames = np.ones((3, 20, 10, 5), dtype=np.int32)
    with patch("tactile.generator.dataset_generator.data_format.save_recording") as writer:
        DatasetGenerator((20, 10, 5)).save_dataset(frames, "sample.npz", fs=30, meta={"source": "synthetic"})
    writer.assert_called_once()
    args, kwargs = writer.call_args
    assert args[0] == "sample.npz"
    np.testing.assert_array_equal(args[1], frames)
    assert args[1].dtype == np.uint16
    assert kwargs == {"fs": 30.0, "meta": {"source": "synthetic"}}


def test_invalid_sampling_rate_is_rejected():
    with pytest.raises(ValueError, match="sampling rate"):
        DatasetGenerator().save_dataset(np.ones((2, 3), dtype=np.uint16), "test.npz", fs=0)


def test_invalid_metadata_is_rejected():
    with pytest.raises(TypeError, match="mapping"):
        DatasetGenerator().save_dataset(np.ones((2, 3), dtype=np.uint16), "test.npz", meta="not a dict")
