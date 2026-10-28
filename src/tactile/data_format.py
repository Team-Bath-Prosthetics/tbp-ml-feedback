"""Shared on-disk format for tactile recordings.

Every part of the project (dataset generator, visualiser, autoencoder training)
saves and loads recordings through these two functions, so the format only
lives in one place.

A recording is one .npz file containing:

    frames   (T, *frame_shape) uint16   raw sensor readings, 0..adc_max
    fs       scalar float32             sampling rate in Hz
    meta     0-d str                    JSON metadata (seed, generator settings, ...)

T is the number of frames (time). The default frame shape is (20, 10, 10):
20 pads, each a 10x10 grid of sensors. Other frame shapes are allowed, because
the real hardware layout is not decided yet.
"""

import json
from pathlib import Path

import numpy as np

FRAME_SHAPE = (20, 10, 10)
DEFAULT_FS = 100.0
ADC_BITS = 12
ADC_MAX = 2**ADC_BITS - 1  # 4095


def save_recording(path, frames, *, fs=DEFAULT_FS, labels=None, contact=None, meta=None):
    """Save one recording to a compressed .npz file.

    `frames` must already hold integer readings in 0..ADC_MAX. Convert float
    generator output first, e.g. np.clip(np.rint(x), 0, ADC_MAX), because
    casting to uint16 truncates and wraps out-of-range values.
    """
    raise NotImplementedError


def load_recording(path):
    """Load a recording saved by save_recording. Returns a dict of arrays; `meta` is a dict."""
    raise NotImplementedError
