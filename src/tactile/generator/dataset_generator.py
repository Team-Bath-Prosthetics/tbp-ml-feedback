"""Validate tactile recordings and delegate persistence to the shared format API.

The repository's :mod:`tactile.data_format` owns the on-disk NPZ schema.
This module deliberately does not write NPZ archives itself.
"""

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tactile import data_format


class DatasetGenerator:
    """Prepare and save recordings from tactile sensors.

    Parameters
    ----------
    frame_shape:
        Expected shape of a *single* frame, excluding the time dimension.
        ``None`` (default) accepts any non-empty frame shape so hardware layouts
        can evolve. Set ``(20, 10, 5)`` for the Week 1 mock hand.
    adc_max:
        Maximum permitted sensor reading, inclusive (12-bit ADC: 4095).
    """

    def __init__(self, frame_shape: tuple[int, ...] | None = None, adc_max: int = 4095) -> None:
        if frame_shape is not None and (not frame_shape or any(not isinstance(n, int) or n <= 0 for n in frame_shape)):
            raise ValueError("frame_shape must contain positive integer dimensions")
        if not isinstance(adc_max, int) or not 0 <= adc_max <= np.iinfo(np.uint16).max:
            raise ValueError("adc_max must be between 0 and 65535")
        self.frame_shape = frame_shape
        self.adc_max = adc_max

    def validate_frames(self, frames: np.ndarray) -> np.ndarray:
        """Return validated uint16 frames without silently clipping readings.

        Raises TypeError for non-NumPy/non-integer input and ValueError for
        invalid shapes or sensor readings outside the ADC range.
        """
        if not isinstance(frames, np.ndarray):
            raise TypeError("frames must be a NumPy array")
        if frames.ndim < 2 or any(dim == 0 for dim in frames.shape):
            raise ValueError("frames must have a non-empty time axis and sensor dimensions")
        if self.frame_shape is not None and frames.shape[1:] != self.frame_shape:
            raise ValueError(f"expected frame shape {self.frame_shape}, got {frames.shape[1:]}")
        if not np.issubdtype(frames.dtype, np.integer):
            raise TypeError("frames must contain integer sensor readings")
        if np.any(frames < 0) or np.any(frames > self.adc_max):
            raise ValueError(f"sensor readings must be between 0 and {self.adc_max}")
        return frames.astype(np.uint16, copy=False)

    def save_dataset(
        self,
        frames: np.ndarray,
        path: str | Path,
        *,
        fs: float = 30.0,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        """Save frames using the repository-wide ``save_recording`` API.

        ``fs`` is the recording frequency in Hz. The Week 1 brief uses 30 Hz;
        the repository's generic default is 100 Hz, so specify it explicitly.
        ``meta`` is passed through to the shared writer for JSON encoding.

        Note: as of Week 1, ``data_format.save_recording`` is an unfinished
        stub and raises NotImplementedError. This class is ready to integrate
        when the shared writer is implemented.
        """
        if not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
            raise ValueError("fs must be a positive, finite sampling rate")
        if meta is not None and not isinstance(meta, Mapping):
            raise TypeError("meta must be a mapping or None")
        validated = self.validate_frames(frames)
        data_format.save_recording(path, validated, fs=float(fs), meta=dict(meta) if meta is not None else None)
