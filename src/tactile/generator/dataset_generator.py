"""Generate synthetic tactile recordings and delegate persistence to the shared format API.

The repository's :mod:`tactile.data_format` owns the on-disk NPZ schema.
This module deliberately does not write NPZ archives itself.
"""

# TODO: dataset generator fails when fs > frames
#TODO: make_sure generator intensitiy varies based on ADC_MAX. Assumes ADC_MAX = 4095

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tactile import data_format
from tactile.data_format import ADC_MAX, DEFAULT_FS, FRAME_SHAPE
from tactile.hand_layout import NUM_PADS, Pad


@dataclass(frozen=True)
class ContactEvent:
    """One smooth pressure patch, with times in seconds and size in sensor cells."""

    pad: Pad
    start: float
    duration: float
    intensity: float # peak ADC-count (0..ADC_MAX)
    size: float = 1.5 # standard deviation of the Gaussian patch
    center: tuple[float, float] | None = None


class DatasetGenerator:
    """Generate, validate, and save tactile recordings.

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

    def _synthesize(
        self,
        num_frames: int,
        *, # pass by name explicitly
        fs: float,
        n_events: int | None, # generator creates n random events
        events: Sequence[ContactEvent] | None, # user-specified events (use either this or n_events, not both)
        seed: int | None,
        noise_std: float,
    ) -> tuple[np.ndarray, tuple[ContactEvent, ...]]:
        if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        if not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
            raise ValueError("fs must be a positive, finite sampling rate")
        if not isinstance(noise_std, (int, float)) or not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be a finite, non-negative value")
        if events is not None and n_events is not None:
            raise ValueError("provide either events or n_events, not both")
        if n_events is not None and (
            not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < 0
        ):
            raise ValueError("n_events must be a non-negative integer")

        frame_shape = self.frame_shape or FRAME_SHAPE
        if len(frame_shape) != 3 or frame_shape[0] != NUM_PADS:
            raise ValueError(f"simulation requires frame_shape ( {NUM_PADS}, height, width )")
        _, height, width = frame_shape
        rng = np.random.default_rng(seed)
        recording_duration = num_frames / fs # time of recording in seconds

        if events is None:
            count = 3 if n_events is None else n_events # generates 3 random events if neither events nor n_events is specified
            sampled_events = []
            for _ in range(count):
                # generate random events between 0.15 and 0.8 seconds
                max_duration = min(0.8, recording_duration)
                min_duration = min(0.15, max_duration)
                duration = float(rng.uniform(min_duration, max_duration)) if max_duration > min_duration else max_duration
                latest_start = max(0.0, recording_duration - duration)
                start = float(rng.uniform(0.0, latest_start)) if latest_start > 0 else 0.0
                sampled_events.append(
                    ContactEvent(
                        pad=Pad(int(rng.integers(NUM_PADS))),
                        start=start,
                        duration=duration,
                        intensity=float(rng.uniform(500.0, 3000.0)), #TODO: make_sure intensitiy varies based on ADC_MAX. Assumes ADC_MAX = 4095
                        size=float(rng.uniform(0.7, 2.0)), # standard deviation of the Gaussian patch
                        center=(
                            float(rng.uniform(0.0, max(0, height - 1))),
                            float(rng.uniform(0.0, max(0, width - 1))),
                        ),
                    )
                )
            resolved_events = tuple(sampled_events)
        else:
            resolved_events = tuple(events)

        rows, columns = np.mgrid[0:height, 0:width]
        times = np.arange(num_frames, dtype=np.float64) / fs # time array [0.00, 0.01, 0.02, ..., recording_duration]
        frames = rng.normal(0.0, noise_std, size=(num_frames, *frame_shape)) # recording array filled with Gaussian noise

        normalized_events = []
        for event in resolved_events:
            if not isinstance(event, ContactEvent):
                raise TypeError("events must contain ContactEvent values")
            try:
                pad = Pad(event.pad)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid pad in contact event: {event.pad!r}") from exc
            values = (event.start, event.duration, event.intensity, event.size)
            if not all(np.isfinite(value) for value in values):
                raise ValueError("contact event values must be finite")
            if event.start < 0 or event.start >= recording_duration:
                raise ValueError("contact event start must fall within the recording")
            if event.duration <= 0 or event.intensity < 0 or event.size <= 0:
                raise ValueError("event duration and size must be positive; intensity cannot be negative")

            center = event.center or ((height - 1) / 2, (width - 1) / 2)
            if len(center) != 2 or not all(np.isfinite(value) for value in center):
                raise ValueError("event center must contain finite row and column coordinates")
            center_row, center_column = center
            if not (0 <= center_row < height and 0 <= center_column < width):
                raise ValueError("event center must lie within the sensor grid")

            event_end = event.start + event.duration
            active = (times >= event.start) & (times < event_end) # mask e.g. [False, False, True, True, ... False] where the event can be active
            active_times = times[active] # get the actual times where the event is active, e.g. [0.02, 0.03, 0.04, ...]
            if active_times.size:
                ramp = min(event.duration * 0.2, 0.1) # up/down time between event e.g. 0.1 seconds until full intensity is reached
                envelope = np.minimum( # touch strength on each active frame
                    1.0,
                    np.minimum((active_times - event.start) / ramp, (event_end - active_times) / ramp),
                )
                pressure = np.exp(
                    -((rows - center_row) ** 2 + (columns - center_column) ** 2)
                    / (2.0 * event.size**2)
                )
                frames[active, int(pad)] += event.intensity * envelope[:, None, None] * pressure

            normalized_events.append( # store the cleaned-up event
                ContactEvent(
                    pad=pad,
                    start=float(event.start),
                    duration=float(event.duration),
                    intensity=float(event.intensity),
                    size=float(event.size),
                    center=(float(center_row), float(center_column)),
                )
            )

        quantized = np.clip(np.rint(frames), 0, self.adc_max).astype(np.uint16)
        return quantized, tuple(normalized_events)

    def generate_frames(
        self,
        num_frames: int,
        *,
        fs: float = DEFAULT_FS,
        n_events: int | None = None,
        events: Sequence[ContactEvent] | None = None,
        seed: int | None = None,
        noise_std: float = 2.0,
    ) -> np.ndarray:
        """Generate ``(T, *frame_shape)`` uint16 frames with seeded contact events.

        If ``events`` is omitted, three random contact events are generated by
        default. Event ``size`` is the spatial Gaussian's standard deviation
        in sensor cells; event ``intensity`` is its peak ADC-count increase.
        """
        frames, _ = self._synthesize(
            num_frames,
            fs=fs,
            n_events=n_events,
            events=events,
            seed=seed,
            noise_std=noise_std,
        )
        return frames

    def generate_dataset(
        self,
        path: str | Path,
        num_frames: int,
        *,
        fs: float = DEFAULT_FS,
        n_events: int | None = None,
        events: Sequence[ContactEvent] | None = None,
        seed: int | None = None,
        noise_std: float = 2.0,
        meta: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Generate frames, record simulation settings in metadata, and delegate saving."""
        if meta is not None and not isinstance(meta, Mapping):
            raise TypeError("meta must be a mapping or None")
        frames, resolved_events = self._synthesize(
            num_frames,
            fs=fs,
            n_events=n_events,
            events=events,
            seed=seed,
            noise_std=noise_std,
        )
        settings = {
            "generator": "synthetic_contact_v1",
            "seed": seed,
            "num_frames": num_frames,
            "fs": float(fs),
            "noise_std": float(noise_std),
            "events": [
                {**asdict(event), "pad": event.pad.name}
                for event in resolved_events
            ],
        }
        recording_meta = {**(dict(meta) if meta is not None else {}), **settings}
        self.save_dataset(frames, path, fs=fs, meta=recording_meta)
        return frames

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
        fs: float = DEFAULT_FS,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        """Save frames using the repository-wide ``save_recording`` API.

        ``fs`` is the recording frequency in Hz (100 Hz by default).
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