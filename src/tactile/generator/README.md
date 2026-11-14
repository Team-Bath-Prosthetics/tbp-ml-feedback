# Tactile dataset generator

`DatasetGenerator` creates simple, reproducible synthetic contact recordings
for exercising the current autoencoder pipeline. This simulation is a useful
software test input, not a validated physical model of the hardware.

## Recording contract

- Frames have shape `(T, *frame_shape)`, with time first. The default is
  `(T, 20, 10, 10)` at 100 Hz.
- The Week 1 mock layout `(20, 10, 5)` at 30 Hz is also supported by passing
  `frame_shape=(20, 10, 5)` and `fs=30`.
- Generated values are raw integer ADC counts in `[0, adc_max]`, returned as
  `uint16`. The default `adc_max` is 4095 (12-bit readings).
- Simulation currently requires a frame shape with 20 pads and two spatial
  dimensions. `validate_frames` can accept other non-empty layouts when no
  `frame_shape` constraint is configured.
- The shared archive convention is `frames`, `fs`, and `meta`, owned by
  `tactile.data_format`.

## Synthetic contacts

- Each contact names a pad, start time, duration, peak intensity, spatial size,
  and optional sensor-grid center.
- The default is three random events per recording. Their duration is between
  0.15 and 0.8 seconds, capped by the recording duration; their intensity and
  spatial size are randomized. Pass `n_events` to change the number, or pass
  explicit `ContactEvent` values with `events`.
- Use `seed` to make the same settings produce the same frames. `noise_std`
  controls the Gaussian background noise and defaults to 2 ADC counts.
- `generate_dataset` records the seed, sampling rate, noise setting, and
  resolved events in metadata, alongside any user-provided metadata.

## Generating and saving

`generate_frames` returns a NumPy array and can be used without the storage
API. `save_dataset` and `generate_dataset` delegate persistence to
`tactile.data_format.save_recording`; this shared writer is maintained
separately and currently raises `NotImplementedError` until implemented.

The generator rounds and clips simulated floating-point values to the ADC range
before converting them to `uint16`. For externally supplied frames,
`validate_frames` rejects non-integer or out-of-range values rather than
silently clipping them.

Example, using the repository defaults:

```python
from tactile.generator.dataset_generator import ContactEvent, DatasetGenerator
from tactile.hand_layout import Pad

generator = DatasetGenerator()
event = ContactEvent(
    pad=Pad.INDEX_TIP,
    start=0.5,
    duration=0.4,
    intensity=1800,
    size=1.2,
)

frames = generator.generate_frames(200, fs=100, events=[event], seed=42)
# Saving works once the shared data-format writer is implemented.
generator.generate_dataset(
    "data/example.npz", 200, fs=100, events=[event], seed=42
)
```

Install the development dependencies and run the focused tests with
`pytest tests/test_dataset_generator.py` from the repository root.
