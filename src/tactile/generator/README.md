# Tactile dataset generator

Week 1 foundation for Team Bath Prosthetics' Novel Feedback dataset generator.

## Scope

`DatasetGenerator` validates a time-series NumPy array and forwards it to the
repository's shared `tactile.data_format.save_recording` function. It does **not**
create synthetic pressure patterns yet; that is a separate workstream.

> **Integration status:** `data_format.save_recording` currently raises
> `NotImplementedError`. Unit tests mock that function to verify the interface.
> Actual `.npz` saving will work only after the shared writer is implemented.

## Data contract

- Shape: `(T, *frame_shape)` with nonempty time and sensor axes.
- Week 1 example: `(T, 20, 10, 5)` at **30 Hz**.
- Repository's current generic frame shape: `(T, 20, 10, 10)` at **100 Hz**.
- Values: integer ADC readings in `[0, 4095]`, converted to `uint16` only after validation.
- Shared archive convention (owned by `data_format.py`): `frames`, `fs`, `meta`.
- Other sensor layouts are allowed while hardware specifications evolve.

## Example

From an environment where the repository package is installed (`pip install -e '.[dev,ml]'`):

```python
import numpy as np
from tactile.generator.dataset_generator import DatasetGenerator

frames = np.zeros((30, 20, 10, 5), dtype=np.uint16)
generator = DatasetGenerator(frame_shape=(20, 10, 5))

generator.validate_frames(frames)  # Can be used before shared storage is ready
# After data_format.save_recording is implemented:
generator.save_dataset(frames, "data/example.npz", fs=30.0,
                       meta={"source": "week1_mock"})
```

## Tests

Run `pytest` from the repository root after installing its development extras.
The tests mock the shared writer so they can pass before the storage API is complete.

## Extending the module

Keep sensor simulation separate from validation and persistence. New generator
methods should produce NumPy frames and call `save_dataset` rather than writing
`.npz` files independently. Do not silently clip out-of-range readings; if a
simulation produces floats, round/clip explicitly at the simulation boundary.

## Known decisions to confirm with the team

- Whether the Week 1 30 Hz configuration or the repository's 100 Hz default is
  expected for a given recording.
- Who will implement `data_format.save_recording` and `load_recording`.
- Final sensor layout and metadata naming conventions.
