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
# TODO: Prevent overwriting if a file with that name already exists. 

import json
from pathlib import Path

import numpy as np

FRAME_SHAPE = (20, 10, 10)
DEFAULT_FS = 100.0
ADC_BITS = 12
ADC_MAX = 2**ADC_BITS - 1  # 4095
DATA_DIR = Path(__file__).resolve().parents[2] / "data" # requires editable install

def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else DATA_DIR / path

def save_recording(path: str | Path, frames: np.ndarray, *, fs: float = DEFAULT_FS, meta: dict | None = None) -> None:
    """Save one recording to a compressed .npz file.

    `frames` must already hold integer readings in 0..ADC_MAX. Convert float
    generator output first, e.g. np.clip(np.rint(x), 0, ADC_MAX), because
    casting to uint16 truncates and wraps out-of-range values.
    """
    frames = np.asarray(frames)
    if frames.ndim < 2 or 0 in frames.shape:
        raise ValueError(f"frames must be (T, *frame_shape) with no empty axes, got {frames.shape}")
    if not np.issubdtype(frames.dtype, np.integer):
        raise TypeError(f"frames must be integers, got {frames.dtype}")
    lo, hi = frames.min(), frames.max()
    if lo < 0 or hi > ADC_MAX:
        raise ValueError(f"frames must be in 0..{ADC_MAX}, got range {lo}..{hi}")

    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file=path,
        frames=frames.astype(np.uint16, copy=False),
        fs=np.float32(fs),
        meta=np.array(json.dumps(meta or {})),
    )


def load_recording(path: str | Path) -> dict:
    """Load a recording saved by save_recording. Returns a dict of arrays; `meta` is a dict."""
    path = _resolve(path)

    with np.load(path) as data:
        missing = {"frames", "fs", "meta"} - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        
        frames = data["frames"]
        fs = float(data["fs"])
        meta = json.loads(str(data["meta"]))

    return {"frames": frames, "fs": fs, "meta": meta}



if __name__ == "__main__":
    frames = np.random.randint(0, ADC_MAX + 1, size=(30, *FRAME_SHAPE))
    save_recording("john.npz", frames, fs=30)
    rec = load_recording("john.npz")
    print(rec["fs"], rec["meta"], rec["frames"])