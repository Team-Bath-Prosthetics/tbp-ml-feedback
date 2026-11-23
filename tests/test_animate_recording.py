import matplotlib

matplotlib.use("Agg")  # no GUI windows during tests

import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from tactile.hand_layout import NUM_PADS, Pad
from tactile.viz.animate_recording import animate_recording, pads_to_image


def labelled_pads():
    """Each pad filled with its own index, so we can see where it ends up."""
    return np.broadcast_to(np.arange(NUM_PADS)[:, None, None], (NUM_PADS, 10, 10)).copy()

def block(image, row, col):
    return image[row * 10:(row + 1) * 10, col * 10:(col + 1) * 10]

def test_animate_recording_runs():
    frames = np.random.randint(0, 4096, size=(5, 20, 10, 10))
    assert isinstance(animate_recording(frames, fs=10, show=False), FuncAnimation)
