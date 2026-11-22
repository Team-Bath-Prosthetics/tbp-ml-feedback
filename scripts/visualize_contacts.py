"""Visualize synthetic tactile contact pressure over time.

Run this script from the repository root:

    python scripts/visualize_contacts.py hold
    python scripts/visualize_contacts.py sliding
    python scripts/visualize_contacts.py rotation

The contact argument selects one synthetic motion. The script generates a
recording at --fs Hz for --frames frames, then saves an animated GIF under
pad_visualizations/<contact>.gif.

Examples:

    python scripts/visualize_contacts.py sliding --frames 200 --fs 100
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from tactile.generator.dataset_generator import ContactEvent, DatasetGenerator
from tactile.hand_layout import Pad
from tactile.viz.animate_recording import visualize_pad_over_time


def build_event(contact_type: str) -> ContactEvent:
    if contact_type == "hold":
        return ContactEvent(
            pad=Pad.INDEX_TIP,
            start=0.1,
            duration=0.6,
            intensity=1000,
            size=1.0,
            center=(5.0, 5.0),
            contact_type="hold",
        )
    if contact_type == "sliding":
        return ContactEvent(
            pad=Pad.INDEX_TIP,
            start=0.1,
            duration=0.7,
            intensity=1000,
            size=1.0,
            center=(2.0, 2.0),
            end_center=(7.0, 7.0),
            contact_type="sliding",
        )
    if contact_type == "rotation":
        return ContactEvent(
            pad=Pad.INDEX_TIP,
            start=0.1,
            duration=0.8,
            intensity=1000,
            size=1.0,
            center=(5.0, 5.0),
            rotation_degrees=180.0,
            rotation_radius=2.0,
            contact_type="rotation",
        )
    raise ValueError("contact type must be hold, sliding, or rotation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synthetic pad pressure over time")
    parser.add_argument(
        "contact",
        nargs="?",
        choices=("hold", "sliding", "rotation"),
        default="hold",
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fs", type=float, default=100.0)
    args = parser.parse_args()

    event = build_event(args.contact)
    frames = DatasetGenerator().generate_frames(
        args.frames,
        fs=args.fs,
        events=[event],
        seed=42,
        noise_std=0,
    )

    visualize_pad_over_time(
        frames,
        fs=args.fs,
        pad=Pad.INDEX_TIP,
        save_path=Path("pad_visualizations") / f"{args.contact}.gif",
    )


if __name__ == "__main__":
    main()
