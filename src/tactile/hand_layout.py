"""Names for the 20 sensor pads on the hand.

A frame has shape (20, 10, 10): 20 pads, each a 10x10 grid of sensors. This
module says which pad is which, so code can write:

    frames[:, Pad.INDEX_TIP]

instead of frames[:, 2]. 

The layout follows the team's sensor-placement
diagram (palm side of the hand):

- Fingers: TIP is the fingertip segment, MIDDLE the middle segment, BASE the
  segment nearest the palm. PALM is the pad on the palm just below that finger.
- The thumb has two segments (TIP, BASE).
- The pinky has a TIP and MIDDLE pad but no BASE pad.
- THENAR pads are the cluster of three on the fleshy base of the thumb:
  UPPER (nearest the fingers), INNER (towards the centre of the palm) and
  LOWER (nearest the wrist).

Order runs thumb -> pinky, fingertip -> palm, then the thenar cluster.
"""

from enum import IntEnum


class Pad(IntEnum):
    """Index of each pad in the second axis of a (T, 20, 10, 10) recording."""

    THUMB_TIP = 0
    THUMB_BASE = 1

    INDEX_TIP = 2
    INDEX_MIDDLE = 3
    INDEX_BASE = 4
    INDEX_PALM = 5

    MIDDLE_TIP = 6
    MIDDLE_MIDDLE = 7
    MIDDLE_BASE = 8
    MIDDLE_PALM = 9

    RING_TIP = 10
    RING_MIDDLE = 11
    RING_BASE = 12
    RING_PALM = 13

    PINKY_TIP = 14
    PINKY_MIDDLE = 15
    PINKY_PALM = 16

    THENAR_UPPER = 17
    THENAR_INNER = 18
    THENAR_LOWER = 19


NUM_PADS = len(Pad)

# PAD_NAMES[i] is the name of pad i, e.g. for plot labels.
PAD_NAMES = tuple(pad.name for pad in Pad)

# Pads grouped by region, thumb -> pinky then thenar.
REGIONS = {
    "THUMB": (Pad.THUMB_TIP, Pad.THUMB_BASE),
    "INDEX": (Pad.INDEX_TIP, Pad.INDEX_MIDDLE, Pad.INDEX_BASE, Pad.INDEX_PALM),
    "MIDDLE": (Pad.MIDDLE_TIP, Pad.MIDDLE_MIDDLE, Pad.MIDDLE_BASE, Pad.MIDDLE_PALM),
    "RING": (Pad.RING_TIP, Pad.RING_MIDDLE, Pad.RING_BASE, Pad.RING_PALM),
    "PINKY": (Pad.PINKY_TIP, Pad.PINKY_MIDDLE, Pad.PINKY_PALM),
    "THENAR": (Pad.THENAR_UPPER, Pad.THENAR_INNER, Pad.THENAR_LOWER),
}
