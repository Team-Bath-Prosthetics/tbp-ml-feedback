from tactile.data_format import FRAME_SHAPE
from tactile.hand_layout import NUM_PADS, PAD_NAMES, REGIONS, Pad


def test_pad_count_matches_frame_shape():
    assert NUM_PADS == FRAME_SHAPE[0]


def test_indices_are_contiguous_from_zero():
    assert sorted(int(p) for p in Pad) == list(range(NUM_PADS))


def test_pad_names_line_up_with_indices():
    assert PAD_NAMES[Pad.INDEX_TIP] == "INDEX_TIP"
    assert len(PAD_NAMES) == NUM_PADS


def test_regions_cover_every_pad_once():
    grouped = [p for pads in REGIONS.values() for p in pads]
    assert sorted(grouped) == list(Pad)
