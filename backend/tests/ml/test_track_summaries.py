"""`summarise_tracks`: the one place that decides what a track is made of.

The ingest stores these numbers, and the tracking script keeps the crop
of the frame it names, so both must read the JSON the same way.
Pixel-free like `choose_frame_number`.
"""

from app.ml.inference.scoring import (
    TrackSummary,
    summarise_tracks,
    track_crop_filename,
)


def _box(frame: int, conf: float, track: int | None):
    det = {"category": "0", "conf": conf, "bbox": [0.1, 0.1, 0.2, 0.2], "frame_number": frame}
    if track is not None:
        det["track_id"] = track
    return det


def test_one_summary_per_track_with_the_highest_confidence_frame():
    dets = [
        _box(30, 0.5, 1),
        _box(60, 0.9, 1),
        _box(90, 0.7, 1),
        _box(300, 0.4, 2),
        _box(330, 0.4, 2),
    ]
    got = summarise_tracks(dets)
    assert got == {
        1: TrackSummary(30, 90, 3, 0.9, 60),
        # A tie on confidence goes to the earliest frame.
        2: TrackSummary(300, 330, 2, 0.4, 300),
    }


def test_boxes_without_a_track_are_ignored():
    """A run without tracking, and a box a person drew, have no track id
    and must not become a phantom track."""
    assert summarise_tracks([_box(10, 0.9, None), _box(20, 0.8, None)]) == {}
    assert summarise_tracks([]) == {}


def test_frame_count_counts_frames_not_boxes():
    """Two boxes of one track on one frame cannot happen from a tracker,
    but a hand-edited JSON could carry them; the count stays honest."""
    dets = [_box(30, 0.5, 1), _box(30, 0.6, 1), _box(60, 0.4, 1)]
    assert summarise_tracks(dets)[1].frame_count == 2


def test_track_ids_are_read_as_integers():
    """The JSON may carry the id as a string; the row key is an int."""
    got = summarise_tracks([_box(30, 0.5, "7")])
    assert list(got) == [7]


def test_the_crop_is_named_by_its_track_key():
    """`track`, not `frame`, so the startup sweep of stale `frame*.jpg`
    stills never touches a card."""
    assert track_crop_filename(7) == "track000007.jpg"
    assert track_crop_filename("12") == "track000012.jpg"
