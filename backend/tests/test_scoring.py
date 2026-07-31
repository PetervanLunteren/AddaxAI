"""
Tests for the shared scoring module (app.ml.inference.scoring).
"""

import pytest

from app.ml.inference.scoring import (
    choose_frame_number,
    compute_union_area,
    pick_best_candidate,
    score_detections,
)

UNIT_BBOX = (0.0, 0.0, 1.0, 1.0)  # area = 1.0, sqrt(area) = 1.0


# ---------------------------------------------------------------------------
# score_detections
# ---------------------------------------------------------------------------


def test_score_detections_sums_by_key():
    detections = [
        ("a", 0.8, UNIT_BBOX),
        ("a", 0.5, UNIT_BBOX),
        ("b", 0.9, UNIT_BBOX),
    ]
    scores = score_detections(detections)
    # a: conf_bin = round(0.8*100) + round(0.5*100) = 80+50 = 130, sqrt(1.0) = 1.0
    # b: conf_bin = round(0.9*100) = 90, sqrt(1.0) = 1.0
    assert scores == {
        "a": (130, pytest.approx(1.0)),
        "b": (90, pytest.approx(1.0)),
    }


def test_score_detections_filters_below_threshold():
    detections = [
        ("a", 0.1, UNIT_BBOX),   # below threshold
        ("a", 0.2, UNIT_BBOX),   # below threshold
        ("b", 0.29, UNIT_BBOX),  # below threshold
        ("c", 0.3, UNIT_BBOX),   # exactly at threshold
    ]
    scores = score_detections(detections)
    assert "a" not in scores
    assert "b" not in scores
    assert scores == {"c": (30, pytest.approx(1.0))}


def test_score_detections_empty():
    assert score_detections([]) == {}


def test_score_detections_all_below_threshold():
    detections = [("a", 0.1, UNIT_BBOX), ("b", 0.2, UNIT_BBOX)]
    assert score_detections(detections) == {}


# ---------------------------------------------------------------------------
# compute_union_area
# ---------------------------------------------------------------------------


def test_union_area_empty():
    assert compute_union_area([]) == 0.0


def test_union_area_single_box():
    assert compute_union_area([(0.0, 0.0, 0.5, 0.5)]) == pytest.approx(0.25)


def test_union_area_non_overlapping():
    boxes = [(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)]
    assert compute_union_area(boxes) == pytest.approx(0.5)


def test_union_area_fully_overlapping():
    boxes = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    assert compute_union_area(boxes) == pytest.approx(1.0)


def test_union_area_partial_overlap():
    # Two 0.5x0.5 boxes overlapping in a 0.25x0.5 strip
    boxes = [(0.0, 0.0, 0.5, 0.5), (0.25, 0.0, 0.5, 0.5)]
    # Union = 0.75 * 0.5 = 0.375
    assert compute_union_area(boxes) == pytest.approx(0.375)


def test_union_area_contained_box():
    # Small box fully inside large box
    boxes = [(0.0, 0.0, 1.0, 1.0), (0.25, 0.25, 0.5, 0.5)]
    assert compute_union_area(boxes) == pytest.approx(1.0)


def test_union_area_degenerate_box():
    # Zero-width and zero-height boxes are skipped
    boxes = [(0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.5, 0.0)]
    assert compute_union_area(boxes) == 0.0


def test_union_area_mixed_degenerate_and_valid():
    boxes = [(0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.5, 0.5)]
    assert compute_union_area(boxes) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# score_detections: bbox area affects ranking
# ---------------------------------------------------------------------------


def test_score_big_box_beats_small_box():
    """Same confidence, but bigger bbox should have larger sqrt_area."""
    big_box = (0.0, 0.0, 0.8, 0.8)   # area = 0.64, sqrt = 0.8
    small_box = (0.0, 0.0, 0.1, 0.1)  # area = 0.01, sqrt = 0.1
    detections = [
        ("big", 0.9, big_box),
        ("small", 0.9, small_box),
    ]
    scores = score_detections(detections)
    # Same conf_bin (90), so tuples compared lexicographically by sqrt_area
    assert scores["big"] > scores["small"]


def test_score_spread_beats_stacked():
    """Spread-out boxes have larger union area than stacked boxes."""
    spread = [
        ("spread", 0.9, (0.0, 0.0, 0.3, 0.3)),
        ("spread", 0.9, (0.5, 0.5, 0.3, 0.3)),
    ]
    stacked = [
        ("stacked", 0.9, (0.0, 0.0, 0.3, 0.3)),
        ("stacked", 0.9, (0.0, 0.0, 0.3, 0.3)),
    ]
    scores = score_detections(spread + stacked)
    # Same conf_bin (180), so compared by sqrt_area
    assert scores["spread"] > scores["stacked"]


def test_score_more_individuals_beats_fewer():
    """More individuals should outscore fewer via higher conf_bin sum."""
    many = [
        ("many", 0.8, (0.0, 0.0, 0.2, 0.2)),
        ("many", 0.8, (0.3, 0.0, 0.2, 0.2)),
        ("many", 0.8, (0.6, 0.0, 0.2, 0.2)),
        ("many", 0.8, (0.0, 0.3, 0.2, 0.2)),
    ]
    single = [
        ("single", 0.9, (0.0, 0.0, 0.6, 0.6)),
    ]
    scores = score_detections(many + single)
    # many: conf_bin = 4*80 = 320; single: conf_bin = 90
    assert scores["many"] > scores["single"]


# ---------------------------------------------------------------------------
# pick_best_candidate
# ---------------------------------------------------------------------------


def test_pick_best_single_clear_winner():
    """One candidate has highest conf_bin — returned directly."""
    scores = {"a": (95, 0.5), "b": (80, 0.8), "c": (70, 0.9)}
    # "a" has highest conf_bin (95), alone in tier 1
    assert pick_best_candidate(scores) == "a"


def test_pick_best_area_breaks_a_confidence_tie():
    """Same conf_bin, both areas within TOP_FRACTION — largest area wins.

    This is the last tier. Sharpness used to sit below it and was removed
    on 2026-07-31: it needed the pixels of every candidate, which forced
    the caller to decode and hold the whole candidate set to settle a
    tiebreak that never fired on real footage.
    """
    scores = {"a": (90, 1.0), "b": (90, 0.95)}
    # Both in same conf_bin, both within 90% of 1.0 (threshold = 0.9)
    assert pick_best_candidate(scores) == "a"


def test_pick_best_empty_scores():
    """A blank video has nothing to anchor on; the caller picks by
    position instead (see choose_frame_number)."""
    assert pick_best_candidate({}) is None


def test_pick_best_confidence_beats_larger_area():
    """Higher confidence wins even when the other candidate has a much larger bbox.

    Reproduces the lynx scenario: 93% conf with smaller bbox should beat
    91% conf with 1.88x larger bbox.
    """
    # REC0007: 93% conf, smaller bbox (area ≈ 0.029)
    sharp = (93, pytest.approx(0.170, abs=0.01))
    # frame24: 91% conf, larger bbox (area ≈ 0.055)
    blurry = (91, pytest.approx(0.234, abs=0.01))

    scores = {"sharp": sharp, "blurry": blurry}
    result = pick_best_candidate(scores)
    assert result == "sharp"


# ---------------------------------------------------------------------------
# choose_frame_number
# ---------------------------------------------------------------------------


def _d(frame_number, conf, bbox=(0.4, 0.4, 0.2, 0.2)):
    return {"frame_number": frame_number, "conf": conf, "bbox": list(bbox)}


def test_choose_frame_number_takes_the_strongest_frame():
    """Decided from the detection list alone, no pixels involved. That is
    what lets the caller decode only the frame it keeps."""
    dets = [_d(0, 0.40), _d(30, 0.95), _d(60, 0.55)]
    assert choose_frame_number(dets, total_frames=90) == 30


def test_choose_frame_number_sums_within_a_frame():
    """Two solid detections on one frame beat one better detection
    elsewhere: the frame showing more of the scene's subjects wins."""
    dets = [_d(0, 0.90), _d(30, 0.50, (0.1, 0.1, 0.2, 0.2)), _d(30, 0.50, (0.6, 0.1, 0.2, 0.2))]
    assert choose_frame_number(dets, total_frames=60) == 30


def test_choose_frame_number_ignores_category():
    """`conf` is the only field read. Nothing here says what the box
    contains, which is what makes this work for any detector."""
    dets = [_d(10, 0.99)]
    assert choose_frame_number(dets, total_frames=100) == 10


def test_choose_frame_number_blank_video_takes_the_middle():
    """No detections: position is as good a choice as any, and the middle
    beats the first frame because camera traps often open on the empty
    scene that triggered them."""
    assert choose_frame_number([], total_frames=100) == 50


def test_choose_frame_number_all_below_the_floor_takes_the_middle():
    """Detections under CONFIDENCE_THRESHOLD score nothing, so the video
    is treated as blank rather than anchored on noise."""
    dets = [_d(10, 0.05), _d(20, 0.09)]
    assert choose_frame_number(dets, total_frames=100) == 50


def test_choose_frame_number_survives_a_zero_length_video():
    """cv2 can report 0 frames. Must not return a negative index."""
    assert choose_frame_number([], total_frames=0) == 0
