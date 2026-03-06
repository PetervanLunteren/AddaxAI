"""
Tests for the shared scoring module (app.ml.scoring).
"""

import numpy as np
import pytest

from app.ml.scoring import (
    compute_sharpness,
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
# compute_sharpness
# ---------------------------------------------------------------------------


def test_compute_sharpness_uniform_image():
    """A uniform image should have very low sharpness (near zero variance)."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    sharpness = compute_sharpness(img)
    assert sharpness == pytest.approx(0.0, abs=0.01)


def test_compute_sharpness_noisy_image():
    """A noisy/edge-rich image should have higher sharpness than a uniform one."""
    uniform = np.full((100, 100, 3), 128, dtype=np.uint8)
    noisy = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)

    assert compute_sharpness(noisy) > compute_sharpness(uniform)


# ---------------------------------------------------------------------------
# pick_best_candidate
# ---------------------------------------------------------------------------


def test_pick_best_single_clear_winner():
    """One candidate has highest conf_bin — returned directly."""
    scores = {"a": (95, 0.5), "b": (80, 0.8), "c": (70, 0.9)}
    # "a" has highest conf_bin (95), alone in tier 1
    assert pick_best_candidate(scores) == "a"


def test_pick_best_tiebreak_without_sharpness():
    """Same conf_bin, area within TOP_FRACTION, no get_sharpest — largest area wins."""
    scores = {"a": (90, 1.0), "b": (90, 0.95)}
    # Both in same conf_bin, both within 90% of 1.0 (threshold = 0.9)
    result = pick_best_candidate(scores)
    assert result == "a"


def test_pick_best_tiebreak_with_sharpness():
    """Same conf_bin, area within TOP_FRACTION — get_sharpest is called."""
    scores = {"a": (90, 1.0), "b": (90, 0.95)}
    # get_sharpest picks "b" as sharpest
    result = pick_best_candidate(scores, get_sharpest=lambda keys: "b")
    assert result == "b"


def test_pick_best_empty_scores_with_fallback():
    """Empty scores + fallback_keys — delegates to get_sharpest."""
    result = pick_best_candidate(
        {},
        get_sharpest=lambda keys: keys[1],
        fallback_keys=["x", "y", "z"],
    )
    assert result == "y"


def test_pick_best_empty_scores_no_fallback():
    """Empty scores, no fallback — returns None."""
    assert pick_best_candidate({}) is None


def test_pick_best_empty_scores_no_sharpest():
    """Empty scores, fallback_keys but no get_sharpest — returns None."""
    assert pick_best_candidate({}, fallback_keys=["a", "b"]) is None


def test_pick_best_sharpness_only_called_when_needed():
    """get_sharpest should NOT be called when there's a clear winner."""
    called = []

    def spy_sharpest(keys):
        called.append(keys)
        return keys[0]

    # "a" has higher conf_bin — wins at tier 1, no sharpness needed
    scores = {"a": (95, 0.5), "b": (90, 1.0)}
    result = pick_best_candidate(scores, get_sharpest=spy_sharpest)
    assert result == "a"
    assert called == []  # not called


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
