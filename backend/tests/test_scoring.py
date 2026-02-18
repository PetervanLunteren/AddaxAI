"""
Tests for the shared scoring module (app.ml.scoring).
"""

import numpy as np

from app.ml.scoring import (
    CONFIDENCE_THRESHOLD,
    TOP_FRACTION,
    compute_sharpness,
    pick_best_candidate,
    score_detections,
)


# ---------------------------------------------------------------------------
# score_detections
# ---------------------------------------------------------------------------


def test_score_detections_sums_by_key():
    detections = [
        ("a", 0.8),
        ("a", 0.5),
        ("b", 0.9),
    ]
    scores = score_detections(detections)
    assert scores == {"a": pytest.approx(1.3), "b": pytest.approx(0.9)}


def test_score_detections_filters_below_threshold():
    detections = [
        ("a", 0.1),   # below threshold
        ("a", 0.2),   # below threshold
        ("b", 0.29),  # below threshold
        ("c", 0.3),   # exactly at threshold
    ]
    scores = score_detections(detections)
    assert "a" not in scores
    assert "b" not in scores
    assert scores == {"c": pytest.approx(0.3)}


def test_score_detections_empty():
    assert score_detections([]) == {}


def test_score_detections_all_below_threshold():
    detections = [("a", 0.1), ("b", 0.2)]
    assert score_detections(detections) == {}


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
    """One candidate scores way above the rest — returned directly."""
    scores = {"a": 1.0, "b": 0.5, "c": 0.3}
    # Only "a" is within 90% of max (0.9), so "a" wins
    assert pick_best_candidate(scores) == "a"


def test_pick_best_tiebreak_without_sharpness():
    """Multiple candidates within TOP_FRACTION, no get_sharpest — highest score wins."""
    scores = {"a": 1.0, "b": 0.95}
    # Both are within 90% of 1.0 (threshold = 0.9)
    result = pick_best_candidate(scores)
    assert result == "a"


def test_pick_best_tiebreak_with_sharpness():
    """Multiple candidates within TOP_FRACTION — get_sharpest is called."""
    scores = {"a": 1.0, "b": 0.95}
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

    # "a" is the only one within 90% of 2.0 (threshold 1.8)
    scores = {"a": 2.0, "b": 1.0}
    result = pick_best_candidate(scores, get_sharpest=spy_sharpest)
    assert result == "a"
    assert called == []  # not called


# Need pytest for approx
import pytest
