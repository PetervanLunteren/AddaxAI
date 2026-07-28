"""
Pure-math tests for confusion matrix and classification report.

Exercises the helpers in app.api.crud.performance that don't touch the
DB: top-N folding, macro / weighted averages, F1 harmonic mean, class
ordering. Rank-resolution tests live in tests/ml/test_taxonomic_rank.py.
"""

from collections import Counter

import pytest

from app.api.crud.performance import (
    DETECTOR_CATEGORIES,
    OTHER_BUCKET,
    SEMANTIC_BUCKETS,
    _apply_top_n,
    _harmonic_mean,
    _macro,
    _ordered_classes,
    _weighted,
)
from app.ml.taxonomic_rank import HIGHER_LEVEL_TAXA, NO_TAXONOMY

# ---------------------------------------------------------------------------
# _ordered_classes
# ---------------------------------------------------------------------------


def test_ordered_classes_puts_detector_head_first() -> None:
    all_classes = {"wolf", "deer", "person", "animal", "vehicle"}
    totals = {"wolf": 10, "deer": 30, "person": 5, "animal": 2, "vehicle": 1}
    ordered = _ordered_classes(all_classes, totals)
    assert ordered[:3] == ["animal", "person", "vehicle"]
    assert ordered[3:] == ["deer", "wolf"]


def test_ordered_classes_omits_missing_detector_categories() -> None:
    all_classes = {"wolf", "deer"}
    totals = {"wolf": 3, "deer": 2}
    ordered = _ordered_classes(all_classes, totals)
    assert ordered == ["wolf", "deer"]
    for c in DETECTOR_CATEGORIES:
        assert c not in ordered


def test_ordered_classes_alphabetical_tiebreaker() -> None:
    all_classes = {"alpha", "beta", "gamma"}
    totals = {"alpha": 5, "beta": 5, "gamma": 5}
    ordered = _ordered_classes(all_classes, totals)
    assert ordered == ["alpha", "beta", "gamma"]


def test_ordered_classes_pins_semantic_buckets_to_bottom() -> None:
    all_classes = {"wolf", "deer", HIGHER_LEVEL_TAXA, NO_TAXONOMY, "animal"}
    totals = {
        "wolf": 3,
        "deer": 2,
        HIGHER_LEVEL_TAXA: 50,  # tons of support, but still pinned bottom
        NO_TAXONOMY: 10,
        "animal": 1,
    }
    ordered = _ordered_classes(all_classes, totals)
    assert ordered[0] == "animal"  # detector head first
    # Real species in the middle, sorted by support
    assert ordered[1:3] == ["wolf", "deer"]
    # Semantic buckets pinned to the bottom, in SEMANTIC_BUCKETS order
    assert ordered[-2:] == list(SEMANTIC_BUCKETS)


# ---------------------------------------------------------------------------
# _apply_top_n
# ---------------------------------------------------------------------------


def test_apply_top_n_noop_when_under_limit() -> None:
    ordered = ["a", "b", "c"]
    counts: Counter = Counter({("a", "a"): 3, ("b", "a"): 1})
    totals = {"a": 4, "b": 1, "c": 0}
    new_ordered, new_counts, other = _apply_top_n(ordered, counts, totals, 10)
    assert new_ordered == ordered
    assert new_counts == counts
    assert other is False


def test_apply_top_n_folds_tail_into_other() -> None:
    ordered = ["a", "b", "c", "d", "e"]
    counts: Counter = Counter(
        {
            ("a", "a"): 10,
            ("b", "b"): 8,
            ("c", "d"): 2,
            ("d", "c"): 3,
            ("e", "a"): 1,
        }
    )
    totals = {"a": 10, "b": 8, "c": 2, "d": 3, "e": 1}
    new_ordered, new_counts, other = _apply_top_n(ordered, counts, totals, 2)
    assert new_ordered == ["a", "b", OTHER_BUCKET]
    assert other is True
    assert sum(new_counts.values()) == sum(counts.values())
    assert new_counts[("a", "a")] == 10
    assert new_counts[(OTHER_BUCKET, "a")] == 1
    assert new_counts[(OTHER_BUCKET, OTHER_BUCKET)] == 5


def test_apply_top_n_exempts_detector_head_from_budget() -> None:
    # top_n counts only real species; detector categories never get
    # squeezed out no matter how small their support.
    ordered = ["animal", "person", "wolf", "deer", "bear"]
    counts: Counter = Counter(
        {
            ("wolf", "wolf"): 20,
            ("deer", "deer"): 15,
            ("bear", "bear"): 10,
            ("animal", "animal"): 2,
            ("person", "person"): 1,
        }
    )
    totals = {"wolf": 20, "deer": 15, "bear": 10, "animal": 2, "person": 1}
    # top_n=2 means keep 2 real classes; detector head is always present.
    new_ordered, _new_counts, other = _apply_top_n(ordered, counts, totals, 2)
    assert new_ordered[:2] == ["animal", "person"]
    assert "wolf" in new_ordered
    assert "deer" in new_ordered
    assert "bear" not in new_ordered
    assert OTHER_BUCKET in new_ordered
    assert other is True


def test_apply_top_n_exempts_semantic_buckets() -> None:
    ordered = ["wolf", "deer", "bear", HIGHER_LEVEL_TAXA, NO_TAXONOMY]
    counts: Counter = Counter(
        {
            ("wolf", "wolf"): 20,
            ("deer", "deer"): 15,
            ("bear", "bear"): 10,
            (HIGHER_LEVEL_TAXA, "wolf"): 5,
            (NO_TAXONOMY, NO_TAXONOMY): 1,
        }
    )
    totals = {"wolf": 20, "deer": 15, "bear": 10, HIGHER_LEVEL_TAXA: 5, NO_TAXONOMY: 1}
    new_ordered, _new_counts, other = _apply_top_n(ordered, counts, totals, 2)
    # Semantic buckets stay regardless of top-N
    assert HIGHER_LEVEL_TAXA in new_ordered
    assert NO_TAXONOMY in new_ordered
    # Bear falls into "other"
    assert "bear" not in new_ordered
    assert OTHER_BUCKET in new_ordered
    assert other is True


def test_apply_top_n_none_disables_collapse() -> None:
    ordered = ["a", "b", "c", "d"]
    counts: Counter = Counter({("a", "a"): 1, ("b", "b"): 1, ("c", "c"): 1, ("d", "d"): 1})
    totals = {"a": 1, "b": 1, "c": 1, "d": 1}
    new_ordered, new_counts, other = _apply_top_n(ordered, counts, totals, None)
    assert new_ordered == ordered
    assert new_counts == counts
    assert other is False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def test_harmonic_mean_typical() -> None:
    assert _harmonic_mean(0.6, 0.4) == pytest.approx(0.48)


def test_harmonic_mean_zero_sum_returns_none() -> None:
    assert _harmonic_mean(0.0, 0.0) is None


def test_harmonic_mean_none_input_returns_none() -> None:
    assert _harmonic_mean(None, 0.5) is None
    assert _harmonic_mean(0.5, None) is None


def test_macro_ignores_none() -> None:
    assert _macro([1.0, None, 0.5]) == pytest.approx(0.75)


def test_macro_all_none_returns_none() -> None:
    assert _macro([None, None]) is None


def test_weighted_support_weighted() -> None:
    assert _weighted([1.0, 0.0], [9, 1]) == pytest.approx(0.9)


def test_weighted_none_values_skipped() -> None:
    assert _weighted([None, 0.5], [10, 2]) == pytest.approx(0.5)


def test_weighted_all_none_returns_none() -> None:
    assert _weighted([None, None], [5, 5]) is None


def test_weighted_zero_support_returns_none() -> None:
    assert _weighted([0.5, 0.5], [0, 0]) is None
