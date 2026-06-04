"""Unit tests for app.ml.inference.observation_sort.order_indices.

Covers the pure-Python ordering layer that runs after the FAISS
similarity walk. The walk itself runs in the conda subprocess and
is not exercised here.
"""

import pytest

from app.ml.inference.observation_sort import (
    VALID_SORTS,
    order_indices,
    suggestions_order,
)


def _metas(*items: dict) -> list[dict]:
    """Build a metadata list with sane defaults so tests stay readable."""
    return [
        {
            "captured_at_local": item.get("captured_at_local"),
            "label_confidence": item.get("label_confidence"),
        }
        for item in items
    ]


def test_similarity_returns_walk_order_unchanged():
    metas = _metas({}, {}, {})
    assert order_indices("similarity", [2, 0, 1], metas) == [2, 0, 1]


def test_similarity_reverse_reverses_walk_order():
    metas = _metas({}, {}, {})
    assert order_indices("similarity_reverse", [2, 0, 1], metas) == [1, 0, 2]


def test_newest_orders_by_captured_at_descending():
    metas = _metas(
        {"captured_at_local": "2026-01-01T08:00:00"},
        {"captured_at_local": "2026-03-15T14:30:00"},
        {"captured_at_local": "2026-02-10T20:00:00"},
    )
    assert order_indices("newest", [0, 1, 2], metas) == [1, 2, 0]


def test_oldest_orders_by_captured_at_ascending():
    metas = _metas(
        {"captured_at_local": "2026-01-01T08:00:00"},
        {"captured_at_local": "2026-03-15T14:30:00"},
        {"captured_at_local": "2026-02-10T20:00:00"},
    )
    assert order_indices("oldest", [0, 1, 2], metas) == [0, 2, 1]


def test_newest_pushes_null_timestamps_to_end():
    metas = _metas(
        {"captured_at_local": None},
        {"captured_at_local": "2026-03-15T14:30:00"},
        {"captured_at_local": "2026-01-01T08:00:00"},
        {"captured_at_local": None},
    )
    result = order_indices("newest", [0, 1, 2, 3], metas)
    # Non-null first (newest → oldest), then nulls in original order.
    assert result == [1, 2, 0, 3]


def test_oldest_pushes_null_timestamps_to_end():
    metas = _metas(
        {"captured_at_local": None},
        {"captured_at_local": "2026-03-15T14:30:00"},
        {"captured_at_local": "2026-01-01T08:00:00"},
        {"captured_at_local": None},
    )
    result = order_indices("oldest", [0, 1, 2, 3], metas)
    assert result == [2, 1, 0, 3]


def test_cls_low_orders_by_label_confidence_ascending():
    metas = _metas(
        {"label_confidence": 0.92},
        {"label_confidence": 0.31},
        {"label_confidence": 0.65},
    )
    assert order_indices("cls_low", [0, 1, 2], metas) == [1, 2, 0]


def test_cls_low_pushes_null_label_confidence_to_end():
    metas = _metas(
        {"label_confidence": None},
        {"label_confidence": 0.5},
        {"label_confidence": None},
        {"label_confidence": 0.1},
    )
    result = order_indices("cls_low", [0, 1, 2, 3], metas)
    assert result == [3, 1, 0, 2]


def test_unknown_sort_mode_raises():
    with pytest.raises(ValueError):
        order_indices("bogus", [0], _metas({}))


def test_valid_sorts_lists_every_supported_mode():
    assert VALID_SORTS == {
        "similarity",
        "similarity_reverse",
        "newest",
        "oldest",
        "cls_low",
        "suggestions",
    }


def test_suggestions_routes_through_do_sort_not_order_indices():
    """suggestions needs neighbour signals that order_indices does not have."""
    with pytest.raises(ValueError, match="handled by do_sort"):
        order_indices("suggestions", [0], _metas({}))


def test_empty_input_returns_empty():
    assert order_indices("similarity", [], []) == []
    assert order_indices("newest", [], []) == []
    assert order_indices("cls_low", [], []) == []


# ── suggestions_order ────────────────────────────────────────────────────


def _sug_meta(label, *, category="animal", verified=False, suggestion_dismissed=False):
    return {
        "label": label,
        "category": category,
        "verified": verified,
        "suggestion_dismissed": suggestion_dismissed,
    }


def test_suggestions_order_groups_by_triple_and_sorts_by_count_desc():
    metas = [
        _sug_meta("aves"),
        _sug_meta("aves"),
        _sug_meta("aves"),
        _sug_meta("canis"),
        _sug_meta("canis"),
    ]
    top_labels = [
        "american crow",
        "american crow",
        "american crow",
        "domestic dog",
        "domestic dog",
    ]
    agreement = [0.1, 0.2, 0.3, 0.1, 0.2]
    # min_count=1 to keep both cohorts.
    result = suggestions_order(metas, top_labels, agreement, min_count=1)
    # Aves cohort (3) comes before canis cohort (2). Within each cohort,
    # ascending agreement.
    assert result == [0, 1, 2, 3, 4]


def test_suggestions_order_drops_small_cohorts_under_min_count():
    metas = [_sug_meta("aves")] * 5 + [_sug_meta("canis")] * 2
    top_labels = ["american crow"] * 5 + ["domestic dog"] * 2
    agreement = [0.0] * 7
    result = suggestions_order(metas, top_labels, agreement, min_count=3)
    # canis cohort (2 members) drops; aves (5) survives. All five aves
    # are present, no canis indices.
    assert sorted(result) == [0, 1, 2, 3, 4]


def test_suggestions_order_caps_at_max_cohorts():
    # Five cohorts of 2 each.
    metas = []
    top_labels = []
    for cohort in range(5):
        for _ in range(2):
            metas.append(_sug_meta(f"label{cohort}"))
            top_labels.append(f"sug{cohort}")
    agreement = [0.0] * 10
    result = suggestions_order(metas, top_labels, agreement, min_count=1, max_cohorts=2)
    # Only members from the two largest cohorts; ties break
    # deterministically on key so the result is stable.
    assert len(result) == 4


def test_suggestions_order_excludes_verified_and_unsuggested():
    metas = [
        _sug_meta("aves", verified=True),  # verified → out
        _sug_meta("aves"),                  # in
        _sug_meta("aves"),                  # no suggestion → out
    ]
    top_labels = ["american crow", "american crow", None]
    agreement = [0.0, 0.0, 0.0]
    result = suggestions_order(metas, top_labels, agreement, min_count=1)
    assert result == [1]


def test_suggestions_order_excludes_dismissed():
    # A dismissed crop is skipped as a cohort member, exactly like a
    # verified one. The remaining members still form the cohort.
    metas = [
        _sug_meta("aves", suggestion_dismissed=True),  # dismissed → out
        _sug_meta("aves"),                              # in
        _sug_meta("aves"),                              # in
    ]
    top_labels = ["american crow", "american crow", "american crow"]
    agreement = [0.0, 0.1, 0.2]
    result = suggestions_order(metas, top_labels, agreement, min_count=1)
    assert result == [1, 2]


def test_suggestions_order_within_cohort_ascending_agreement():
    metas = [_sug_meta("aves")] * 3
    top_labels = ["american crow"] * 3
    agreement = [0.6, 0.1, 0.3]
    result = suggestions_order(metas, top_labels, agreement, min_count=1)
    assert result == [1, 2, 0]  # 0.1, 0.3, 0.6


def test_suggestions_order_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        suggestions_order([], [], [], min_count=0)
    with pytest.raises(ValueError):
        suggestions_order([], [], [], max_cohorts=0)


def test_suggestions_order_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        suggestions_order([{}], ["a", "b"], [0.0])
