"""Unit tests for app.ml.inference.observation_sort.order_indices.

Covers the pure-Python ordering layer that runs after the FAISS
similarity walk. The walk itself runs in the conda subprocess and
is not exercised here.
"""

import pytest

from app.ml.inference.observation_sort import VALID_SORTS, order_indices


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
    }


def test_empty_input_returns_empty():
    assert order_indices("similarity", [], []) == []
    assert order_indices("newest", [], []) == []
    assert order_indices("cls_low", [], []) == []
