"""Unit tests for is_non_label_detection() in label_exclusion.py."""

from app.ml.label_exclusion import (
    NON_LABEL_CLASSES,
    build_excluded_class_ids,
    is_non_label_detection,
)


def _make_excluded(categories: dict[str, str]) -> set[str]:
    """Build excluded_class_ids from a classification_categories dict."""
    return build_excluded_class_ids(categories)


def test_no_classifications_key():
    """Detection without classifications key is not skipped (unclassified)."""
    det = {"category": "1", "conf": 0.9, "bbox": [0, 0, 0.5, 0.5]}
    excluded = _make_excluded({"1": "blank"})
    assert is_non_label_detection(det, excluded) is False


def test_empty_classifications_list():
    """Detection with empty classifications list is not skipped."""
    det = {"classifications": []}
    excluded = _make_excluded({"1": "blank"})
    assert is_non_label_detection(det, excluded) is False


def test_all_excluded():
    """Detection with only non-label classifications is skipped."""
    det = {"classifications": [["1", 0.9], ["2", 0.1]]}
    excluded = _make_excluded({"1": "blank", "2": "empty"})
    assert is_non_label_detection(det, excluded) is True


def test_some_remain_after_filtering():
    """Detection with mixed classifications (non-label + real) is not skipped."""
    det = {"classifications": [["1", 0.6], ["2", 0.4]]}
    excluded = _make_excluded({"1": "blank", "2": "lion"})
    assert is_non_label_detection(det, excluded) is False


def test_empty_excluded_set():
    """Empty excluded set never skips anything."""
    det = {"classifications": [["1", 1.0]]}
    assert is_non_label_detection(det, set()) is False


def test_vide_excluded():
    """'vide' (French for empty) is in NON_LABEL_CLASSES and triggers skip."""
    assert "vide" in NON_LABEL_CLASSES
    det = {"classifications": [["1", 1.0]]}
    excluded = _make_excluded({"1": "vide"})
    assert is_non_label_detection(det, excluded) is True


def test_non_label_classes_complete():
    """All expected non-label classes are present."""
    expected = {"bait", "blank", "empty", "false detection", "none", "vide"}
    assert NON_LABEL_CLASSES == expected
