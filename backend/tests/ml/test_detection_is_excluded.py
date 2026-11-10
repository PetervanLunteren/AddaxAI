"""The one label-set rule every postprocessing output filters through."""

from app.ml.postprocessing_outputs._label_filter import detection_is_excluded
from app.models import Detection


def _det(label: str | None, label_taxonomy_id: str | None) -> Detection:
    return Detection(label=label, label_taxonomy_id=label_taxonomy_id)


def test_matches_by_taxonomy_id():
    assert detection_is_excluded(_det("red fox", "tax-1"), {"tax-1"})


def test_matches_unmapped_label_by_raw_string():
    assert detection_is_excluded(_det("red fox", None), frozenset({"red fox"}))


def test_no_match_when_neither_field_is_in_the_set():
    assert not detection_is_excluded(_det("red fox", "tax-1"), {"tax-2", "badger"})


def test_empty_or_missing_set_matches_nothing():
    assert not detection_is_excluded(_det("red fox", "tax-1"), None)
    assert not detection_is_excluded(_det("red fox", "tax-1"), frozenset())


def test_missing_fields_never_match():
    assert not detection_is_excluded(_det(None, None), {"", "tax-1"})
