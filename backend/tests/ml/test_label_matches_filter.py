"""The one label-set rule every postprocessing output filters through."""

from app.ml.postprocessing_outputs._label_filter import label_matches_filter


def test_matches_by_taxonomy_id():
    assert label_matches_filter("red fox", "tax-1", {"tax-1"})


def test_matches_unmapped_label_by_raw_string():
    assert label_matches_filter("red fox", None, frozenset({"red fox"}))


def test_no_match_when_neither_field_is_in_the_set():
    assert not label_matches_filter("red fox", "tax-1", {"tax-2", "badger"})


def test_empty_or_missing_set_matches_nothing():
    assert not label_matches_filter("red fox", "tax-1", None)
    assert not label_matches_filter("red fox", "tax-1", frozenset())


def test_missing_fields_never_match():
    assert not label_matches_filter(None, None, {"", "tax-1"})
