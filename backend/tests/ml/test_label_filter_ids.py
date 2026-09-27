from app.ml.label_filter_ids import (
    decode_unmapped_label,
    encode_unmapped_label,
    label_matches_filter,
    parse_label_filter_ids,
)


def test_unmapped_label_ids_are_stable_comma_safe_and_unicode_safe():
    label = "fox, red"
    token = encode_unmapped_label(label)

    assert "," not in token
    assert decode_unmapped_label(token) == label
    assert parse_label_filter_ids([token]) == ([], [], [label])
    assert decode_unmapped_label("unmapped:not-valid!") is None


def test_encoded_unmapped_label_does_not_match_same_named_taxonomy_label():
    token = encode_unmapped_label("fox")

    assert label_matches_filter("fox", None, {token})
    assert not label_matches_filter("fox", "taxonomy-uuid", {token})
    # Plain labels from older folder-run saves retain their legacy behavior.
    assert label_matches_filter("fox", "taxonomy-uuid", {"fox"})


def test_encoded_unmapped_label_uses_category_when_label_is_missing():
    token = encode_unmapped_label("red fox")

    assert label_matches_filter(None, None, {token}, category="red fox")
    assert not label_matches_filter(None, "taxonomy-uuid", {token}, category="red fox")
