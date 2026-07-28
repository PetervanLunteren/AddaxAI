"""Unit tests for label exclusion and non-label skip logic."""

from app.ml.label_exclusion import (
    NON_LABEL_CLASSES,
    build_excluded_class_ids,
    build_non_label_class_ids,
    is_non_label_detection,
    should_skip_detection,
)

# ---------- NON_LABEL_CLASSES ----------

def test_non_label_classes_complete():
    """All expected non-label classes are present."""
    expected = {"bait", "blank", "empty", "false detection", "none", "vide"}
    assert NON_LABEL_CLASSES == expected


# ---------- build_non_label_class_ids ----------

def test_non_label_ids():
    """Returns IDs for NON_LABEL_CLASSES only."""
    cats = {"1": "lion", "2": "blank", "3": "Bait", "4": "zebra"}
    result = build_non_label_class_ids(cats)
    assert result == {"2", "3"}  # blank + Bait (case-insensitive)


def test_non_label_ids_empty_categories():
    """Empty categories returns empty set."""
    assert build_non_label_class_ids({}) == set()


def test_non_label_ids_no_matches():
    """No NON_LABEL classes in categories returns empty set."""
    cats = {"1": "lion", "2": "zebra"}
    assert build_non_label_class_ids(cats) == set()


# ---------- build_excluded_class_ids (used by postprocessing) ----------

def test_build_excluded_includes_both():
    """Includes both NON_LABEL and user exclusions."""
    cats = {"1": "lion", "2": "blank", "3": "zebra"}
    result = build_excluded_class_ids(cats, ["lion"])
    assert "1" in result  # user excluded
    assert "2" in result  # NON_LABEL


# ---------- should_skip_detection ----------

def test_skip_no_classifications():
    """Unclassified detection is not skipped."""
    det = {"category": "1", "conf": 0.9, "bbox": [0, 0, 0.5, 0.5]}
    assert should_skip_detection(det, {"1"}) is False


def test_skip_empty_classifications():
    """Empty classifications list is not skipped."""
    det = {"classifications": []}
    assert should_skip_detection(det, {"1"}) is False


def test_skip_blank_top1():
    """Blank as top-1: skip (false positive)."""
    det = {"classifications": [["2", 0.65], ["1", 0.19], ["3", 0.10]]}
    non_label = {"2"}  # blank
    assert should_skip_detection(det, non_label) is True


def test_skip_cattle_top1():
    """Real species as top-1: not skipped."""
    det = {"classifications": [["1", 0.92], ["2", 0.05], ["3", 0.03]]}
    non_label = {"2"}  # blank
    assert should_skip_detection(det, non_label) is False


# ---------- is_non_label_detection (legacy, kept for backward compat) ----------

def test_legacy_no_classifications():
    """Unclassified detection is not skipped."""
    det = {"category": "1", "conf": 0.9}
    excluded = build_excluded_class_ids({"1": "blank"})
    assert is_non_label_detection(det, excluded) is False


def test_legacy_all_excluded():
    """Detection with only non-label classifications is skipped."""
    det = {"classifications": [["1", 0.9], ["2", 0.1]]}
    excluded = build_excluded_class_ids({"1": "blank", "2": "empty"})
    assert is_non_label_detection(det, excluded) is True


def test_legacy_vide_excluded():
    """'vide' triggers skip."""
    assert "vide" in NON_LABEL_CLASSES
    det = {"classifications": [["1", 1.0]]}
    excluded = build_excluded_class_ids({"1": "vide"})
    assert is_non_label_detection(det, excluded) is True


# ---------- filter_classifications (no renormalization) ----------

def test_filter_classifications_no_renormalization():
    """Remaining confidences keep raw values after filtering."""
    from app.ml.label_exclusion import filter_classifications

    classifications = [["1", 0.65], ["2", 0.28], ["3", 0.07]]
    result = filter_classifications(classifications, {"2"})
    assert result == [["1", 0.65], ["3", 0.07]]


def test_filter_classifications_sorted_descending():
    """Result is sorted by confidence descending."""
    from app.ml.label_exclusion import filter_classifications

    classifications = [["1", 0.10], ["2", 0.50], ["3", 0.30]]
    result = filter_classifications(classifications, {"2"})
    assert result[0] == ["3", 0.30]
    assert result[1] == ["1", 0.10]


# ---------- apply_label_exclusion_to_results ----------

_FELIDAE = {
    "class": "mammalia", "order": "carnivora",
    "family": "felidae",
}
_TAXONOMY = {
    "lion": {**_FELIDAE, "genus": "panthera", "species": "leo"},
    "tiger": {**_FELIDAE, "genus": "panthera", "species": "tigris"},
    "bobcat": {**_FELIDAE, "genus": "lynx", "species": "rufus"},
    "fox": {
        "class": "mammalia", "order": "carnivora",
        "family": "canidae", "genus": "vulpes", "species": "vulpes",
    },
    "zebra": {
        "class": "mammalia", "order": "perissodactyla",
        "family": "equidae", "genus": "equus", "species": "quagga",
    },
}
_CATS = {
    "1": "lion", "2": "bobcat", "3": "fox",
    "4": "zebra", "5": "blank", "6": "tiger",
}


def test_apply_label_exclusion_noop_when_taxonomy():
    """With taxonomy, apply_label_exclusion_to_results is a no-op."""
    from app.ml.label_exclusion import apply_label_exclusion_to_results

    md_results = {
        "classification_categories": dict(_CATS),
        "images": [{
            "detections": [{
                "classifications": [
                    ["1", 0.60], ["5", 0.30], ["4", 0.10],
                ],
            }],
        }],
    }
    original_cls = [
        list(c)
        for c in md_results["images"][0]["detections"][0]["classifications"]
    ]
    apply_label_exclusion_to_results(
        md_results, excluded_labels=["lion"], taxonomy_lookup=_TAXONOMY
    )
    assert (
        md_results["images"][0]["detections"][0]["classifications"]
        == original_cls
    )


# ---------- strip_non_label_from_results ----------

def test_strip_non_label_from_results():
    """strip_non_label_from_results removes blank and bait."""
    from app.ml.label_exclusion import strip_non_label_from_results

    md_results = {
        "classification_categories": {
            "1": "lion", "2": "blank", "3": "bait",
        },
        "images": [{
            "detections": [{
                "classifications": [
                    ["1", 0.60], ["2", 0.30], ["3", 0.10],
                ],
            }],
        }],
    }
    strip_non_label_from_results(md_results)
    cls = md_results["images"][0]["detections"][0]["classifications"]
    assert len(cls) == 1
    assert cls[0][0] == "1"
