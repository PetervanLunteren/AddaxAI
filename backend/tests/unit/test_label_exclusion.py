"""Unit tests for label exclusion and non-label skip logic."""

from app.ml.label_exclusion import (
    NON_LABEL_CLASSES,
    build_excluded_class_ids,
    build_non_label_class_ids,
    build_user_excluded_class_ids,
    filter_and_rollup_classifications,
    is_non_label_detection,
    should_skip_detection,
)

# ---------- NON_LABEL_CLASSES ----------

def test_non_label_classes_complete():
    """All expected non-label classes are present."""
    expected = {"bait", "blank", "empty", "false detection", "none", "vide"}
    assert NON_LABEL_CLASSES == expected


# ---------- build_user_excluded_class_ids ----------

def test_user_excluded_no_labels():
    """No user exclusions returns empty set."""
    cats = {"1": "lion", "2": "blank"}
    assert build_user_excluded_class_ids(cats) == set()
    assert build_user_excluded_class_ids(cats, []) == set()


def test_user_excluded_specific_labels():
    """Returns IDs for user-excluded labels only, not NON_LABEL."""
    cats = {"1": "lion", "2": "blank", "3": "zebra"}
    result = build_user_excluded_class_ids(cats, ["lion"])
    assert result == {"1"}


def test_user_excluded_does_not_include_non_label():
    """User exclusion function never includes NON_LABEL classes."""
    cats = {"1": "lion", "2": "blank", "3": "bait"}
    result = build_user_excluded_class_ids(cats, ["lion"])
    assert "2" not in result  # blank
    assert "3" not in result  # bait
    assert result == {"1"}


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


# ---------- build_excluded_class_ids (legacy, still used by postprocessing) ----------

def test_build_excluded_includes_both():
    """Legacy function includes both NON_LABEL and user exclusions."""
    cats = {"1": "lion", "2": "blank", "3": "zebra"}
    result = build_excluded_class_ids(cats, ["lion"])
    assert "1" in result  # user excluded
    assert "2" in result  # NON_LABEL


# ---------- should_skip_detection ----------

def test_skip_no_classifications():
    """Unclassified detection is not skipped."""
    det = {"category": "1", "conf": 0.9, "bbox": [0, 0, 0.5, 0.5]}
    assert should_skip_detection(det, set(), {"1"}) is False


def test_skip_empty_classifications():
    """Empty classifications list is not skipped."""
    det = {"classifications": []}
    assert should_skip_detection(det, set(), {"1"}) is False


def test_skip_blank_top1_no_user_exclusions():
    """Blank as top-1, no user exclusions: skip (false positive)."""
    det = {"classifications": [["2", 0.65], ["1", 0.19], ["3", 0.10]]}
    non_label = {"2"}  # blank
    assert should_skip_detection(det, set(), non_label) is True


def test_skip_blank_top1_user_excludes_blank():
    """Blank as top-1, user excludes blank: blank removed, cattle becomes
    top-1, cattle is not NON_LABEL, so not skipped."""
    det = {"classifications": [["2", 0.65], ["1", 0.19], ["3", 0.10]]}
    user_excluded = {"2"}  # user excluded blank
    non_label = {"2"}  # blank is also NON_LABEL
    assert should_skip_detection(det, user_excluded, non_label) is False


def test_skip_cattle_top1():
    """Real species as top-1: not skipped."""
    det = {"classifications": [["1", 0.92], ["2", 0.05], ["3", 0.03]]}
    non_label = {"2"}  # blank
    assert should_skip_detection(det, set(), non_label) is False


def test_skip_all_user_excluded():
    """All classifications removed by user exclusion: skip."""
    det = {"classifications": [["1", 0.6], ["2", 0.4]]}
    user_excluded = {"1", "2"}
    assert should_skip_detection(det, user_excluded, set()) is True


def test_skip_only_non_labels_remain():
    """User excludes real species, only blank remains: skip."""
    det = {"classifications": [["1", 0.6], ["2", 0.3], ["3", 0.1]]}
    user_excluded = {"1"}  # exclude the real species
    non_label = {"2", "3"}  # blank + bait
    # After user filter: [["2", 0.3], ["3", 0.1]] → top-1 is blank → skip
    assert should_skip_detection(det, user_excluded, non_label) is True


# ---------- filter_and_rollup_classifications ----------

# Shared test fixtures
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

_CATS = {"1": "lion", "2": "bobcat", "3": "fox", "4": "zebra", "5": "blank", "6": "tiger"}
_ID_TO_NAME = {k: v for k, v in _CATS.items()}


def test_rollup_excluded_to_genus():
    """Lion excluded redirects score to panthera genus (nearest ancestor)."""
    classifications = [["1", 0.90], ["2", 0.04], ["3", 0.03], ["4", 0.03]]
    excluded = {"1"}  # lion
    cats = dict(_CATS)
    rollup = filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats
    )
    top = rollup.classifications[0]
    top_label = cats.get(str(top[0]))
    # Lion rolls up to genus "panthera" (nearest non-excluded ancestor)
    assert top_label == "panthera"
    assert top[1] > 0.85


def test_rollup_multiple_excluded_same_genus():
    """Lion + tiger both excluded accumulate at panthera genus."""
    classifications = [["1", 0.50], ["6", 0.30], ["2", 0.10], ["4", 0.10]]
    excluded = {"1", "6"}  # lion + tiger (both genus panthera)
    cats = dict(_CATS)
    rollup = filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats
    )
    top = rollup.classifications[0]
    top_label = cats.get(str(top[0]))
    # Both roll up to panthera genus
    assert top_label == "panthera"
    assert top[1] >= 0.80  # lion 50% + tiger 30%


def test_rollup_non_excluded_unaffected():
    """Non-excluded species keep their original scores."""
    classifications = [["4", 0.60], ["1", 0.30], ["3", 0.10]]
    excluded = {"1"}  # lion
    cats = dict(_CATS)
    rollup = filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats
    )
    # Zebra should still be present with its original score
    zebra_entry = [
        r for r in rollup.classifications
        if cats.get(str(r[0])) == "zebra"
    ]
    assert len(zebra_entry) == 1
    assert zebra_entry[0][1] == 0.60


def test_rollup_no_taxonomy_drops_score():
    """Excluded class with no taxonomy entry: score is dropped."""
    classifications = [["5", 0.90], ["4", 0.10]]
    excluded = {"5"}  # blank (not in taxonomy)
    cats = dict(_CATS)
    rollup = filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats
    )
    # Blank has no taxonomy, so its 90% is dropped. Zebra remains.
    assert len(rollup.classifications) >= 1
    top_label = cats.get(str(rollup.classifications[0][0]))
    assert top_label == "zebra"


def test_rollup_all_ancestors_excluded_uses_broadest():
    """When all ancestors are excluded, use broadest available."""
    classifications = [["1", 0.90], ["4", 0.10]]
    excluded = {"1"}  # only lion is a model class
    cats = dict(_CATS)
    # Lion walks up: genus=panthera (not excluded) -> stops there.
    rollup = filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats
    )
    top_label = cats.get(str(rollup.classifications[0][0]))
    assert top_label == "panthera"


def test_rollup_empty_excluded():
    """No exclusions returns classifications unchanged."""
    classifications = [["1", 0.90], ["4", 0.10]]
    cats = dict(_CATS)
    rollup = filter_and_rollup_classifications(
        classifications, set(), _ID_TO_NAME, _TAXONOMY, cats
    )
    assert rollup.classifications == classifications
    assert rollup.new_entries == []


def test_rollup_creates_new_class_id():
    """Rollup creates a new class ID in classification_categories."""
    classifications = [["1", 0.90], ["4", 0.10]]
    excluded = {"1"}  # lion
    cats = dict(_CATS)
    original_len = len(cats)
    filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats
    )
    # "panthera" should have been added to cats
    assert len(cats) > original_len
    panthera_ids = [k for k, v in cats.items() if v == "panthera"]
    assert len(panthera_ids) == 1


def test_rollup_returns_new_entries():
    """Rollup returns new_entries with name and level for persistence."""
    classifications = [["1", 0.90], ["4", 0.10]]
    excluded = {"1"}  # lion
    cats = dict(_CATS)
    rollup = filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats
    )
    assert len(rollup.new_entries) == 1
    entry = rollup.new_entries[0]
    assert entry["name"] == "panthera"
    assert entry["level"] == "genus"


# ---------- is_non_label_detection (legacy, kept for backward compat) ----------

def test_legacy_no_classifications():
    """Legacy: unclassified detection is not skipped."""
    det = {"category": "1", "conf": 0.9}
    excluded = build_excluded_class_ids({"1": "blank"})
    assert is_non_label_detection(det, excluded) is False


def test_legacy_all_excluded():
    """Legacy: detection with only non-label classifications is skipped."""
    det = {"classifications": [["1", 0.9], ["2", 0.1]]}
    excluded = build_excluded_class_ids({"1": "blank", "2": "empty"})
    assert is_non_label_detection(det, excluded) is True


def test_legacy_vide_excluded():
    """Legacy: 'vide' triggers skip."""
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


# ---------- exclusion rollup descriptions ----------

def test_rollup_adds_descriptions_for_new_ancestors():
    """New ancestor categories get 7-token descriptions."""
    classifications = [["1", 0.90], ["4", 0.10]]
    excluded = {"1"}  # lion
    cats = dict(_CATS)
    descriptions: dict[str, str] = {}
    filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats,
        classification_category_descriptions=descriptions,
    )
    # A description should have been added for the new ancestor
    assert len(descriptions) == 1
    desc = next(iter(descriptions.values()))
    # 7-token format: name;class;order;family;genus;species;name
    assert len(desc.split(";")) == 7


def test_rollup_no_crash_when_descriptions_none():
    """No crash when classification_category_descriptions is None."""
    classifications = [["1", 0.90], ["4", 0.10]]
    excluded = {"1"}  # lion
    cats = dict(_CATS)
    # Should not raise
    filter_and_rollup_classifications(
        classifications, excluded, _ID_TO_NAME, _TAXONOMY, cats,
        classification_category_descriptions=None,
    )


# ---------- apply_label_exclusion_to_results (NON_LABEL kept) ----------

def test_apply_label_exclusion_noop_when_taxonomy():
    """With taxonomy, apply_label_exclusion_to_results is a no-op."""
    from app.ml.label_exclusion import apply_label_exclusion_to_results

    md_results = {
        "classification_categories": dict(_CATS),
        "images": [{
            "detections": [{
                "classifications": [["1", 0.60], ["5", 0.30], ["4", 0.10]],
            }],
        }],
    }
    original_cls = [list(c) for c in md_results["images"][0]["detections"][0]["classifications"]]
    apply_label_exclusion_to_results(
        md_results, excluded_labels=["lion"], taxonomy_lookup=_TAXONOMY
    )
    # All classifications unchanged (including excluded lion and blank)
    assert md_results["images"][0]["detections"][0]["classifications"] == original_cls


# ---------- strip_non_label_from_results ----------

def test_strip_non_label_from_results():
    """strip_non_label_from_results removes blank and bait."""
    from app.ml.label_exclusion import strip_non_label_from_results

    md_results = {
        "classification_categories": {"1": "lion", "2": "blank", "3": "bait"},
        "images": [{
            "detections": [{
                "classifications": [["1", 0.60], ["2", 0.30], ["3", 0.10]],
            }],
        }],
    }
    strip_non_label_from_results(md_results)
    cls = md_results["images"][0]["detections"][0]["classifications"]
    assert len(cls) == 1
    assert cls[0][0] == "1"
