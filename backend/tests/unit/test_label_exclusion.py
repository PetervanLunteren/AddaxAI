"""Unit tests for label exclusion and the "nothing here" labels."""

from app.ml.label_exclusion import (
    NON_LABEL_CLASSES,
    build_excluded_class_ids,
    is_a_real_detection,
    is_non_label,
)
from app.models import Detection
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
)

# ---------- NON_LABEL_CLASSES ----------

def test_non_label_classes_complete():
    """All expected non-label classes are present."""
    expected = {
        "bait", "blank", "empty", "false detection", "non-animal", "none", "vide",
    }
    assert NON_LABEL_CLASSES == expected


# ---------- build_excluded_class_ids (used by postprocessing) ----------

def test_build_excluded_is_the_users_exclusions_only():
    """User exclusions, matched without regard to case, as the rollup
    matches the same exclusions. A non-label class is not excluded: it
    is the model's answer and stays in the list."""
    cats = {"1": "Lion", "2": "Blank", "3": "zebra"}
    result = build_excluded_class_ids(cats, ["lion"])
    assert result == {"1"}
    assert build_excluded_class_ids(cats, None) == set()


# ---------- filter_classifications (no renormalization) ----------

def test_filter_classifications_no_renormalization():
    """Remaining confidences keep raw values after filtering."""
    from app.ml.label_exclusion import filter_classifications

    classifications = [["1", 0.65], ["2", 0.28], ["3", 0.07]]
    result = filter_classifications(classifications, {"2"})
    assert result == [["1", 0.65], ["3", 0.07]]


def test_filter_classifications_empties_below_the_scale_minimum():
    """A 99% excluded class leaves 0.3% behind: below what any slider can
    show, so the box is unclassified rather than labelled with a guess."""
    from app.ml.label_exclusion import filter_classifications

    assert filter_classifications([["1", 0.99], ["2", 0.003]], {"1"}) == []
    assert filter_classifications([["1", 0.9], ["2", 0.01]], {"1"}) == [["2", 0.01]]


def test_filter_classifications_sorted_descending():
    """Result is sorted by confidence descending."""
    from app.ml.label_exclusion import filter_classifications

    classifications = [["1", 0.10], ["2", 0.50], ["3", 0.30]]
    result = filter_classifications(classifications, {"2"})
    assert result[0] == ["3", 0.30]
    assert result[1] == ["1", 0.10]


# ---------- apply_label_exclusion_to_results ----------

_CATS = {
    "1": "lion", "2": "bobcat", "3": "fox",
    "4": "zebra", "5": "blank", "6": "tiger",
}


def _one_detection_results() -> dict:
    return {
        "classification_categories": dict(_CATS),
        "images": [{
            "detections": [{
                "classifications": [
                    ["1", 0.60], ["5", 0.30], ["4", 0.10],
                ],
            }],
        }],
    }


def test_apply_label_exclusion_noop_when_rollup_handles_it():
    """When rollup owns exclusion the lists stay untouched for Path A."""
    from app.ml.label_exclusion import apply_label_exclusion_to_results

    md_results = _one_detection_results()
    original_cls = [
        list(c)
        for c in md_results["images"][0]["detections"][0]["classifications"]
    ]
    apply_label_exclusion_to_results(
        md_results, excluded_labels=["lion"], rollup_handles_exclusion=True
    )
    assert (
        md_results["images"][0]["detections"][0]["classifications"]
        == original_cls
    )


def test_apply_label_exclusion_drops_excluded_and_keeps_next_best_score():
    """Without rollup the excluded top-1 goes and the next best class
    leads at its own score, not inflated. The non-label class "blank"
    is not an exclusion, so here it is what leads."""
    from app.ml.label_exclusion import apply_label_exclusion_to_results

    md_results = _one_detection_results()
    apply_label_exclusion_to_results(md_results, excluded_labels=["lion"])
    assert (
        md_results["images"][0]["detections"][0]["classifications"]
        == [["5", 0.30], ["4", 0.10]]
    )


# ---------- the two lanes of the read-time rule ----------


def test_the_sql_and_python_lanes_agree(db):
    """`is_a_real_detection` answers for a query, `is_non_label` for a
    label already in hand. Having both is only safe while they cannot
    disagree, which is what this pins, the same way
    `tests/ml/test_detection_visibility.py` pins its own pair."""
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=deployment.id)

    labels = [*sorted(NON_LABEL_CLASSES), "FALSE DETECTION", "deer", None, ""]
    for label in labels:
        make_detection(db, file_id=f.id, label=label, confidence=0.9)
    db.commit()

    kept = {
        d.label
        for d in db.query(Detection)
        .filter(Detection.file_id == f.id)
        .filter(is_a_real_detection())
        .all()
    }
    for label in labels:
        assert (label in kept) == (not is_non_label(label)), repr(label)
