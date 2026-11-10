"""
Label exclusion and the "nothing here" labels.

Two separate concerns handled here:

1. **User exclusion** (filter): labels the user marked as not present
   in their project area are removed from the classification list.
   Remaining confidences keep their raw values (no renormalization).
   This changes which label is assigned to a detection.

2. **Non-label classes**: labels that mean "nothing is here" (blank,
   false detection, bait, ...). A box carrying one is stored like any
   other, whether the model wrote it at ingest or a person wrote it by
   pressing X. ``is_a_real_detection()`` is the one predicate that keeps
   such a box out of every count, export and media output, and the
   Labels grid still shows it as a card to confirm or rescue.

These two steps are independent. JSON files on disk remain untouched
as raw ground truth.
"""

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement

from app.core.confidence import CONFIDENCE_SCALE_MIN
from app.core.logging_config import get_logger
from app.models import Detection

logger = get_logger(__name__)


# Non-label classes: labels that mean "nothing here" or "false positive".
# Two writers: a classification model whose class list has one (fifteen
# zoo models do), written at ingest with the model's own score, and a
# person pressing X on the Labels page. Both produce the same row. Add new
# junk classes here (e.g. "calibration", "setup") — and in the two hand
# copies: `NON_LABEL_CLASSES` in `frontend/src/lib/detection-utils.ts` and
# the SQL list in `app/ml/inference/similarity_script.py`.
NON_LABEL_CLASSES = frozenset({
    "bait", "blank", "empty", "false detection", "non-animal", "none", "vide",
})

def is_a_real_detection(
    column: InstrumentedAttribute[str | None] = Detection.label,
) -> ColumnElement[bool]:
    """Predicate: this detection is not one of the "nothing here" labels.

    A rejected box stays in the database, whoever rejected it: the model
    at ingest, or a person pressing X later. That verdict has to reach
    every surface that asks "what is on this file", and three of them
    ask in three different ways, so the rule lives here once rather than
    in each. The Labels grid deliberately does not apply it: a rejection
    is a card like any other there, so a person can confirm or overturn
    it.

    Two lanes, the same shape as ``ml/detection_visibility.py``: this one
    for a query, ``is_non_label`` below for a label already in memory. A
    parity test pins that the two agree.

    ``column`` is the label it reads: the current one by default, or
    ``Detection.original_label`` for the AI's own call (Calibration).
    """
    return or_(
        column.is_(None),
        func.lower(column).notin_(sorted(NON_LABEL_CLASSES)),
    )


def is_non_label(label: str | None) -> bool:
    """The same rule for a label already in hand. ``None`` is a real
    detection the classifier simply never named, not a rejection."""
    return bool(label) and label.lower() in NON_LABEL_CLASSES


def is_human_decided(detection) -> bool:
    """A label postprocessing must never overwrite.

    Two cases. A verified box: a person judged it, on a photo, on a card
    or on a frame they opened. A box whose ``classification_method`` is
    "track": it inherited its label from the card of the animal it
    belongs to, and is deliberately *not* verified, because the person
    never saw that frame; "verified" has to keep meaning "a person saw
    this box", which a training set built on verified rows depends on.
    Both are the person's call, and a reprocess that rewrote the second
    would re-fragment a verified animal's species frame by frame.

    Not "human" on its own: every path that writes it also verifies, and
    an unticked file must hand its boxes back to the machine.
    """
    return bool(detection.verified) or detection.classification_method == "track"


def threshold_or_verified(threshold) -> ColumnElement[bool]:
    """The user-facing scope rule (DEVELOPERS.md), in one place.

    A detection passes on its own confidence, or because a human
    verified it **as something real**. The refinement is the second
    half: verifying a file rejects its invisible sub-threshold boxes by
    marking them "false detection" (``set_file_verified``), and those
    rows must stay invisible at every threshold, in every list, count,
    export and media output. Without it they are verified, so the plain
    threshold-or-verified rule surfaces thousands of them the moment
    someone verifies a project (a real project measured 4,050 weak boxes
    against 7,526 passing ones).

    An above-threshold rejected box still passes: the person pressed X
    on it in the grid, so the grid and the exports keep showing that
    verdict rather than making the box vanish.

    ``threshold`` is a float, or a column expression where the query
    compares per row (the folder-run summaries join ``Project``).
    The similarity sort worker keeps a hand-written SQL copy of this
    (``similarity_script.py``, no ``app.*`` imports there) — keep the
    two in step.
    """
    return or_(
        Detection.confidence >= threshold,
        and_(
            Detection.verified == True,  # noqa: E712
            is_a_real_detection(),
        ),
    )


def classification_score_in_range(
    min_score: float | None, max_score: float | None
) -> ColumnElement[bool]:
    """The classification confidence range, in one place.

    It is a filter on the classifier's score, so it applies to classified
    boxes only: a box the classifier never scored has nothing to compare
    and passes, at both ends. In SQL a NULL fails every comparison, so
    the plain ``label_confidence >= min`` hid every unclassified box the
    moment the slider left its floor, with nothing on screen saying so.
    A user with 733 unclassified moose could reach them only with the
    handle parked exactly on the floor (2026-09-06). Those boxes are the
    ones a person most needs to find and label by hand; whoever wants
    them out has the "Animal" leaf in the label filter for that.

    Call it only when at least one bound is set. The similarity sort
    worker keeps a hand-written SQL copy (``similarity_script.py``, no
    ``app.*`` imports there) — keep the two in step.
    """
    bounds = []
    if min_score is not None:
        bounds.append(Detection.label_confidence >= min_score)
    if max_score is not None:
        bounds.append(Detection.label_confidence <= max_score)
    return or_(Detection.label_confidence.is_(None), and_(*bounds))


# Non-wildlife classes: real detections that are not wild animals.
# Superset of NON_LABEL_CLASSES, adding every human and vehicle class
# name found across the model zoo. Used by wildlife-only statistics
# (the dashboard "Wildlife detected" chart). Matched case-insensitively
# against Detection/EventObservation labels. When a new model ships a
# class like "person" or "car", add it here.
NON_WILDLIFE_CLASSES = NON_LABEL_CLASSES | frozenset({
    "human", "homo_sapiens", "vehicle",
})

# The detector categories that are not wildlife. A species label on a
# box of any other category ("animal", "fish", "elasmobranch") leaves the
# category as the detector said it; on one of these it moves the box to
# the wildlife category the picker sent. Categories, not labels: see
# NON_WILDLIFE_CLASSES for the label side.
NON_WILDLIFE_CATEGORIES = frozenset({"person", "vehicle"})


def is_wildlife(category: str | None) -> bool:
    """Is this detector category a wild animal?

    The one rule, in one place: **person and vehicle are not wildlife,
    every other category is.** AddaxAI was written when a detector said
    only "animal", "person" or "vehicle", so a great many places asked
    ``category == "animal"`` when they meant this. SharkTrack says
    "elasmobranch" and the Community Fish Detector says "fish", and under
    the old test a shark was neither wildlife nor a person: it fell out
    of the deployment's animal count, out of the dashboard's species
    bars, and out of the species name in two exports.

    Deliberately a closed list of what is *not* wildlife rather than a
    list of what is: a detector we have never seen ships a class we
    cannot enumerate, and counting it as wildlife is the harmless
    direction. If one ever emits something that is neither (a "boat"),
    it is one more entry in ``NON_WILDLIFE_CATEGORIES``.

    ``None`` is not wildlife: it is the absence of a detection, not an
    animal.
    """
    return category is not None and category not in NON_WILDLIFE_CATEGORIES


def is_wildlife_category(column) -> ColumnElement[bool]:
    """``is_wildlife`` for a query, over ``Detection.category`` or
    ``EventObservation.category``.

    Two lanes over one rule, the same shape as ``is_a_real_detection`` /
    ``is_non_label`` above and as ``ml/detection_visibility.py``. A
    parity test pins that the two agree, which is what makes having two
    of them safe. Negate it (``~is_wildlife_category(...)``) for the
    "this box has no species, show its category" branches.
    """
    return and_(column.isnot(None), column.notin_(sorted(NON_WILDLIFE_CATEGORIES)))


def filter_classifications(
    classifications: list[list],
    excluded_class_ids: set[str],
) -> list[list]:
    """
    Remove excluded labels from classifications.

    Remaining confidences keep their raw values (no renormalization).
    A list whose best remaining class scores below ``CONFIDENCE_SCALE_MIN``
    empties: v6 renormalised such a leftover to 100%, which dressed a
    guess as a certainty, and keeping it raw hands out labels at 0.3%
    that no slider can reach. Unclassified is the honest answer there.

    Args:
        classifications: List of [class_id, confidence] pairs
        excluded_class_ids: Set of class IDs to exclude

    Returns:
        New list of [class_id, confidence] sorted by confidence descending,
        with excluded labels removed. Returns empty list if no labels remain.
    """
    if not classifications or not excluded_class_ids:
        return classifications

    remaining = [
        [cls_id, conf]
        for cls_id, conf in classifications
        if str(cls_id) not in excluded_class_ids
    ]

    if not remaining:
        return []

    remaining.sort(key=lambda x: x[1], reverse=True)
    if remaining[0][1] < CONFIDENCE_SCALE_MIN:
        return []
    return remaining


def build_excluded_class_ids(
    class_categories: dict[str, str],
    excluded_labels: list[str] | None = None,
) -> set[str]:
    """
    Build the set of class IDs the user excluded.

    Only the user's own exclusions. The non-label classes are not in
    it: a model's "false detection" is a label like any other, so with
    rollup off it must survive this filter the way it survives the
    rollup path. Until 2026-09 they were always stripped here, which
    was harmless while the ingest dropped such boxes, and wrong once it
    keeps them: the filter emptied the list (the next best class sits
    far below ``CONFIDENCE_SCALE_MIN``), the box came back unclassified,
    and a rejected stump counted as an animal the moment rollup was
    turned off.

    Args:
        class_categories: Mapping of class_id -> class_name from JSON
        excluded_labels: Optional user-configured label names to exclude

    Returns:
        Set of class ID strings to exclude
    """
    if not class_categories or not excluded_labels:
        return set()

    # Names compare lowercase throughout, as the rollup does with the
    # same exclusions; an exact match here would silently skip a class
    # whose model spells it with a capital.
    name_lower_to_ids: dict[str, list[str]] = {}
    for cls_id, name in class_categories.items():
        name_lower_to_ids.setdefault(name.lower(), []).append(cls_id)

    excluded_class_ids: set[str] = set()
    for name in excluded_labels:
        for cls_id in name_lower_to_ids.get(name.lower(), []):
            excluded_class_ids.add(str(cls_id))

    return excluded_class_ids


def apply_label_exclusion_to_results(
    md_results: dict,
    excluded_labels: list[str] | None = None,
    *,
    rollup_handles_exclusion: bool = False,
) -> dict:
    """
    Apply label exclusion to a full MegaDetector JSON results dict (in place).

    Used in the postprocessing path (Phase 7), before smoothing.

    With ``rollup_handles_exclusion`` this is a no-op: the classification
    lists stay untouched so the geofence-aware rollup in
    apply_taxonomic_rollup_to_results() can redirect an excluded top-1 to
    its nearest allowed ancestor with the model's full confidence
    landscape (matching the official SpeciesNet API). The caller sets it
    only when rollup is on and a taxonomy is available; nothing else
    handles an excluded top-1, so deferring without a rollup to defer to
    means the label is later erased instead of replaced.

    Otherwise the user-excluded classes are removed from every
    classification list. The next best included class becomes the
    top-1, at its own score (no renormalisation), and a list that empties
    leaves the detection unclassified. The non-label classes stay: a
    model's "false detection" is its answer, not an exclusion.

    Args:
        md_results: Full MegaDetector JSON dict (modified in place)
        excluded_labels: Optional list of label names to exclude
        rollup_handles_exclusion: True when taxonomic rollup will run
            on these results and owns the excluded classes.

    Returns:
        The modified dict (same reference as input)
    """
    if rollup_handles_exclusion:
        return md_results

    class_categories = md_results.get("classification_categories", {})
    excluded_class_ids = build_excluded_class_ids(
        class_categories, excluded_labels
    )

    if not excluded_class_ids:
        return md_results

    # Iterate `images or []` / `detections or []` so failure entries from
    # process_video (corrupt video → `detections: null`) don't crash this.
    for img in md_results.get("images") or []:
        for det in img.get("detections") or []:
            if "classifications" not in det or not det["classifications"]:
                continue
            det["classifications"] = filter_classifications(
                det["classifications"], excluded_class_ids
            )

    return md_results
