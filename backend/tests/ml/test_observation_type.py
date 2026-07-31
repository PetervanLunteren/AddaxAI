"""Tests for the threshold-aware observation_type derivation.

The rule: a file is the raw detector category of its single strongest
passing detection, strongest being verified first then confidence, and
``"blank"`` when nothing passes.
"""

from dataclasses import dataclass

from app.ml.observation_type import (
    derive_observation_type,
    strongest_passing_detection,
)


@dataclass
class _Det:
    category: str
    confidence: float
    verified: bool = False
    # Not part of the Protocol. Only here so the picking tests below can
    # tell two otherwise identical detections apart.
    label: str | None = None


def test_no_detections_is_blank():
    assert derive_observation_type([], 0.5) == "blank"


def test_over_threshold_animal():
    assert derive_observation_type([_Det("animal", 0.9)], 0.5) == "animal"


def test_all_below_threshold_is_blank():
    # A single sub-threshold box has no trusted content — the case that
    # used to leak into the "animal" fallback folder.
    assert derive_observation_type([_Det("animal", 0.33)], 0.5) == "blank"


def test_verified_below_threshold_still_counts():
    assert (
        derive_observation_type([_Det("animal", 0.1, verified=True)], 0.5)
        == "animal"
    )


# ── The strongest detection decides, not the category ────────────────


def test_more_confident_category_wins():
    """The change made on 2026-07-31. This used to assert "animal",
    because a fixed priority put any animal above any person. That is
    what filed a clip of a person in camouflage under chimpanzee/, off
    one false-positive animal box, while its own picture said Person."""
    dets = [_Det("person", 0.95), _Det("animal", 0.80)]
    assert derive_observation_type(dets, 0.5) == "person"


def test_more_confident_animal_still_wins():
    """The mirror of the above: nothing is biased against animals, the
    stronger box simply wins whichever way round it is."""
    dets = [_Det("person", 0.80), _Det("animal", 0.95)]
    assert derive_observation_type(dets, 0.5) == "animal"


def test_sub_threshold_animal_yields_to_passing_person():
    # Animal box is below threshold, person box is above, so the trusted
    # content is a person.
    dets = [_Det("animal", 0.3), _Det("person", 0.9)]
    assert derive_observation_type(dets, 0.5) == "person"


def test_verified_beats_higher_confidence():
    """A human looked at the person box and confirmed it. That outranks
    a model that is merely more confident about something else, the same
    ordering build_event_primary_labels uses for the event folder."""
    dets = [_Det("animal", 0.99), _Det("person", 0.30, verified=True)]
    assert derive_observation_type(dets, 0.5) == "person"


def test_tie_is_deterministic_regardless_of_order():
    """Callers pass an unordered ORM collection, so an exact tie on
    verified and confidence must not derive differently between runs."""
    a = [_Det("person", 0.9), _Det("animal", 0.9)]
    b = [_Det("animal", 0.9), _Det("person", 0.9)]
    assert derive_observation_type(a, 0.5) == derive_observation_type(b, 0.5)


# ── Any detector's vocabulary passes through ─────────────────────────


def test_unknown_category_is_passed_through():
    """The category is the detector's, not ours. A fish or shark detector
    keeps its own names all the way to the folder. This used to return
    "blank", silently discarding every detection a non-MegaDetector model
    produced."""
    assert derive_observation_type([_Det("shark", 0.9)], 0.5) == "shark"


def test_strongest_wins_across_a_novel_vocabulary():
    dets = [_Det("fish", 0.4), _Det("shark", 0.8), _Det("turtle", 0.6)]
    assert derive_observation_type(dets, 0.2) == "shark"


# ── Picking the detection, not just reading its category ─────────────
#
# The Files export needs the deciding box itself so it can carry that
# box's species. These pin that the picker and the category reading stay
# one rule.


def test_strongest_passing_detection_returns_the_object():
    weak = _Det("animal", 0.6, label="fox")
    strong = _Det("animal", 0.9, label="deer")
    assert strongest_passing_detection([weak, strong], 0.5) is strong


def test_strongest_passing_detection_is_none_when_nothing_passes():
    assert strongest_passing_detection([], 0.5) is None
    assert strongest_passing_detection([_Det("animal", 0.33)], 0.5) is None


def test_strongest_passing_detection_prefers_verified():
    """Object-level mirror of test_verified_beats_higher_confidence, so the
    export cannot report the machine's box over the one a human confirmed."""
    machine = _Det("animal", 0.99, label="deer")
    human = _Det("animal", 0.30, verified=True, label="fox")
    assert strongest_passing_detection([machine, human], 0.5) is human


def test_derive_observation_type_reads_the_picked_detection():
    """The DRY pin. If these two ever disagree, a file's category and its
    species describe different boxes and the export contradicts itself."""
    cases = [
        [],
        [_Det("animal", 0.33)],
        [_Det("animal", 0.9)],
        [_Det("person", 0.95), _Det("animal", 0.80)],
        [_Det("animal", 0.99), _Det("person", 0.30, verified=True)],
        [_Det("fish", 0.4), _Det("shark", 0.8), _Det("turtle", 0.6)],
    ]
    for dets in cases:
        best = strongest_passing_detection(dets, 0.5)
        expected = "blank" if best is None else best.category
        assert derive_observation_type(dets, 0.5) == expected


def test_exact_tie_takes_the_first_in_iteration_order():
    """Two boxes tying on verified, confidence AND category still tie, so
    the caller owns the ordering. build_files_rows passes rows ordered by
    Detection.id, which makes the pick stable for one database."""
    first = _Det("animal", 0.9, label="deer")
    second = _Det("animal", 0.9, label="fox")
    assert strongest_passing_detection([first, second], 0.5) is first
    assert strongest_passing_detection([second, first], 0.5) is second


# ── Threshold lookups refuse to guess ────────────────────────────────
#
# 0.0 is not a neutral fallback. It is the threshold at which every
# detection passes, including MegaDetector's raw 0.005 output floor, so a
# broken lookup used to silently reclassify files and rebuild counts
# against the wrong floor. These pin the refusals.


def test_project_threshold_for_file_refuses_a_broken_chain(db):
    """File -> deployment -> project is NOT NULL with ON DELETE CASCADE
    the whole way, so this is unreachable in a healthy database. That is
    exactly why it must not be papered over."""
    import pytest

    from app.api.crud.file import _project_threshold_for_file
    from app.models import File

    orphan = File(
        id="orphan-file",
        deployment_id="does-not-exist",
        file_path="/fake/x.jpg",
        file_type="image",
        file_format="jpg",
    )
    with pytest.raises(ValueError, match="no reachable project"):
        _project_threshold_for_file(db, orphan)


def test_detection_threshold_refuses_an_empty_id_list(db):
    """There is no sensible threshold for no detections, and returning
    0.0 made callers look like they had one."""
    import pytest

    from app.api.crud.event_observation import (
        get_project_threshold_for_detections,
    )

    with pytest.raises(ValueError, match="at least one"):
        get_project_threshold_for_detections(db, [])


def test_detection_threshold_refuses_unresolvable_ids(db):
    """The join runs through the detection rows, so asking after they are
    deleted resolves to nothing. Callers must capture the threshold
    first; this makes getting that wrong loud instead of silent."""
    import pytest

    from app.api.crud.event_observation import (
        get_project_threshold_for_detections,
    )

    with pytest.raises(ValueError, match="No project reachable"):
        get_project_threshold_for_detections(db, ["already-deleted"])
