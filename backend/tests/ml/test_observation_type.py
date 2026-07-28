"""Tests for the threshold-aware observation_type derivation."""

from dataclasses import dataclass

from app.ml.observation_type import derive_observation_type


@dataclass
class _Det:
    category: str
    confidence: float
    verified: bool = False


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


def test_priority_animal_over_person():
    dets = [_Det("person", 0.9), _Det("animal", 0.9)]
    assert derive_observation_type(dets, 0.5) == "animal"


def test_sub_threshold_animal_yields_to_passing_person():
    # Animal box is below threshold, person box is above — the trusted
    # content is a person, so the file is "human", not "animal".
    dets = [_Det("animal", 0.3), _Det("person", 0.9)]
    assert derive_observation_type(dets, 0.5) == "human"


def test_unknown_category_ignored():
    assert derive_observation_type([_Det("mystery", 0.9)], 0.5) == "blank"
