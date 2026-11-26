"""Every detector class is a label row, and the server decides a hand-made
box's class.

An underwater detector says "fish" or "elasmobranch", not "animal". A box
without a species points at its class's builtin row, whatever the class,
so the label filter finds it; and a box drawn by hand, or a species added
by hand on the Counts page, takes the project detector's `primary_class`
instead of the "animal" the frontend used to send.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from app.api.crud.sequence_observation import calculate_max_n_for_sequence
from app.ml.taxonomy_db import builtin_label_ids
from app.models import Detection, SequenceObservation
from tests.conftest import (
    label_counts,
    label_tree,
    make_deployment,
    make_detection,
    make_project,
    make_sequence_with_files,
    make_site,
)


@pytest.fixture
def fish_detector(monkeypatch):
    """The project's detector is the Community Fish Detector."""
    from app.ml import primary_class

    manifest = Mock(model_id="CFD-NANO-1-0", classes=["fish"], primary_class="fish")
    monkeypatch.setattr(
        primary_class, "ManifestManager",
        lambda *a, **k: Mock(get_model=Mock(return_value=manifest)),
    )
    return manifest


def _fish_sequence(db):
    project = make_project(db, detection_model_id="CFD-NANO-1-0")
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    sequence = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2026, 2, 10, 8, 33),
    )
    return project, sequence, sequence.files[0]


def _box(client, file_id, **body):
    r = client.post(
        "/api/detections",
        json={"file_id": file_id, "bbox_x": 0.1, "bbox_y": 0.1,
              "bbox_width": 0.2, "bbox_height": 0.2, **body},
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_a_box_drawn_on_a_fish_project_is_a_fish(client, db, fish_detector):
    """No category sent: the box takes the detector's primary class and its
    builtin row, so it is a "Fish" in the filter, not a stray "Animal"."""
    _, _, f = _fish_sequence(db)
    db.commit()

    plain = _box(client, f.id)
    named = _box(client, f.id, label="cod")

    fish_row = builtin_label_ids(db)["fish"]
    assert (plain["category"], plain["label_taxonomy_id"], plain["scientific_name"]) == (
        "fish", fish_row, "Fish",
    )
    assert named["category"] == "fish"
    assert named["label"] == "cod"


def test_a_category_sent_beside_a_species_is_ignored(client, db, fish_detector):
    """Saved hotkeys still carry the old "animal" beside a species. The
    species decides; the box stays a fish."""
    _, _, f = _fish_sequence(db)
    db.commit()
    assert _box(client, f.id, label="cod", category="animal")["category"] == "fish"


def test_a_person_box_named_as_a_species_moves_to_the_primary_class(
    client, db, fish_detector
):
    """A species on a person box says the detector was wrong about what
    kind of thing it is: the box moves to the class a species belongs to."""
    _, _, f = _fish_sequence(db)
    det = make_detection(db, file_id=f.id, category="person")
    db.commit()

    r = client.patch(f"/api/detections/{det.id}", json={"label": "cod"})
    assert r.status_code == 200
    assert r.json()["category"] == "fish"


def test_bulk_relabel_keeps_each_box_on_its_own_class(client, db, fish_detector):
    """Clearing a species without naming a class leaves every box on its
    own class's row (it used to leave them with none); picking a class
    moves them to it."""
    _, _, f = _fish_sequence(db)
    cod = make_detection(db, file_id=f.id, category="fish", label="cod")
    db.commit()

    r = client.post("/api/detections/bulk-relabel",
                    json={"detection_ids": [cod.id], "label": None})
    assert r.status_code == 200
    db.expire_all()
    builtin = builtin_label_ids(db)
    assert (cod.category, cod.label, cod.label_taxonomy_id) == ("fish", None, builtin["fish"])

    r = client.post("/api/detections/bulk-relabel",
                    json={"detection_ids": [cod.id], "label": None, "category": "person"})
    assert r.status_code == 200
    db.expire_all()
    assert (cod.category, cod.label_taxonomy_id) == ("person", builtin_label_ids(db)["person"])


def test_a_fish_project_has_a_fish_leaf_at_the_root(db, fish_detector):
    """The label filter tree offers the detector's class as a root leaf,
    so a project with nothing but fish can still filter on them."""
    project, _, f = _fish_sequence(db)
    for _ in range(3):
        make_detection(db, file_id=f.id, category="fish")
    db.flush()

    tree = label_tree(db, project.id)
    fish = builtin_label_ids(db)["fish"]
    assert [n["name"] for n in tree["tree"]] == ["Fish"]
    assert tree["all_leaf_ids"] == [fish]
    assert label_counts(db, project.id, "detection") == {fish: 3}


def test_adding_a_species_by_hand_gets_the_primary_class(client, db, fish_detector):
    _, sequence, _ = _fish_sequence(db)
    db.commit()

    r = client.post(f"/api/sequences/{sequence.id}/observations",
                    json={"count": 2, "label": "cod"})
    assert r.status_code == 200, r.text
    obs = db.query(SequenceObservation).filter_by(sequence_id=sequence.id).one()
    assert (obs.category, obs.label, obs.human_count) == ("fish", "cod", 2)


def test_adding_a_class_the_sequence_already_counts_changes_nothing(
    client, db, fish_detector
):
    """The AI counted 3 fish; picking "Fish" in Add species again must not
    overwrite that with 1. The row stays as it was; + and - change it."""
    project, sequence, f = _fish_sequence(db)
    for _ in range(3):
        make_detection(db, file_id=f.id, category="fish")
    calculate_max_n_for_sequence(db, sequence.id, project.counting_threshold)
    db.commit()

    r = client.post(f"/api/sequences/{sequence.id}/observations",
                    json={"count": 1, "category": "fish"})
    assert r.status_code == 200, r.text
    rows = db.query(SequenceObservation).filter_by(sequence_id=sequence.id).all()
    assert [(o.label, o.max_n, o.human_count, o.effective_count) for o in rows] == [
        ("fish", 3, None, 3)
    ]


def test_a_class_count_added_by_hand_survives_a_rebuild_and_meets_the_ai_row(
    client, db, fish_detector
):
    """A count for the class itself ("Fish", no species) used to be stored
    with no label and no row: the next MaxN rebuild dropped it, and it
    never met the AI's row for that class. Now it is stored as an AI row
    of the class is, so when the AI does see a fish the two are one row
    carrying the person's count."""
    project, sequence, f = _fish_sequence(db)
    db.commit()

    r = client.post(f"/api/sequences/{sequence.id}/observations",
                    json={"count": 4, "category": "fish"})
    assert r.status_code == 200, r.text

    make_detection(db, file_id=f.id, category="fish")
    calculate_max_n_for_sequence(db, sequence.id, project.counting_threshold)
    db.commit()
    rows = db.query(SequenceObservation).filter_by(sequence_id=sequence.id).all()
    assert [(o.label, o.max_n, o.human_count) for o in rows] == [("fish", 1, 4)]
    assert rows[0].label_taxonomy_id == builtin_label_ids(db)["fish"]


def test_a_detector_without_a_primary_class_is_a_400_naming_it(client, db, monkeypatch):
    """A catalog error is reported by model name, not guessed around."""
    from app.ml import primary_class

    manifest = Mock(model_id="OLD-DET-1-0", classes=["fish"], primary_class=None)
    monkeypatch.setattr(
        primary_class, "ManifestManager",
        lambda *a, **k: Mock(get_model=Mock(return_value=manifest)),
    )
    _, _, f = _fish_sequence(db)
    db.commit()

    r = client.post(
        "/api/detections",
        json={"file_id": f.id, "bbox_x": 0.1, "bbox_y": 0.1,
              "bbox_width": 0.2, "bbox_height": 0.2},
    )
    assert r.status_code == 400
    assert "OLD-DET-1-0" in r.json()["detail"]
    assert db.query(Detection).count() == 0


def test_counts_name_the_detectors_own_classes(client, db, fish_detector):
    """The info sheets and the dashboard's Files tile count per class by
    the class's own name: a fish is a fish, not an "animal", and a class
    the project never saw (person, vehicle) is not listed at zero."""
    project, sequence, f = _fish_sequence(db)
    for _ in range(2):
        make_detection(db, file_id=f.id, category="fish")
    f.observation_type = "fish"
    calculate_max_n_for_sequence(db, sequence.id, project.counting_threshold)
    db.commit()

    dep_id = sequence.deployment_id
    sheet = client.get(f"/api/deployments/{dep_id}/info").json()
    assert sheet["detection_categories"] == {"classes": {"fish": 2}, "empty": 0}
    tile = client.get(f"/api/statistics/categories?project_id={project.id}").json()
    assert tile == {"classes": {"fish": 1}, "empty": 0}
