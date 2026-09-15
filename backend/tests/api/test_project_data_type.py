"""A project's detector and classifier are for one data type.

Classification runs on every box that is not a person or vehicle, and a
box whose top-1 class is "blank" is never loaded, so a camera trap
classifier on an underwater detector drops sharks without a word. The
setup forms only offer models of one data type; the project API refuses
the mix on every other way in.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_project

_CATALOG = {
    "MD5A-0-0": SimpleNamespace(friendly_name="MegaDetector v5a", domain="camera_trap"),
    "CFD-NANO-1-0": SimpleNamespace(
        friendly_name="Community Fish Detector nano", domain="underwater"
    ),
    "SPECIESNET-v4-0-2-A": SimpleNamespace(friendly_name="SpeciesNet", domain="camera_trap"),
    "DINOV2-VITS14": SimpleNamespace(friendly_name="DINOv2", domain=None),
}


def _get_model(model_id: str) -> SimpleNamespace:
    if model_id not in _CATALOG:
        raise ValueError(f"Unknown model: {model_id}")
    return _CATALOG[model_id]


@pytest.fixture(autouse=True)
def catalog():
    mgr = MagicMock()
    mgr.get_model.side_effect = _get_model
    with patch("app.ml.manifest_manager.ManifestManager", return_value=mgr):
        yield mgr


def test_create_refuses_an_underwater_detector_with_a_camera_trap_classifier(client):
    resp = client.post(
        "/api/projects",
        json={
            "name": "reef",
            "detection_model_id": "CFD-NANO-1-0",
            "classification_model_id": "SPECIESNET-v4-0-2-A",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        "Community Fish Detector nano is an underwater model and SpeciesNet "
        "a camera trap model. A project uses models of one data type."
    )


def test_create_allows_an_underwater_detector_without_a_classifier(client):
    resp = client.post(
        "/api/projects",
        json={
            "name": "reef",
            "detection_model_id": "CFD-NANO-1-0",
            "classification_model_id": "none",
        },
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["detection_model_id"] == "CFD-NANO-1-0"


def test_update_checks_a_sent_classifier_against_the_stored_detector(client, db):
    p = make_project(db, detection_model_id="CFD-NANO-1-0")
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"classification_model_id": "SPECIESNET-v4-0-2-A"},
    )
    assert resp.status_code == 400


def test_update_checks_a_sent_detector_against_the_stored_classifier(client, db):
    p = make_project(db, classification_model_id="SPECIESNET-v4-0-2-A")
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"detection_model_id": "CFD-NANO-1-0"},
    )
    assert resp.status_code == 400


def test_update_switches_data_type_when_both_models_change_together(client, db):
    """What the setup forms send when the toggle moves to underwater: the
    new detector and no classifier, in one request."""
    p = make_project(db, classification_model_id="SPECIESNET-v4-0-2-A")
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"detection_model_id": "CFD-NANO-1-0", "classification_model_id": None},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["detection_model_id"] == "CFD-NANO-1-0"
    assert resp.json()["classification_model_id"] is None


def test_update_refuses_a_detector_that_is_not_installed(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}", json={"detection_model_id": "NOT-A-MODEL"}
    )
    assert resp.status_code == 400
    assert "NOT-A-MODEL" in resp.json()["detail"]


def test_renaming_a_project_whose_models_are_gone_still_works(client, db):
    """Only the models a request names are checked, so a stored model
    that was removed since never blocks an unrelated edit."""
    p = make_project(
        db, detection_model_id="GONE-DET", classification_model_id="GONE-CLS"
    )
    resp = client.patch(f"/api/projects/{p.id}", json={"name": "renamed"})
    assert resp.status_code == 200, resp.text


def test_duplicate_checks_the_classifier_against_the_source_detector(client, db):
    source = make_project(db, name="reef", detection_model_id="CFD-NANO-1-0")
    resp = client.post(
        f"/api/projects/{source.id}/duplicate",
        json={"name": "reef copy", "classification_model_id": "SPECIESNET-v4-0-2-A"},
    )
    assert resp.status_code == 400
