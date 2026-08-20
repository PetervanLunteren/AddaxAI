"""Tests for the /api/detections endpoints."""

from unittest.mock import patch

from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
)


def _setup_file(db):
    """Create project → site → deployment → file and return the file."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    return make_file(db, deployment_id=d.id)


def test_create_detection(client, db):
    f = _setup_file(db)
    resp = client.post("/api/detections", json={
        "file_id": f.id,
        "category": "animal",
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_width": 0.3,
        "bbox_height": 0.3,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["category"] == "animal"
    assert data["confidence"] == 1.0  # human-drawn


def test_update_detection(client, db):
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id)
    resp = client.patch(f"/api/detections/{det.id}", json={"category": "person"})
    assert resp.status_code == 200
    assert resp.json()["category"] == "person"


def test_update_detection_not_found(client):
    resp = client.patch("/api/detections/nonexistent", json={"category": "person"})
    assert resp.status_code == 404


def test_patch_relabel_stamps_human_confidence(client, db):
    """A human relabel via PATCH replaces the model's stale score with 1.0,
    matching bulk relabel (the replaced label's score must not linger)."""
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id, label="deer", label_confidence=0.6)
    resp = client.patch(f"/api/detections/{det.id}", json={"label": "elk"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "elk"
    assert data["label_confidence"] == 1.0


def test_relabel_and_verify_leave_original_label_untouched(client, db):
    """The AI's final call (original_label) survives a human relabel and a
    verify, so ai_classification_label keeps showing what the model said."""
    f = _setup_file(db)
    det = make_detection(
        db,
        file_id=f.id,
        label="equidae",
        label_confidence=0.98,
        original_label="equidae",
        original_label_confidence=0.98,
        classification_method="machine",
    )

    # Human relabels to a species.
    resp = client.patch(f"/api/detections/{det.id}", json={"label": "plains zebra"})
    assert resp.status_code == 200
    db.refresh(det)
    assert det.label == "plains zebra"
    assert det.classification_method == "human"
    assert det.original_label == "equidae"           # AI call preserved

    # Human verifies; still no change to the AI call.
    resp = client.patch(f"/api/detections/{det.id}/verify", json={"verified": True})
    assert resp.status_code == 200
    db.refresh(det)
    assert det.verified is True
    assert det.original_label == "equidae"


def test_patch_relabel_honours_explicit_confidence(client, db):
    """An explicit label_confidence in the payload still wins over the 1.0 default."""
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id, label="deer", label_confidence=0.6)
    resp = client.patch(
        f"/api/detections/{det.id}",
        json={"label": "elk", "label_confidence": 0.42},
    )
    assert resp.status_code == 200
    assert resp.json()["label_confidence"] == 0.42


def test_delete_detection(client, db):
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id)
    resp = client.delete(f"/api/detections/{det.id}")
    assert resp.status_code == 204


def test_delete_detection_not_found(client):
    resp = client.delete("/api/detections/nonexistent")
    assert resp.status_code == 404


def test_delete_detections_by_file(client, db):
    f = _setup_file(db)
    make_detection(db, file_id=f.id)
    make_detection(db, file_id=f.id)
    resp = client.delete(f"/api/detections/by-file/{f.id}")
    assert resp.status_code == 200
    assert resp.json()["deleted_count"] == 2


def test_get_crop_success(client, db):
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id)
    fake_jpeg = b"\xff\xd8\xff\xe0fake-jpeg-data"
    with patch(
        "app.api.routers.detections.get_or_create_crop",
        return_value=fake_jpeg,
    ):
        resp = client.get(f"/api/detections/{det.id}/crop")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_get_crop_not_found(client, db):
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id)
    with patch(
        "app.api.routers.detections.get_or_create_crop",
        return_value=None,
    ):
        resp = client.get(f"/api/detections/{det.id}/crop")
    assert resp.status_code == 404


def test_verify_detection(client, db):
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id)
    resp = client.patch(f"/api/detections/{det.id}/verify", json={"verified": True})
    assert resp.status_code == 200
    assert resp.json()["verified"] is True


def test_unverify_detection(client, db):
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id)
    # First verify
    client.patch(f"/api/detections/{det.id}/verify", json={"verified": True})
    # Then unverify
    resp = client.patch(f"/api/detections/{det.id}/verify", json={"verified": False})
    assert resp.status_code == 200
    assert resp.json()["verified"] is False


def test_bulk_verify(client, db):
    f = _setup_file(db)
    d1 = make_detection(db, file_id=f.id)
    d2 = make_detection(db, file_id=f.id)
    resp = client.post("/api/detections/bulk-verify", json={
        "detection_ids": [d1.id, d2.id],
        "verified": True,
    })
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 2


def test_bulk_relabel(client, db):
    f = _setup_file(db)
    d1 = make_detection(db, file_id=f.id)
    d2 = make_detection(db, file_id=f.id)
    resp = client.post("/api/detections/bulk-relabel", json={
        "detection_ids": [d1.id, d2.id],
        "label": "leopard",
    })
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 2


def test_bulk_relabel_unknown_stays_a_real_observation(client, db):
    """Relabelling to "unknown" keeps a counted, verified observation.

    Unlike "false detection", "unknown" is deliberately NOT a
    NON_LABEL_CLASS, so it survives as a real observation. The relabel
    keeps the category, verifies the detection, and auto-creates a custom
    taxonomy row resolving both names to "Unknown". This backs the
    Labels-page "Unknown" action.
    """
    from app.ml.label_exclusion import NON_LABEL_CLASSES

    assert "unknown" not in NON_LABEL_CLASSES

    f = _setup_file(db)
    d = make_detection(db, file_id=f.id, label="crow", category="animal")
    resp = client.post("/api/detections/bulk-relabel", json={
        "detection_ids": [d.id],
        "label": "unknown",
    })
    assert resp.status_code == 200

    db.refresh(d)
    assert d.label == "unknown"
    assert d.verified is True
    assert d.category == "animal"  # category left untouched
    assert d.common_name == "Unknown"
    assert d.scientific_name == "Unknown"
    assert d.label_taxonomy_id is not None  # auto-created custom row


def test_bulk_dismiss_sets_and_clears_flag(client, db):
    f = _setup_file(db)
    d1 = make_detection(db, file_id=f.id, label="crow")
    d2 = make_detection(db, file_id=f.id, label="crow")

    # Dismiss: sets the flag, leaves label and verified untouched.
    resp = client.post("/api/detections/bulk-dismiss", json={
        "detection_ids": [d1.id, d2.id],
        "dismissed": True,
    })
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 2
    db.refresh(d1)
    db.refresh(d2)
    assert d1.suggestion_dismissed is True
    assert d2.suggestion_dismissed is True
    assert d1.label == "crow"
    assert d1.verified is False

    # Undo: clears the flag again.
    resp = client.post("/api/detections/bulk-dismiss", json={
        "detection_ids": [d1.id, d2.id],
        "dismissed": False,
    })
    assert resp.status_code == 200
    db.refresh(d1)
    assert d1.suggestion_dismissed is False


def test_create_detection_rejects_half_set_bbox(client, db):
    """Pydantic guard: bbox fields must be all-set or all-null. Targets
    the bbox-drawn create endpoint with a partial bbox to confirm the
    validator fires."""
    f = _setup_file(db)
    resp = client.post(
        "/api/detections",
        json={
            "file_id": f.id,
            "category": "animal",
            "bbox_x": 0.1,
            "bbox_y": 0.1,
            "bbox_width": 0.3,
            # bbox_height intentionally missing
        },
    )
    assert resp.status_code == 422


def test_bulk_revert_to_original_restores_ai_label(client, db):
    """Undo: a human relabel + verify is reverted to the model's
    original prediction, verified cleared, method back to machine."""
    f = _setup_file(db)
    det = make_detection(
        db,
        file_id=f.id,
        category="animal",
        label="deer",
        label_confidence=0.42,
        original_label="deer",
        original_label_confidence=0.42,
        classification_method="machine",
        verified=False,
    )
    # Simulate a human match-majority: relabel + verify.
    client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [det.id], "label": "elk"},
    )
    db.refresh(det)
    assert det.label == "elk" and det.verified is True
    assert det.classification_method == "human"

    resp = client.post(
        "/api/detections/bulk-revert-to-original",
        json={"detection_ids": [det.id]},
    )
    assert resp.status_code == 200
    reverted = resp.json()["reverted"]
    assert reverted[0]["detection_id"] == det.id
    assert reverted[0]["label"] == "deer"
    assert reverted[0]["verified"] is False

    db.refresh(det)
    assert det.label == "deer"
    assert det.label_confidence == 0.42
    assert det.verified is False
    assert det.classification_method == "machine"


def test_bulk_revert_unverifies_a_plain_verify(client, db):
    """A detection that was only verified (label untouched) reverts to
    unverified with its label intact."""
    f = _setup_file(db)
    det = make_detection(
        db, file_id=f.id, label="fox", original_label="fox", verified=False
    )
    client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [det.id], "verified": True},
    )
    db.refresh(det)
    assert det.verified is True

    client.post(
        "/api/detections/bulk-revert-to-original",
        json={"detection_ids": [det.id]},
    )
    db.refresh(det)
    assert det.verified is False
    assert det.label == "fox"


def test_bulk_revert_no_original_clears_label(client, db):
    """A detection with no original AI label (e.g. added by hand) reverts
    to an unlabeled, unverified state."""
    f = _setup_file(db)
    det = make_detection(
        db, file_id=f.id, label="dog", original_label=None, verified=True
    )
    client.post(
        "/api/detections/bulk-revert-to-original",
        json={"detection_ids": [det.id]},
    )
    db.refresh(det)
    assert det.label is None
    assert det.verified is False


def test_bulk_revert_404_for_unknown(client):
    resp = client.post(
        "/api/detections/bulk-revert-to-original",
        json={"detection_ids": ["nope"]},
    )
    assert resp.status_code == 404


def test_a_hand_drawn_box_is_created_verified(client, db):
    """A box someone drew by hand is the strongest signal there is, so
    it gets the same protection as a box they confirmed.

    Without the flag it was the one human decision the pipeline could
    overwrite: postprocessing skips only verified rows, and the
    machine-final mirror at the end of
    `update_database_from_smoothed_results` rewrites `original_label`
    for everything else, so "revert to AI" would have reverted to the
    human's own label.
    """
    from app.models import Detection

    f = _setup_file(db)
    resp = client.post("/api/detections", json={
        "file_id": f.id,
        "category": "animal",
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_width": 0.3,
        "bbox_height": 0.3,
    })
    assert resp.status_code == 201

    det = db.query(Detection).filter(Detection.id == resp.json()["id"]).one()
    assert det.verified is True
    assert det.verified_at_utc is not None


def test_drawing_a_box_takes_the_photo_out_of_the_empties(client, db):
    """The loop the Empties tab promises: find the animal the detector
    missed, draw it, and the photo leaves the list. It works because a
    hand-drawn box is stored at confidence 1.0, so it passes any floor."""
    f = _setup_file(db)
    project_id = f.deployment.project_id

    before = client.get(f"/api/projects/{project_id}/labels/empties").json()
    assert before["total"] == 1

    client.post("/api/detections", json={
        "file_id": f.id,
        "category": "animal",
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_width": 0.3,
        "bbox_height": 0.3,
    })

    after = client.get(f"/api/projects/{project_id}/labels/empties").json()
    assert after["total"] == 0
