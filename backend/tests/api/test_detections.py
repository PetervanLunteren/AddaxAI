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


def test_drawing_a_box_on_a_file_that_is_gone_is_a_404(client, db):
    """A stale file id is the client's mistake, not a server fault.

    The grid holds ids fetched earlier, so a deployment deleted in another
    window is the ordinary way to arrive with one. Left to the foreign key
    it surfaced as `500 Internal Server Error`, which tells the user
    nothing and reads as a crash.
    """
    resp = client.post("/api/detections", json={
        "file_id": "00000000-0000-0000-0000-000000000000",
        "category": "animal",
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_width": 0.3,
        "bbox_height": 0.3,
    })
    assert resp.status_code == 404
    assert resp.json()["detail"] == "File not found"


def test_a_label_only_edit_recomputes_what_the_file_is_about(client, db):
    """The label feeds `observation_type`, not just the category.

    Marking the only box a false detection takes it out of the running for
    what the file is about, so the file becomes blank. Bulk relabel always
    did this; the PATCH path recomputed only on a category change and left
    the value stale.
    """
    from app.models import Detection, File

    f = _setup_file(db)
    f.observation_type = "animal"
    det = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
    db.commit()

    resp = client.patch(
        f"/api/detections/{det.id}", json={"label": "false detection"}
    )
    assert resp.status_code == 200

    db.expire_all()
    assert db.get(Detection, det.id).label == "false detection"
    assert db.get(File, f.id).observation_type == "blank"


def test_an_empty_bulk_request_is_zero_rows_not_a_missing_row(client, db):
    """Both bulk endpoints answer the same way to an empty list.

    Relabel used to 404 with "No detections found" while bulk-verify
    answered 200 with a count of zero. Nothing asked for is nothing done,
    and there is nothing to undo.
    """
    nothing = {"updated_count": 0, "before": {"detections": [], "files": []}}
    assert client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [], "verified": True},
    ).json() == nothing
    assert client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [], "label": "chital"},
    ).json() == nothing


def test_an_unknown_id_in_a_bulk_request_is_still_a_404(client, db):
    """The empty-list shortcut must not swallow a genuinely missing row."""
    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": ["ghost"], "label": "chital"},
    )
    assert resp.status_code == 404


def test_drawing_a_box_marks_the_file_verified_past_the_request(client, db):
    """The rollup has to survive the end of the request, not just reach it.

    `get_db` yields a session and then only closes it, so anything a route
    leaves pending is discarded. `recompute_file_verified` does not commit
    (its caller owns the transaction) and the commit after it used to sit
    behind "did this file belong to an event", which a file only fails when
    event generation never reached it. So the box was saved, the rollup was
    not, and the photo exported `is_verified = FALSE` right after a person
    had drawn on it.

    `db.rollback()` here is the request boundary: it throws away exactly what
    `close()` throws away, and leaves everything already committed. Without it
    the assertion reads the live transaction and passes either way, which is
    what let this sit unnoticed.
    """
    from app.models import Event, File

    f = _setup_file(db)
    db.commit()
    assert not db.query(Event).filter(Event.deployment_id == f.deployment_id).all()

    resp = client.post("/api/detections", json={
        "file_id": f.id,
        "category": "animal",
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_width": 0.3,
        "bbox_height": 0.3,
    })
    assert resp.status_code == 201

    db.rollback()
    assert db.get(File, f.id).verified is True


def test_drawing_a_box_takes_the_photo_out_of_the_empties(client, db):
    """The loop the Empties tab promises: find the animal the detector
    missed, draw it, and the photo leaves the list. It works because a
    hand-drawn box is stored at confidence 1.0, so it passes any floor."""
    f = _setup_file(db)
    project_id = f.deployment.project_id
    show_only = {"empty": "show_only"}

    before = client.get(f"/api/projects/{project_id}/labels/files", params=show_only).json()
    assert before["total"] == 1

    client.post("/api/detections", json={
        "file_id": f.id,
        "category": "animal",
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_width": 0.3,
        "bbox_height": 0.3,
    })

    after = client.get(f"/api/projects/{project_id}/labels/files", params=show_only).json()
    assert after["total"] == 0


def test_clearing_a_label_leaves_the_builtin_animal_row(client, db):
    """A PATCH that empties the label must not empty the taxonomy id: the
    label filter matches on that id, so a box without one is counted and
    shown but unreachable through "Select all" or the Animal leaf."""
    from app.ml.taxonomy_db import ensure_builtin_labels

    animal_row = ensure_builtin_labels(db)["animal"]
    f = _setup_file(db)
    det = make_detection(db, file_id=f.id, label="deer", label_confidence=0.6)

    resp = client.patch(f"/api/detections/{det.id}", json={"label": None})
    assert resp.status_code == 200

    db.refresh(det)
    assert det.label is None
    assert det.label_confidence is None
    assert det.label_taxonomy_id == animal_row
    assert det.common_name == "Animal"
