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


# ── Event-level observations (no bbox) ─────────────────────────────


def test_create_observation_on_image_has_null_frame_number(client, db):
    """Images have no best frame, so observations land at NULL
    frame_number — same as the AI's image detections, which keeps
    them in the same MaxN group."""
    f = _setup_file(db)
    resp = client.post(
        "/api/detections/observation",
        json={"file_id": f.id, "category": "animal"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["bbox_x"] is None
    assert data["bbox_y"] is None
    assert data["bbox_width"] is None
    assert data["bbox_height"] is None
    assert data["category"] == "animal"
    assert data["classification_method"] == "human"
    assert data["confidence"] == 1.0
    assert data["verified"] is True
    assert data["verified_at_utc"] is not None
    assert data["frame_number"] is None


def test_create_observation_on_video_inherits_best_frame_number(client, db):
    """Observations on a video are stamped with the video's
    `best_frame_number`. This lands them in the same MaxN group as
    the AI's best-frame detections, so verifying a clip ("I saw 7
    more deer the AI missed") correctly bumps MaxN by 7."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(
        db,
        deployment_id=d.id,
        file_type="video",
        best_frame_number=42,
    )
    resp = client.post(
        "/api/detections/observation",
        json={"file_id": f.id, "category": "animal", "label": "red deer"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["label"] == "red deer"
    assert data["frame_number"] == 42


def test_create_observation_ignores_frame_number_if_supplied(client, db):
    """Even if a client sends `frame_number`, the backend overrides it
    with the file's `best_frame_number` (or NULL for images). The
    field is not part of the documented schema; this test pins the
    deliberate override so a regression that honours client-supplied
    frame numbers gets caught."""
    f = _setup_file(db)
    resp = client.post(
        "/api/detections/observation",
        json={
            "file_id": f.id,
            "category": "animal",
            "label": "red deer",
            "frame_number": 320,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["label"] == "red deer"
    assert data["label_confidence"] == 1.0
    # Image fixture has no best_frame_number → observation stays null.
    assert data["frame_number"] is None


def test_get_crop_404s_for_no_bbox_observation(client, db):
    """Event-level observations have no bbox to crop; the crop endpoint
    should 404 cleanly rather than crash."""
    f = _setup_file(db)
    resp = client.post(
        "/api/detections/observation",
        json={"file_id": f.id, "category": "animal"},
    )
    det_id = resp.json()["id"]
    crop_resp = client.get(f"/api/detections/{det_id}/crop")
    assert crop_resp.status_code == 404


def test_create_observation_schema_rejects_half_set_bbox(client, db):
    """Pydantic guard: bbox fields must be all-set or all-null. The
    /observation endpoint doesn't accept bbox at all, so this test
    targets the bbox-drawn create endpoint with a partial bbox to
    confirm the validator fires."""
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
