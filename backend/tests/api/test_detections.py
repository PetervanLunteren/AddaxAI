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
