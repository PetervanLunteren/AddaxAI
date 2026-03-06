"""Tests for the /api/deployments endpoints."""

from unittest.mock import patch

from tests.conftest import make_deployment, make_project, make_site


def test_list_deployments_empty(client):
    resp = client.get("/api/deployments")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_deployment(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    resp = client.post("/api/deployments", json={
        "site_id": s.id,
        "start_date": "2024-01-01",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["site_id"] == s.id


def test_create_deployment_invalid_site(client):
    resp = client.post("/api/deployments", json={
        "site_id": "nonexistent",
        "start_date": "2024-01-01",
    })
    assert resp.status_code == 400


def test_get_deployment(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    resp = client.get(f"/api/deployments/{d.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == d.id


def test_get_deployment_not_found(client):
    resp = client.get("/api/deployments/nonexistent")
    assert resp.status_code == 404


def test_update_deployment(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    resp = client.patch(f"/api/deployments/{d.id}", json={"notes": "updated"})
    assert resp.status_code == 200
    assert resp.json()["notes"] == "updated"


def test_delete_deployment(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    resp = client.delete(f"/api/deployments/{d.id}")
    assert resp.status_code == 204


def test_delete_deployment_not_found(client):
    resp = client.delete("/api/deployments/nonexistent")
    assert resp.status_code == 404


def test_preview_folder_success(client):
    mock_result = {
        "image_count": 10,
        "video_count": 2,
        "total_count": 12,
        "gps_location": None,
        "sample_files": [],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "missing_datetime": 0,
        "datetime_validation_log": [],
    }
    with patch("app.api.routers.deployments.scan_folder", return_value=mock_result):
        resp = client.get("/api/deployments/preview-folder?path=/some/folder")
    assert resp.status_code == 200
    assert resp.json()["image_count"] == 10


def test_preview_folder_not_found(client):
    with patch(
        "app.api.routers.deployments.scan_folder",
        side_effect=FileNotFoundError("not found"),
    ):
        resp = client.get("/api/deployments/preview-folder?path=/missing")
    assert resp.status_code == 400
