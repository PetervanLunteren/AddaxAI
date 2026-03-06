"""Tests for the /api/deployment-queue endpoints."""

from unittest.mock import patch

from tests.conftest import make_project


def _create_entry(client, project_id, site_id=None):
    payload = {
        "project_id": project_id,
        "folder_path": "/some/folder",
        "video_count": 0,
        "image_count": 5,
    }
    if site_id:
        payload["site_id"] = site_id
    return client.post("/api/deployment-queue", json=payload)


def test_list_queue_entries_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/deployment-queue?project_id={p.id}")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_queue_entry(client, db):
    p = make_project(db)
    resp = _create_entry(client, p.id)
    assert resp.status_code == 201
    data = resp.json()
    assert data["project_id"] == p.id
    assert data["status"] == "pending"


def test_create_queue_entry_invalid_project(client):
    resp = _create_entry(client, "nonexistent")
    assert resp.status_code == 400


def test_get_queue_entry(client, db):
    p = make_project(db)
    entry_id = _create_entry(client, p.id).json()["id"]
    resp = client.get(f"/api/deployment-queue/{entry_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == entry_id


def test_get_queue_entry_not_found(client):
    resp = client.get("/api/deployment-queue/nonexistent")
    assert resp.status_code == 404


def test_delete_queue_entry(client, db):
    p = make_project(db)
    entry_id = _create_entry(client, p.id).json()["id"]
    resp = client.delete(f"/api/deployment-queue/{entry_id}")
    assert resp.status_code == 204


def test_delete_queue_entry_not_found(client):
    resp = client.delete("/api/deployment-queue/nonexistent")
    assert resp.status_code == 404


def test_process_queue_no_pending(client, db):
    p = make_project(db)
    resp = client.post("/api/deployment-queue/process", json={"project_id": p.id})
    assert resp.status_code == 202
    assert resp.json()["jobs_started"] == 0


def test_process_queue_with_pending(client, db):
    p = make_project(db)
    _create_entry(client, p.id)
    with patch("app.api.routers.deployment_queue.ws_manager"):
        resp = client.post("/api/deployment-queue/process", json={"project_id": p.id})
    assert resp.status_code == 202
    assert resp.json()["jobs_started"] == 1
    assert len(resp.json()["job_ids"]) == 1
