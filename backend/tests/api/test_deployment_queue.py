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


def test_file_mtime_fallback_defaults_off(client, db):
    """Omitting the field must never silently enable the fallback."""
    p = make_project(db)
    resp = _create_entry(client, p.id)
    assert resp.status_code == 201
    assert resp.json()["use_file_mtime_fallback"] is False


def test_file_mtime_fallback_round_trips(client, db):
    """The user ticks the box at queue-add time and the worker reads it
    minutes or days later, so it has to survive on the row."""
    p = make_project(db)
    resp = client.post(
        "/api/deployment-queue",
        json={
            "project_id": p.id,
            "folder_path": "/some/folder",
            "video_count": 1,
            "image_count": 0,
            "use_file_mtime_fallback": True,
        },
    )
    assert resp.status_code == 201
    entry_id = resp.json()["id"]

    fetched = client.get(f"/api/deployment-queue/{entry_id}")
    assert fetched.status_code == 200
    assert fetched.json()["use_file_mtime_fallback"] is True


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


def test_paired_cameras_defaults_off(client, db):
    """Omitting the field must never silently pair the subfolders."""
    p = make_project(db)
    resp = _create_entry(client, p.id)
    assert resp.status_code == 201
    assert resp.json()["paired_cameras"] is False


def test_paired_cameras_round_trips(client, db, tmp_path):
    """Chosen at queue-add time, read by the worker later, so it lives on
    the row."""
    p = make_project(db)
    for cam in ("cam1", "cam2"):
        (tmp_path / cam).mkdir()
        (tmp_path / cam / "a.jpg").write_bytes(b"x")
    resp = client.post(
        "/api/deployment-queue",
        json={
            "project_id": p.id,
            "folder_path": str(tmp_path),
            "image_count": 2,
            "paired_cameras": True,
        },
    )
    assert resp.status_code == 201
    fetched = client.get(f"/api/deployment-queue/{resp.json()['id']}")
    assert fetched.json()["paired_cameras"] is True


def test_paired_cameras_needs_two_camera_subfolders(client, db, tmp_path):
    """The flag describes a folder layout, so the queue refuses it for a
    folder that does not have that layout: flat files, or one subfolder."""
    from app.services.csv_import_deployments import PAIRED_CAMERAS_NEED_SUBFOLDERS

    p = make_project(db)
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "a.jpg").write_bytes(b"x")
    one = tmp_path / "one"
    (one / "cam1").mkdir(parents=True)
    (one / "cam1" / "a.jpg").write_bytes(b"x")
    pair = tmp_path / "pair"
    for cam in ("cam1", "cam2"):
        (pair / cam).mkdir(parents=True)
        (pair / cam / "a.jpg").write_bytes(b"x")

    def post(folder, paired):
        return client.post(
            "/api/deployment-queue",
            json={
                "project_id": p.id,
                "folder_path": str(folder),
                "image_count": 1,
                "paired_cameras": paired,
            },
        )

    for folder in (flat, one):
        resp = post(folder, True)
        assert resp.status_code == 400
        assert resp.json()["detail"] == PAIRED_CAMERAS_NEED_SUBFOLDERS
    assert post(pair, True).status_code == 201
    # Unpaired never looks at the layout.
    assert post(flat, False).status_code == 201
