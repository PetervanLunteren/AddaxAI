"""Tests for the /api/projects endpoints."""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_project


@pytest.fixture(autouse=True)
def mock_manifest_manager():
    """Patch ManifestManager so model validation always succeeds."""
    mock_mgr = MagicMock()
    mock_mgr.get_model.return_value = MagicMock(model_id="MD5A-0-0")
    with patch("app.ml.manifest_manager.ManifestManager", return_value=mock_mgr):
        yield mock_mgr


# --- List / Create / Get / Update / Delete ---


def test_list_projects_empty(client):
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_project(client):
    resp = client.post("/api/projects", json={"name": "My Project"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "My Project"
    assert "id" in data
    assert "created_at" in data


def test_create_project_duplicate_name(client, db):
    make_project(db, name="dup")
    resp = client.post("/api/projects", json={"name": "dup"})
    assert resp.status_code == 409


def test_get_project(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == p.id


def test_get_project_not_found(client):
    resp = client.get("/api/projects/nonexistent")
    assert resp.status_code == 404


def test_update_project_name(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"name": "new-name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new-name"


def test_update_project_not_found(client):
    resp = client.patch("/api/projects/nonexistent", json={"name": "x"})
    assert resp.status_code == 404


def test_delete_project(client, db):
    p = make_project(db)
    resp = client.delete(f"/api/projects/{p.id}")
    assert resp.status_code == 204


def test_delete_project_not_found(client):
    resp = client.delete("/api/projects/nonexistent")
    assert resp.status_code == 404


# --- Stats ---


def test_get_project_stats_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_count"] == 0
    assert data["deployment_count"] == 0


def test_get_project_stats_not_found(client):
    resp = client.get("/api/projects/nonexistent/stats")
    assert resp.status_code == 404


def test_get_detection_stats_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/detection-stats")
    assert resp.status_code == 200
    assert resp.json() == {}


def test_get_detection_count_zero(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/detection-count")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_get_label_stats_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/label-stats")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_independent_event_stats_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/independent-event-stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["labels"] == []


# --- Reprocess / Re-embed ---


def test_reprocess_not_found(client):
    resp = client.post("/api/projects/nonexistent/reprocess")
    assert resp.status_code == 404


def test_reprocess_success(client, db):
    p = make_project(db)
    with patch("app.api.routers.projects.ws_manager"):
        resp = client.post(f"/api/projects/{p.id}/reprocess")
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_re_embed_no_model(client, db):
    p = make_project(db)
    # Must explicitly set after creation since column default overrides None
    p.embedding_model_id = None
    db.flush()
    resp = client.post(f"/api/projects/{p.id}/re-embed")
    assert resp.status_code == 202
    assert resp.json()["job_id"] is None


def test_re_embed_with_model(client, db):
    p = make_project(db, embedding_model_id="DINOV2-VITB14")
    with patch("app.api.routers.projects.ws_manager"):
        resp = client.post(f"/api/projects/{p.id}/re-embed")
    assert resp.status_code == 202
    assert resp.json()["job_id"] is not None


# --- Postprocessing status ---


def test_postprocessing_status_no_classifications(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/postprocessing-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_classifications"] is False


# --- Batch size overrides ---


def test_create_project_default_batch_sizes_are_null(client):
    """A new project should have NULL for all three batch_size fields."""
    resp = client.post("/api/projects", json={"name": "bs-default"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["detection_batch_size"] is None
    assert data["classification_batch_size"] is None
    assert data["embedding_batch_size"] is None


def test_update_project_batch_sizes_persist(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={
            "detection_batch_size": 16,
            "classification_batch_size": 64,
            "embedding_batch_size": 128,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["detection_batch_size"] == 16
    assert data["classification_batch_size"] == 64
    assert data["embedding_batch_size"] == 128

    # Reload via GET to confirm persistence
    resp = client.get(f"/api/projects/{p.id}")
    data = resp.json()
    assert data["detection_batch_size"] == 16
    assert data["classification_batch_size"] == 64
    assert data["embedding_batch_size"] == 128


def test_update_project_batch_size_to_null(client, db):
    """Setting a batch_size back to null reverts to the default."""
    p = make_project(db)
    client.patch(f"/api/projects/{p.id}", json={"detection_batch_size": 16})
    resp = client.patch(f"/api/projects/{p.id}", json={"detection_batch_size": None})
    assert resp.status_code == 200
    assert resp.json()["detection_batch_size"] is None


def test_update_project_batch_size_zero_is_rejected(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"detection_batch_size": 0})
    assert resp.status_code == 422


def test_update_project_batch_size_negative_is_rejected(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"classification_batch_size": -1})
    assert resp.status_code == 422


def test_update_project_batch_size_too_large_is_rejected(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"embedding_batch_size": 999})
    assert resp.status_code == 422
