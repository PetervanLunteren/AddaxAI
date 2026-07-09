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
    resp = client.post(
        "/api/projects",
        json={"name": "My Project", "timezone": "UTC"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "My Project"
    assert data["timezone"] == "UTC"
    assert "id" in data
    assert "created_at_utc" in data


def test_create_project_duplicate_name(client, db):
    make_project(db, name="dup")
    resp = client.post("/api/projects", json={"name": "dup", "timezone": "UTC"})
    assert resp.status_code == 409


def test_create_project_without_timezone_starts_unset(client):
    """Timezone is optional: a new project starts with none and later
    auto-derives one from its first site's coordinates."""
    resp = client.post("/api/projects", json={"name": "no-tz"})
    assert resp.status_code == 201
    assert resp.json()["timezone"] is None


def test_create_project_rejects_invalid_timezone(client):
    """Bogus IANA strings get rejected by the field validator."""
    resp = client.post(
        "/api/projects",
        json={"name": "bad-tz", "timezone": "Moon/Crater"},
    )
    assert resp.status_code == 422


def test_create_project_accepts_valid_iana_timezone(client):
    resp = client.post(
        "/api/projects",
        json={"name": "amsterdam", "timezone": "Europe/Amsterdam"},
    )
    assert resp.status_code == 201
    assert resp.json()["timezone"] == "Europe/Amsterdam"


def test_update_project_timezone(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"timezone": "America/New_York"},
    )
    assert resp.status_code == 200
    assert resp.json()["timezone"] == "America/New_York"


def test_update_project_rejects_invalid_timezone(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"timezone": "Foo/Bar"},
    )
    assert resp.status_code == 422


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


def _add_observation(db, *, event_id, label, category="animal", max_n=1, human_count=None):
    from app.models.event_observation import EventObservation

    db.add(
        EventObservation(
            event_id=event_id,
            label=label,
            label_taxonomy_id=None,
            category=category,
            max_n=max_n,
            human_count=human_count,
        )
    )
    db.flush()


def test_independent_observation_stats_honours_human_count(client, db):
    """Abundance uses effective_count (human_count when set, not raw max_n),
    counts human-only species, and excludes person/vehicle."""
    from datetime import datetime

    from tests.conftest import make_deployment, make_event_with_files

    p = make_project(db)
    dep = make_deployment(db, project_id=p.id)
    ev = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 1, 12, 0, 0)
    )
    # AI said 2 lions, human confirmed 5.
    _add_observation(db, event_id=ev.id, label="lion", max_n=2, human_count=5)
    # Human added a species the AI missed (max_n=0).
    _add_observation(db, event_id=ev.id, label="fox", max_n=0, human_count=1)
    # Person is not part of label refinement and must be excluded.
    _add_observation(db, event_id=ev.id, label="person", category="person", max_n=4)

    resp = client.get(f"/api/projects/{p.id}/independent-observation-stats")
    assert resp.status_code == 200
    data = resp.json()
    labels = {row["label"]: row["count"] for row in data["labels"]}
    assert labels == {"lion": 5, "fox": 1}
    assert data["total"] == 6


def test_independent_event_stats_from_observations(client, db):
    """Frequency counts distinct events per label from the materialized
    observations (so human-added species count, person is excluded)."""
    from datetime import datetime

    from tests.conftest import make_deployment, make_event_with_files

    p = make_project(db)
    dep = make_deployment(db, project_id=p.id)
    ev1 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 1, 12, 0, 0)
    )
    ev2 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 2, 12, 0, 0)
    )
    _add_observation(db, event_id=ev1.id, label="lion", max_n=1)
    _add_observation(db, event_id=ev2.id, label="lion", max_n=2)
    _add_observation(db, event_id=ev1.id, label="person", category="person", max_n=1)

    resp = client.get(f"/api/projects/{p.id}/independent-event-stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["labels"] == [{"label": "lion", "count": 2}]
    assert data["total"] == 2


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
    resp = client.post(
        "/api/projects", json={"name": "bs-default", "timezone": "UTC"}
    )
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


def test_reprocess_accepts_folder_run_projects(client, db):
    """The folder-run Labels step's analysis panel PATCHes the project
    and reprocesses through the same endpoints as the Settings page;
    neither may gate on project mode."""
    p = make_project(db, mode="folder_run")

    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"independence_interval": 900, "taxonomic_rollup": False},
    )
    assert resp.status_code == 200
    assert resp.json()["independence_interval"] == 900

    with patch("app.api.routers.projects.ws_manager"):
        resp = client.post(f"/api/projects/{p.id}/reprocess")
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_re_embed_accepts_min_confidence_override(client, db):
    """The labels grid's unprocessed-range banner backfills embeddings
    below the classification gate; the override rides on the job
    payload."""
    from app.models import Job

    p = make_project(db, embedding_model_id="DINOV2-VITB14")
    with patch("app.api.routers.projects.ws_manager"):
        resp = client.post(
            f"/api/projects/{p.id}/re-embed",
            json={"min_confidence": 0.02},
        )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = db.get(Job, job_id)
    assert job.payload["min_confidence"] == 0.02
