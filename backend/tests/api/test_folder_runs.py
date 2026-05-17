"""Tests for /api/folder-runs.

The endpoint orchestrates a project (mode='folder_run') and a queue
entry. These tests pin the contract: create returns both, the project
has the right mode, the queue entry has no site, the step state
round-trips through GET, and lookups for non-folder-run project IDs
404 cleanly.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_project


@pytest.fixture(autouse=True)
def mock_manifest_manager():
    """The folder-run create path runs through the regular project
    create flow, which validates models against the on-disk manifest.
    Stub it out so the test does not need the real model directory."""
    mock_mgr = MagicMock()
    mock_mgr.get_model.return_value = MagicMock(model_id="MD5A-0-0")
    with patch("app.ml.manifest_manager.ManifestManager", return_value=mock_mgr):
        yield mock_mgr


def test_create_folder_run_auto_name(client):
    resp = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/Volumes/Photos/Kruger_April",
            "image_count": 412,
            "video_count": 7,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["project"]["name"] == "Kruger_April"
    assert body["project"]["mode"] == "folder_run"
    assert body["project"]["timezone"] == "UTC"
    assert body["project"]["folder_run_state"] == {
        "step": "model",
        "source_folder": "/Volumes/Photos/Kruger_April",
    }
    assert body["step"] == "model"
    assert body["queue_entry"]["folder_path"] == "/Volumes/Photos/Kruger_April"
    assert body["queue_entry"]["site_id"] is None
    assert body["queue_entry"]["video_count"] == 7
    assert body["queue_entry"]["image_count"] == 412


def test_create_folder_run_explicit_name(client):
    resp = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/anything",
            "name": "my-test-run",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["project"]["name"] == "my-test-run"


def test_create_folder_run_rejects_duplicate_name(client):
    client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/a", "name": "dup-run"},
    )
    resp = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/b", "name": "dup-run"},
    )
    assert resp.status_code == 409


def test_same_source_folder_resumes_existing_run(client):
    """Legacy-AddaxAI behaviour: re-selecting an analysed folder
    returns the existing folder-run project rather than creating a
    new one. There is no "recent work" list in the UI; this is how
    users revisit their work."""
    first = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/resume-me"},
    ).json()
    first_id = first["project"]["id"]

    second = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/resume-me"},
    )
    assert second.status_code == 201
    assert second.json()["project"]["id"] == first_id


def test_resume_preserves_persisted_step(client):
    """If the user got to step `save` and then reopens the folder,
    they should land on `save`, not back at `folder`."""
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/resume-step"},
    ).json()
    run_id = created["project"]["id"]

    client.patch(
        f"/api/folder-runs/{run_id}/step", json={"step": "save"}
    )

    resumed = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/resume-step"},
    ).json()
    assert resumed["project"]["id"] == run_id
    assert resumed["step"] == "save"


def test_resume_ignores_explicit_name(client):
    """A second POST with a different explicit name on the same
    folder still resumes the existing run rather than failing on
    a duplicate name or creating a parallel project."""
    first = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/named-folder",
            "name": "first-attempt",
        },
    ).json()

    second = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/named-folder",
            "name": "second-attempt",
        },
    ).json()

    assert second["project"]["id"] == first["project"]["id"]
    # The original name is preserved; we are resuming, not renaming.
    assert second["project"]["name"] == "first-attempt"


def test_different_source_folder_creates_new_run(client):
    first = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/folder-a"},
    ).json()
    second = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/folder-b"},
    ).json()

    assert first["project"]["id"] != second["project"]["id"]


def test_get_folder_run(client):
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/get-test"},
    ).json()
    run_id = created["project"]["id"]

    resp = client.get(f"/api/folder-runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["project"]["id"] == run_id
    assert resp.json()["step"] == "model"


def test_get_folder_run_404_for_unknown(client):
    resp = client.get("/api/folder-runs/does-not-exist")
    assert resp.status_code == 404


def test_output_preview_endpoint_returns_counts(client, db):
    """End-to-end: create a run, manually attach a file + detection,
    and GET the preview. Confirms the response shape matches the
    Pydantic schema and the underlying compute is wired up."""
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
        make_project,
    )

    project = make_project(
        db, name="preview-endpoint", mode="folder_run"
    )
    dep = make_deployment(db, project_id=project.id)
    f = make_file(
        db,
        deployment_id=dep.id,
        observation_type="animal",
        size_bytes=1234,
    )
    make_detection(db, file_id=f.id, confidence=0.95, label="dog")

    resp = client.post(
        f"/api/folder-runs/{project.id}/output-preview", json={}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_files"] == 1
    assert data["image_count"] == 1
    assert data["video_count"] == 0
    assert data["total_bytes"] == 1234
    assert data["files_with_known_size"] == 1
    assert data["dropped_by_filter"] == 0
    assert data["in_scope_files"] == 1
    # Unmapped label → falls back to Other/<label> in the tree, and
    # to a single-segment "dog" entry in the flat bucket.
    assert data["by_taxonomic_tree"] == {"Other/dog": 1}
    assert data["by_flat"] == {"dog": 1}
    assert data["multi_species_files"] == 0


def test_output_preview_honours_excluded_label_ids(client, db):
    """Excluding the only species drops the file from in-scope."""
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
        make_project,
    )

    project = make_project(
        db, name="preview-filter", mode="folder_run"
    )
    dep = make_deployment(db, project_id=project.id)
    f = make_file(
        db,
        deployment_id=dep.id,
        observation_type="animal",
        size_bytes=1234,
    )
    make_detection(db, file_id=f.id, confidence=0.95, label="dog")

    resp = client.post(
        f"/api/folder-runs/{project.id}/output-preview",
        json={"excluded_label_ids": ["dog"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dropped_by_filter"] == 1
    assert data["in_scope_files"] == 0
    assert data["by_taxonomic_tree"] == {}


def test_output_preview_404_for_research_project(client, db):
    research = make_project(db, name="research-preview", mode="research")

    resp = client.post(
        f"/api/folder-runs/{research.id}/output-preview", json={}
    )
    assert resp.status_code == 404


def test_get_folder_run_404_for_research_project(client, db):
    """A research-project id must not resolve as a folder run, even if
    the caller knows the id. Prevents accidentally landing the stepper
    on a real project."""
    research = make_project(db, name="research-only", mode="research")

    resp = client.get(f"/api/folder-runs/{research.id}")
    assert resp.status_code == 404


def test_patch_step_persists_and_round_trips(client):
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/step-test"},
    ).json()
    run_id = created["project"]["id"]

    resp = client.patch(
        f"/api/folder-runs/{run_id}/step",
        json={"step": "model"},
    )
    assert resp.status_code == 200
    assert resp.json()["step"] == "model"

    follow_up = client.get(f"/api/folder-runs/{run_id}").json()
    assert follow_up["step"] == "model"
    # The other state keys (source_folder) survive the step update.
    assert (
        follow_up["project"]["folder_run_state"]["source_folder"]
        == "/tmp/step-test"
    )


def test_patch_step_rejects_unknown_step(client):
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/bad-step"},
    ).json()
    run_id = created["project"]["id"]

    resp = client.patch(
        f"/api/folder-runs/{run_id}/step",
        json={"step": "garbage"},
    )
    assert resp.status_code == 422


def test_folder_run_invisible_in_research_projects_list(client):
    client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/hidden"},
    )
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


def test_folder_run_visible_when_listing_by_mode(client):
    client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/listed", "name": "listed-run"},
    )
    resp = client.get("/api/projects?mode=folder_run")
    assert resp.status_code == 200
    names = [p["name"] for p in resp.json()]
    assert "listed-run" in names
