"""Tests for the project mode column and the mode filter on list / stats.

Folder runs are projects with mode='folder_run'. The user-facing
Research projects list must default to mode='research' so folder runs
stay invisible; the home screen explicitly passes ?mode=folder_run to
fetch the recents strip. The bulk stats endpoint must apply the same
filter so trap-nights and observation counts don't double-count files
that belong to a folder run.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.api.crud import project as crud_project
from tests.conftest import (
    make_deployment,
    make_file,
    make_project,
    make_site,
)


@pytest.fixture(autouse=True)
def mock_manifest_manager():
    """Patch ManifestManager so model validation in create paths is a no-op."""
    mock_mgr = MagicMock()
    mock_mgr.get_model.return_value = MagicMock(model_id="MD5A-0-0")
    with patch("app.ml.manifest_manager.ManifestManager", return_value=mock_mgr):
        yield mock_mgr


def test_existing_rows_default_to_research(db):
    """make_project() does not set `mode`; the ORM default fills it."""
    p = make_project(db)
    assert p.mode == "research"


def test_list_endpoint_defaults_to_research_only(client, db):
    make_project(db, name="research-a", mode="research")
    make_project(db, name="research-b", mode="research")
    make_project(db, name="folder-x", mode="folder_run")

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    names = sorted(p["name"] for p in resp.json())
    assert names == ["research-a", "research-b"]


def test_list_endpoint_can_filter_to_folder_runs(client, db):
    make_project(db, name="research-a", mode="research")
    make_project(db, name="folder-x", mode="folder_run")
    make_project(db, name="folder-y", mode="folder_run")

    resp = client.get("/api/projects?mode=folder_run")
    assert resp.status_code == 200
    names = sorted(p["name"] for p in resp.json())
    assert names == ["folder-x", "folder-y"]


def test_list_endpoint_can_return_all_with_explicit_all(client, db):
    make_project(db, name="research-a", mode="research")
    make_project(db, name="folder-x", mode="folder_run")

    resp = client.get("/api/projects?mode=all")
    assert resp.status_code == 200
    names = sorted(p["name"] for p in resp.json())
    assert names == ["folder-x", "research-a"]


def test_list_response_carries_mode_field(client, db):
    make_project(db, name="folder-x", mode="folder_run")
    resp = client.get("/api/projects?mode=folder_run")
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["mode"] == "folder_run"


def test_list_endpoint_rejects_invalid_mode(client):
    resp = client.get("/api/projects?mode=garbage")
    assert resp.status_code == 422


def test_bulk_stats_filter_excludes_folder_runs(db):
    """get_all_projects_stats(mode='research') must skip folder-run rows."""
    research = make_project(db, name="research", mode="research")
    folder = make_project(db, name="folder", mode="folder_run")

    research_site = make_site(db, project_id=research.id)
    make_deployment(db, site_id=research_site.id)
    folder_dep = make_deployment(db, project_id=folder.id)
    make_file(db, deployment_id=folder_dep.id)

    stats = crud_project.get_all_projects_stats(db, mode="research")

    assert research.id in stats
    assert folder.id not in stats


def test_bulk_stats_filter_can_target_folder_runs(db):
    research = make_project(db, name="research", mode="research")
    folder = make_project(db, name="folder", mode="folder_run")
    folder_dep = make_deployment(db, project_id=folder.id)
    make_file(db, deployment_id=folder_dep.id)

    stats = crud_project.get_all_projects_stats(db, mode="folder_run")

    assert folder.id in stats
    assert research.id not in stats
    assert stats[folder.id]["deployment_count"] == 1
    assert stats[folder.id]["file_count"] == 1


def test_bulk_stats_filter_none_returns_both(db):
    research = make_project(db, name="research", mode="research")
    folder = make_project(db, name="folder", mode="folder_run")
    make_deployment(db, project_id=folder.id)
    research_site = make_site(db, project_id=research.id)
    make_deployment(db, site_id=research_site.id)

    stats = crud_project.get_all_projects_stats(db, mode=None)

    assert research.id in stats
    assert folder.id in stats


def test_create_project_defaults_mode_to_research(client):
    resp = client.post(
        "/api/projects",
        json={"name": "fresh-create", "timezone": "UTC"},
    )
    assert resp.status_code == 201
    assert resp.json()["mode"] == "research"


def test_create_project_accepts_folder_run_mode(client):
    resp = client.post(
        "/api/projects",
        json={"name": "from-folder-run", "timezone": "UTC", "mode": "folder_run"},
    )
    assert resp.status_code == 201
    assert resp.json()["mode"] == "folder_run"


def test_promote_via_update_flips_mode_in_place(client, db):
    p = make_project(db, name="to-promote", mode="folder_run")
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"mode": "research"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "research"
