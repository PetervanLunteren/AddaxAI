"""Tests for the /api/files endpoints."""

import tempfile
from pathlib import Path

from tests.conftest import (
    make_deployment,
    make_file,
    make_project,
    make_site,
)


def _setup_deployment(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    return make_deployment(db, site_id=s.id), p


def test_list_files_empty(client):
    resp = client.get("/api/files")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_files_by_deployment(client, db):
    d, _ = _setup_deployment(db)
    make_file(db, deployment_id=d.id)
    resp = client.get(f"/api/files?deployment_id={d.id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_file(client, db):
    d, _ = _setup_deployment(db)
    f = make_file(db, deployment_id=d.id)
    resp = client.get(f"/api/files/{f.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == f.id


def test_get_file_not_found(client):
    resp = client.get("/api/files/nonexistent")
    assert resp.status_code == 404


def test_update_file_verified(client, db):
    d, _ = _setup_deployment(db)
    f = make_file(db, deployment_id=d.id)
    resp = client.patch(f"/api/files/{f.id}", json={"verified": True})
    assert resp.status_code == 200
    assert resp.json()["verified"] is True


def test_get_file_image_success(client, db):
    d, _ = _setup_deployment(db)
    # Create a real temp JPEG file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        # Minimal JPEG header
        tmp.write(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        tmp_path = tmp.name
    f = make_file(db, deployment_id=d.id, file_path=tmp_path, file_format="jpg")
    resp = client.get(f"/api/files/{f.id}/image")
    assert resp.status_code == 200
    Path(tmp_path).unlink(missing_ok=True)


def test_get_file_image_file_missing_on_disk(client, db):
    d, _ = _setup_deployment(db)
    f = make_file(db, deployment_id=d.id, file_path="/nonexistent/photo.jpg")
    resp = client.get(f"/api/files/{f.id}/image")
    assert resp.status_code == 404


def test_get_file_image_not_in_db(client):
    resp = client.get("/api/files/nonexistent/image")
    assert resp.status_code == 404


def test_get_observation_type_stats(client, db):
    d, p = _setup_deployment(db)
    resp = client.get(f"/api/files/stats/observation-types?project_id={p.id}")
    assert resp.status_code == 200
