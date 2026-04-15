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
        "start_date_local": "2024-01-01",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["site_id"] == s.id


def test_create_deployment_invalid_site(client):
    resp = client.post("/api/deployments", json={
        "site_id": "nonexistent",
        "start_date_local": "2024-01-01",
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


def test_delete_deployment_cascades_artifacts_on_disk(client, db, tmp_path):
    """
    Deleting a deployment removes the project-scoped .addaxai folder on
    disk, and rolls up empty parent dirs so the .addaxai marker
    disappears entirely when the last project is gone.
    """
    p = make_project(db)
    s = make_site(db, project_id=p.id)

    deploy_dir = tmp_path / "deployment"
    deploy_dir.mkdir()
    artifacts = deploy_dir / ".addaxai" / "projects" / p.id
    artifacts.mkdir(parents=True)
    (artifacts / "results.json").write_text('{"images": []}')
    (artifacts / "video_frames").mkdir()
    (artifacts / "video_frames" / "frame000001.jpg").write_bytes(b"\x00")

    d = make_deployment(db, site_id=s.id, folder_path=str(deploy_dir))
    db.commit()

    resp = client.delete(f"/api/deployments/{d.id}")
    assert resp.status_code == 204

    # Project-scoped artifacts dir is gone, and the parent .addaxai is
    # gone too because there were no other projects in there.
    assert not artifacts.exists()
    assert not (deploy_dir / ".addaxai").exists()
    # The original deployment folder itself is untouched — only AddaxAI
    # state was removed, never the user's images/videos.
    assert deploy_dir.exists()


def test_delete_deployment_keeps_other_projects_artifacts(client, db, tmp_path):
    """
    If two projects analyzed the same physical folder, deleting one
    project's deployment leaves the other project's artifacts intact.
    """
    p1 = make_project(db)
    p2 = make_project(db)
    s1 = make_site(db, project_id=p1.id)

    deploy_dir = tmp_path / "shared_deployment"
    deploy_dir.mkdir()
    p1_artifacts = deploy_dir / ".addaxai" / "projects" / p1.id
    p2_artifacts = deploy_dir / ".addaxai" / "projects" / p2.id
    p1_artifacts.mkdir(parents=True)
    p2_artifacts.mkdir(parents=True)
    (p1_artifacts / "results.json").write_text("p1")
    (p2_artifacts / "results.json").write_text("p2")

    d1 = make_deployment(db, site_id=s1.id, folder_path=str(deploy_dir))
    db.commit()

    resp = client.delete(f"/api/deployments/{d1.id}")
    assert resp.status_code == 204

    assert not p1_artifacts.exists()
    assert p2_artifacts.exists()
    assert (p2_artifacts / "results.json").read_text() == "p2"
    # .addaxai/projects/ still has p2's subdir, so the marker stays.
    assert (deploy_dir / ".addaxai" / "projects").exists()


def test_delete_deployment_missing_folder_path_does_not_crash(client, db):
    """
    A deployment with folder_path=None (legacy / never-linked) should
    still delete cleanly without trying to scrub a nonexistent disk path.
    """
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path=None)
    db.commit()

    resp = client.delete(f"/api/deployments/{d.id}")
    assert resp.status_code == 204


def test_delete_deployment_unreadable_folder_swallowed(client, db, tmp_path):
    """
    A deployment whose folder_path doesn't exist on disk anymore (e.g.
    external drive unmounted) must still delete from the DB; the
    artifact cleanup is best-effort and never blocks.
    """
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    fake_path = tmp_path / "never_existed"
    d = make_deployment(db, site_id=s.id, folder_path=str(fake_path))
    db.commit()

    resp = client.delete(f"/api/deployments/{d.id}")
    assert resp.status_code == 204
    assert not fake_path.exists()  # we didn't accidentally create it


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
