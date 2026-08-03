"""Tests for the /api/deployments endpoints."""

import os
from datetime import date, datetime
from unittest.mock import patch

import pytest

from app.models.event_observation import EventObservation
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def test_list_deployments_empty(client):
    resp = client.get("/api/deployments")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_deployment(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    resp = client.post("/api/deployments", json={
        "project_id": p.id,
        "site_id": s.id,
        "start_date_local": "2024-01-01",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["site_id"] == s.id
    assert data["project_id"] == p.id


def test_create_deployment_without_site(client, db):
    p = make_project(db)
    resp = client.post("/api/deployments", json={
        "project_id": p.id,
        "start_date_local": "2024-01-01",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["site_id"] is None
    assert data["project_id"] == p.id


def test_create_deployment_invalid_site(client, db):
    p = make_project(db)
    resp = client.post("/api/deployments", json={
        "project_id": p.id,
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
        "mtime_start_date": None,
        "mtime_end_date": None,
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


# ---------------------------------------------------------------------------
# /info endpoint
# ---------------------------------------------------------------------------


def _build_info_fixture(db):
    """Project + site + deployment with mixed images/videos and a few
    classified, verified, and below-threshold detections."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id, name="Camp A")
    dep = make_deployment(
        db,
        site_id=site.id,
        folder_path="/tmp/demo",
        start_date_local=date(2024, 6, 1),
        end_date_local=date(2024, 6, 30),
    )
    # 3 images, 2 videos.
    image_files = [
        make_file(
            db,
            deployment_id=dep.id,
            file_type="image",
            file_format="jpg",
            captured_at_local=datetime(2024, 6, 15, 8, i),
        )
        for i in range(3)
    ]
    for i in range(2):
        make_file(
            db,
            deployment_id=dep.id,
            file_type="video",
            file_format="mp4",
            captured_at_local=datetime(2024, 6, 16, 9, i),
        )
    # Detections: one well above threshold, one verified below threshold,
    # one unverified below threshold (should NOT be averaged in).
    make_detection(
        db, file_id=image_files[0].id, confidence=0.9,
        label="lion", label_confidence=0.8,
    )
    make_detection(
        db, file_id=image_files[1].id, confidence=0.2,
        verified=True, label="leopard", label_confidence=0.4,
    )
    make_detection(
        db, file_id=image_files[2].id, confidence=0.1,  # below threshold, dropped
    )
    # One event with one observation of MaxN=3. Pass files_verified=[]
    # so the helper does not auto-create an extra file (we have the
    # explicit images above and want the file count assertions to be
    # exact).
    event = make_event_with_files(
        db,
        deployment_id=dep.id,
        event_start_local=datetime(2024, 6, 15, 8, 0),
        files_verified=[],
    )
    db.add(
        EventObservation(
            event_id=event.id,
            label="lion",
            label_taxonomy_id=None,
            category="animal",
            max_n=3,
        )
    )
    db.flush()
    return dep


def test_deployment_info_happy_path(client, db):
    dep = _build_info_fixture(db)
    resp = client.get(f"/api/deployments/{dep.id}/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deployment_id"] == dep.id
    assert data["folder_path"] == "/tmp/demo"
    assert data["site_name"] == "Camp A"
    assert data["site_id"] == dep.site_id
    assert data["files"] == {"total": 5, "images": 3, "videos": 2}
    assert data["event_count"] == 1
    assert data["observation_count"] == 3
    # One animal observation with MaxN=3 from _build_info_fixture.
    assert data["detection_categories"]["animal"] == 3
    assert data["detection_categories"]["person"] == 0
    assert data["detection_categories"]["vehicle"] == 0
    assert data["detection_categories"]["empty"] == 0
    # Top species block: single lion entry with count=3.
    assert data["top_species"] == [
        {"label": "lion", "common_name": None, "scientific_name": None, "count": 3}
    ]
    # Trap nights is folder-aware: sum of (max - min + 1) per folder over
    # files. Here all 5 files live under /fake/ and span June 15
    # to June 16, so trap_nights = 2 regardless of the manually-set
    # Deployment.start_date_local / end_date_local. Rate = 3 / 2 * 100.
    assert data["trap_nights"] == 2
    assert data["observation_rate_per_100_trap_nights"] == pytest.approx(150.0)
    # Verification: no files verified in the fixture.
    assert data["verification"] == {"verified": 0, "total": 5}
    # Total size is 0 because test factory doesn't set size_bytes.
    assert data["total_size_bytes"] == 0
    # Threshold-with-verified filter should include the verified 0.2
    # and the unverified 0.9 but NOT the unverified 0.1.
    # Mean detection = (0.9 + 0.2) / 2 = 0.55
    assert data["mean_detection_confidence"] == pytest.approx(0.55)
    # Classification mean = (0.8 + 0.4) / 2 = 0.6
    assert data["mean_classification_confidence"] == pytest.approx(0.6)
    assert data["first_captured_at_local"].startswith("2024-06-15T08:00")
    assert data["last_captured_at_local"].startswith("2024-06-16T09:01")


def test_deployment_info_not_found(client):
    resp = client.get("/api/deployments/nonexistent/info")
    assert resp.status_code == 404


def test_deployment_info_empty_deployment(client, db):
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id, name="Empty Camp")
    dep = make_deployment(db, site_id=site.id, folder_path=None)

    resp = client.get(f"/api/deployments/{dep.id}/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["files"] == {"total": 0, "images": 0, "videos": 0}
    assert data["event_count"] == 0
    assert data["observation_count"] == 0
    assert data["mean_detection_confidence"] is None
    assert data["mean_classification_confidence"] is None
    assert data["first_captured_at_local"] is None
    assert data["last_captured_at_local"] is None


def test_deployment_info_images_only(client, db):
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id, name="Images Camp")
    dep = make_deployment(db, site_id=site.id)
    make_file(db, deployment_id=dep.id, file_type="image", file_format="jpg")
    make_file(db, deployment_id=dep.id, file_type="image", file_format="jpg")
    resp = client.get(f"/api/deployments/{dep.id}/info")
    data = resp.json()
    assert data["files"] == {"total": 2, "images": 2, "videos": 0}


def test_deployment_info_videos_only(client, db):
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id, name="Videos Camp")
    dep = make_deployment(db, site_id=site.id)
    make_file(db, deployment_id=dep.id, file_type="video", file_format="mp4")
    resp = client.get(f"/api/deployments/{dep.id}/info")
    data = resp.json()
    assert data["files"] == {"total": 1, "images": 0, "videos": 1}


def test_deployment_info_no_classifications(client, db):
    """A detection with no `label_confidence` should produce a null
    classification mean, even when the detection mean is populated."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id, name="Det Only")
    dep = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=dep.id, file_type="image", file_format="jpg")
    make_detection(db, file_id=f.id, confidence=0.8)  # no label_confidence
    resp = client.get(f"/api/deployments/{dep.id}/info")
    data = resp.json()
    assert data["mean_detection_confidence"] == 0.8
    assert data["mean_classification_confidence"] is None


def test_deployment_info_verified_below_threshold_is_counted(client, db):
    """A verified detection with confidence < threshold must still count
    in the mean because of the verified override rule."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id, name="Camp V")
    dep = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=dep.id, file_type="image", file_format="jpg")
    # Only detection is below threshold but verified.
    make_detection(db, file_id=f.id, confidence=0.3, verified=True)
    resp = client.get(f"/api/deployments/{dep.id}/info")
    data = resp.json()
    assert data["mean_detection_confidence"] == 0.3


# ── file-datetime probe: the opt-in mtime fallback ────────────────────────


def _jpeg_with_mtime(folder, name, when):
    """Write a tiny EXIF-less JPEG and set its modification time."""
    from PIL import Image

    path = folder / name
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="JPEG")
    os.utime(path, (when.timestamp(), when.timestamp()))
    return path


def test_file_datetime_without_fallback_returns_null(client, tmp_path):
    """Default behaviour: no capture date means no date, mtime ignored."""
    _jpeg_with_mtime(tmp_path, "a.jpg", datetime(2024, 4, 7, 15, 55, 26))

    resp = client.get(
        "/api/deployments/file-datetime"
        f"?folder={tmp_path}&file=a.jpg"
    )

    assert resp.status_code == 200
    assert resp.json()["file_datetime"] is None


def test_file_datetime_with_fallback_returns_mtime(client, tmp_path):
    """Without this the Adjust-dates modal would show "unknown" for every
    file in a folder that opted in, so the offset could never be set."""
    _jpeg_with_mtime(tmp_path, "a.jpg", datetime(2024, 4, 7, 15, 55, 26))

    resp = client.get(
        "/api/deployments/file-datetime"
        f"?folder={tmp_path}&file=a.jpg&use_file_mtime_fallback=true"
    )

    assert resp.status_code == 200
    assert resp.json()["file_datetime"] == datetime(2024, 4, 7, 15, 55, 26).isoformat()


# ── Relink refusal ───────────────────────────────────────────────────────
# A refused relink used to leave no trace anywhere: the endpoint answers
# 200 with the reasons in the body, the UI rendered only how many samples
# failed, and nothing was logged. A user (and a maintainer reading
# backend.log afterwards) could not tell which file was wrong or why.


def _deployment_with_files_on_disk(db, folder, names_and_sizes):
    """A deployment whose File rows point at real files under `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    project = make_project(db)
    deployment = make_deployment(
        db, project_id=project.id, folder_path=str(folder)
    )
    for name, size in names_and_sizes:
        path = folder / name
        path.write_bytes(b"x" * size)
        make_file(
            db,
            deployment_id=deployment.id,
            file_path=str(path),
            size_bytes=size,
        )
    db.commit()
    return deployment


def test_relink_refusal_reports_the_reason_per_file(db, tmp_path, caplog):
    """A wrong folder is refused, and says which file and why."""
    from app.api.crud.deployment import relink_deployment

    old = tmp_path / "old"
    deployment = _deployment_with_files_on_disk(db, old, [("a.jpg", 100)])

    # Same filename, different contents: the lookalike case the size check
    # exists to catch.
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / "a.jpg").write_bytes(b"y" * 999)

    with caplog.at_level("WARNING"):
        result = relink_deployment(db, deployment.id, str(wrong))

    assert result.success is False
    assert result.files_rewritten == 0
    assert result.verify_result is not None
    assert len(result.verify_result.mismatches) == 1
    reason = result.verify_result.mismatches[0]
    assert "Size mismatch" in reason
    assert "a.jpg" in reason
    assert "expected 100" in reason and "got 999" in reason

    # The reason reaches the log, not just the response body.
    assert "Relink refused" in caplog.text
    assert reason in caplog.text


def test_relink_refusal_names_a_missing_file(db, tmp_path, caplog):
    """The other failure mode: the file simply is not there."""
    from app.api.crud.deployment import relink_deployment

    old = tmp_path / "old"
    deployment = _deployment_with_files_on_disk(db, old, [("a.jpg", 100)])
    empty = tmp_path / "empty"
    empty.mkdir()

    with caplog.at_level("WARNING"):
        result = relink_deployment(db, deployment.id, str(empty))

    assert result.success is False
    assert any("Missing" in m and "a.jpg" in m for m in result.verify_result.mismatches)
    assert "Relink refused" in caplog.text


def test_relink_refusal_changes_nothing_in_the_database(db, tmp_path):
    """The refusal path must not half-apply. Files keep their old paths and
    the deployment keeps its folder, so retrying with the right folder is
    always possible."""
    from app.api.crud.deployment import relink_deployment

    old = tmp_path / "old"
    deployment = _deployment_with_files_on_disk(db, old, [("a.jpg", 100)])
    original_paths = sorted(f.file_path for f in deployment.files)
    empty = tmp_path / "empty"
    empty.mkdir()

    relink_deployment(db, deployment.id, str(empty))
    db.expire_all()

    assert deployment.folder_path == str(old)
    assert sorted(f.file_path for f in deployment.files) == original_paths


def test_relink_succeeds_when_the_folder_really_holds_the_files(db, tmp_path):
    """The happy path, so the tests above cannot pass by refusing everything."""
    from app.api.crud.deployment import relink_deployment

    old = tmp_path / "old"
    deployment = _deployment_with_files_on_disk(db, old, [("a.jpg", 100)])

    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "a.jpg").write_bytes(b"x" * 100)

    result = relink_deployment(db, deployment.id, str(moved))

    assert result.success is True
    assert result.files_rewritten == 1
    assert deployment.folder_path == str(moved)
    assert deployment.folder_status == "valid"
