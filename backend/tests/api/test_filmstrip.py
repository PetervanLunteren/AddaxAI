"""API tests for the on-demand video filmstrip endpoint."""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from tests.conftest import (  # noqa: E402
    make_deployment,
    make_file,
    make_project,
    make_site,
)


def test_filmstrip_returns_frames_for_video(client, db, tmp_path, make_video):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=20, fps=10, size=(160, 120))
    f = make_file(
        db,
        deployment_id=dep.id,
        file_type="video",
        file_format="mp4",
        file_path=str(video),
        frame_rate=10.0,
    )

    resp = client.get(f"/api/files/{f.id}/filmstrip")

    assert resp.status_code == 200
    frames = resp.json()["frames"]
    assert len(frames) > 0
    assert all(fr["image"].startswith("data:image/jpeg;base64,") for fr in frames)


def test_filmstrip_rejects_non_video(client, db):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=dep.id)  # default file_type="image"

    resp = client.get(f"/api/files/{f.id}/filmstrip")

    assert resp.status_code == 400


def test_filmstrip_missing_file_returns_404(client):
    resp = client.get("/api/files/does-not-exist/filmstrip")
    assert resp.status_code == 404
