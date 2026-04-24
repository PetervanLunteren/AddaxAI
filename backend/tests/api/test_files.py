"""Tests for the /api/files endpoints."""

import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from app.models import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
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


# ── Files verify tab endpoints ─────────────────────────────────────────────


def _setup_verify_fixture(db):
    """Create a project, a deployment, and a mix of images/videos/frames.

    Returns (project, deployment, image, video, frame).
    Image is unverified; video has a best frame with one detection at 0.95.
    The frame row (file_type="frame") must not appear in the grid.
    """
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)

    image = make_file(
        db,
        deployment_id=d.id,
        file_type="image",
        captured_at_local=datetime(2024, 6, 1, 12, 0, 0),
    )
    video = make_file(
        db,
        deployment_id=d.id,
        file_type="video",
        file_format="mp4",
        captured_at_local=datetime(2024, 6, 2, 12, 0, 0),
        best_frame_number=3,
        best_frame_path="/fake/video/frame003.jpg",
    )
    frame = make_file(
        db,
        deployment_id=d.id,
        file_type="frame",
        captured_at_local=datetime(2024, 6, 2, 12, 0, 0),
        source_video_id=video.id,
        source_frame_number=3,
    )
    make_detection(db, file_id=frame.id, confidence=0.95)
    make_detection(db, file_id=image.id, confidence=0.9)
    db.commit()
    return p, d, image, video, frame


def test_list_for_verify_excludes_frame_rows(client, db):
    p, _, image, video, frame = _setup_verify_fixture(db)
    resp = client.get(f"/api/files/list-for-verify?project_id={p.id}")
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()]
    assert image.id in ids
    assert video.id in ids
    assert frame.id not in ids


def test_list_for_verify_verification_filter(client, db):
    p, _, image, video, _ = _setup_verify_fixture(db)
    # Verify the image
    client.patch(f"/api/files/{image.id}", json={"verified": True})

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&verification=unverified"
    )
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()]
    assert image.id not in ids
    assert video.id in ids

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&verification=verified"
    )
    ids = [row["id"] for row in resp.json()]
    assert image.id in ids
    assert video.id not in ids


def test_list_for_verify_video_inherits_frame_detections(client, db):
    p, _, _, video, _ = _setup_verify_fixture(db)
    resp = client.get(f"/api/files/list-for-verify?project_id={p.id}")
    rows = {row["id"]: row for row in resp.json()}
    video_row = rows[video.id]
    assert len(video_row["detections"]) == 1
    assert video_row["detections"][0]["confidence"] == 0.95


def test_count_for_verify_matches_list(client, db):
    p, _, _, _, _ = _setup_verify_fixture(db)
    list_resp = client.get(f"/api/files/list-for-verify?project_id={p.id}")
    count_resp = client.get(f"/api/files/count-for-verify?project_id={p.id}")
    assert count_resp.status_code == 200
    assert count_resp.json()["count"] == len(list_resp.json())


def test_verification_stats(client, db):
    p, _, image, _, _ = _setup_verify_fixture(db)
    client.patch(f"/api/files/{image.id}", json={"verified": True})
    resp = client.get(f"/api/files/verification-stats?project_id={p.id}")
    assert resp.status_code == 200
    data = resp.json()
    # Two media items (image + video); one verified.
    assert data["total_files"] == 2
    assert data["verified_files"] == 1


def test_adjacent_next_unverified(client, db):
    p, _, image, video, _ = _setup_verify_fixture(db)
    # Order is captured_at DESC: video (Jun 2) then image (Jun 1).
    # From the video, the next (older) unverified file should be the image.
    resp = client.get(
        f"/api/files/{video.id}/adjacent?project_id={p.id}"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["previous_id"] is None
    assert data["next_id"] == image.id
    assert data["next_unverified_id"] == image.id


def test_label_tree_count_by_file(client, db):
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    # Build a tiny taxonomy: one species "wolf" under mammalia/carnivora/canidae/canis
    cls_model_id = p.classification_model_id or str(uuid.uuid4())
    if not p.classification_model_id:
        # Ensure taxonomy lookup works even without a real model.
        p.classification_model_id = cls_model_id
    tax = LabelTaxonomy(
        id=str(uuid.uuid4()),
        classification_model_id=cls_model_id,
        name="wolf",
        taxon_class="mammalia",
        taxon_order="carnivora",
        taxon_family="canidae",
        taxon_genus="canis",
        taxon_species="lupus",
        level="species",
    )
    db.add(tax)
    db.flush()

    # Image 1 with two wolf detections
    img1 = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(
        db, file_id=img1.id, confidence=0.9,
        label="wolf", label_taxonomy_id=tax.id,
    )
    make_detection(
        db, file_id=img1.id, confidence=0.8,
        label="wolf", label_taxonomy_id=tax.id,
    )
    # Image 2 with one wolf detection
    img2 = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(
        db, file_id=img2.id, confidence=0.95,
        label="wolf", label_taxonomy_id=tax.id,
    )
    db.commit()

    resp = client.get(
        f"/api/events/label-tree?project_id={p.id}&count_by=file"
    )
    assert resp.status_code == 200
    body = resp.json()
    # Two distinct files; three detections.
    counts = body["label_event_counts"]
    assert counts.get("wolf") == 2
    assert body["count_unit"] == "file"


def test_label_tree_rejects_invalid_count_by(client, db):
    p = make_project(db)
    resp = client.get(
        f"/api/events/label-tree?project_id={p.id}&count_by=bogus"
    )
    assert resp.status_code == 400
