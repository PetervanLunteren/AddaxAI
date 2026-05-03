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


def test_update_file_flagged_sets_timestamp(client, db):
    d, _ = _setup_deployment(db)
    f = make_file(db, deployment_id=d.id)
    resp = client.patch(f"/api/files/{f.id}", json={"flagged": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["flagged"] is True
    assert body["flagged_at_utc"] is not None
    resp = client.patch(f"/api/files/{f.id}", json={"flagged": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["flagged"] is False
    assert body["flagged_at_utc"] is None


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
    Image is unverified with its own detection. Frame is unverified with
    its own detection. The video row holds mp4 metadata only and is not
    listed in the Images grid.
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


def test_list_for_verify_excludes_video_rows(client, db):
    p, _, image, video, frame = _setup_verify_fixture(db)
    resp = client.get(f"/api/files/list-for-verify?project_id={p.id}")
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()]
    assert image.id in ids
    assert frame.id in ids
    assert video.id not in ids


def test_list_for_verify_verification_filter(client, db):
    p, _, image, _, frame = _setup_verify_fixture(db)
    # Verify the image
    client.patch(f"/api/files/{image.id}", json={"verified": True})

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&verification=unverified"
    )
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()]
    assert image.id not in ids
    assert frame.id in ids

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&verification=verified"
    )
    ids = [row["id"] for row in resp.json()]
    assert image.id in ids
    assert frame.id not in ids


def test_list_for_verify_includes_frame_rows(client, db):
    p, _, _, _, frame = _setup_verify_fixture(db)
    resp = client.get(f"/api/files/list-for-verify?project_id={p.id}")
    rows = {row["id"]: row for row in resp.json()}
    frame_row = rows[frame.id]
    assert len(frame_row["detections"]) == 1
    assert frame_row["detections"][0]["confidence"] == 0.95


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
    # Two stills in the grid (image + frame); one verified.
    assert data["total_files"] == 2
    assert data["verified_files"] == 1


def test_adjacent_next_unverified(client, db):
    p, _, image, _, frame = _setup_verify_fixture(db)
    # Order is captured_at DESC: frame (Jun 2) then image (Jun 1).
    # From the frame, the next (older) unverified file should be the image.
    resp = client.get(
        f"/api/files/{frame.id}/adjacent?project_id={p.id}"
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


def test_list_for_verify_flagged_filter(client, db):
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    flagged = make_file(db, deployment_id=d.id, file_type="image")
    unflagged = make_file(db, deployment_id=d.id, file_type="image")
    # Files verify tab filters to files with at least one visible detection.
    make_detection(db, file_id=flagged.id, confidence=0.9)
    make_detection(db, file_id=unflagged.id, confidence=0.9)
    db.commit()
    client.patch(f"/api/files/{flagged.id}", json={"flagged": True})

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&flagged=flagged"
    )
    ids = [row["id"] for row in resp.json()]
    assert flagged.id in ids
    assert unflagged.id not in ids

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&flagged=not_flagged"
    )
    ids = [row["id"] for row in resp.json()]
    assert flagged.id not in ids
    assert unflagged.id in ids


def test_list_for_verify_favorited_filter(client, db):
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    fav = make_file(db, deployment_id=d.id, file_type="image")
    not_fav = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(db, file_id=fav.id, confidence=0.9)
    make_detection(db, file_id=not_fav.id, confidence=0.9)
    db.commit()
    client.patch(f"/api/files/{fav.id}", json={"favorited": True})

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&favorited=favorited"
    )
    ids = [row["id"] for row in resp.json()]
    assert fav.id in ids
    assert not_fav.id not in ids


def test_list_for_verify_label_confidence_range(client, db):
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    high = make_file(db, deployment_id=d.id, file_type="image")
    low = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(
        db, file_id=high.id, confidence=0.9, label_confidence=0.9,
    )
    make_detection(
        db, file_id=low.id, confidence=0.9, label_confidence=0.3,
    )
    db.commit()

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&min_label_confidence=0.5"
    )
    ids = [row["id"] for row in resp.json()]
    assert high.id in ids
    assert low.id not in ids


def test_list_for_verify_label_confidence_excludes_null(client, db):
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    classified = make_file(db, deployment_id=d.id, file_type="image")
    unclassified = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(
        db, file_id=classified.id, confidence=0.9, label_confidence=0.3,
    )
    # Unclassified detection: confidence high but no label_confidence.
    make_detection(
        db, file_id=unclassified.id, confidence=0.9, label_confidence=None,
    )
    db.commit()

    # Active filter (any of the cls bounds set) excludes NULL detections,
    # even when the bound itself permits low values.
    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&min_label_confidence=0.0"
    )
    ids = [row["id"] for row in resp.json()]
    assert classified.id in ids
    assert unclassified.id not in ids


def test_list_for_verify_empty_filter(client, db):
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    animal = make_file(
        db, deployment_id=d.id, file_type="image",
        observation_type="animal",
    )
    blank = make_file(
        db, deployment_id=d.id, file_type="image",
        observation_type="blank",
    )
    # Both files need a visible detection to pass the threshold filter
    # used by list-for-verify (we are testing the empty filter, not the
    # threshold filter).
    make_detection(db, file_id=animal.id, confidence=0.9)
    make_detection(db, file_id=blank.id, confidence=0.9)
    db.commit()

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&empty=show_only"
    )
    ids = [row["id"] for row in resp.json()]
    assert blank.id in ids
    assert animal.id not in ids

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&empty=hide"
    )
    ids = [row["id"] for row in resp.json()]
    assert animal.id in ids
    assert blank.id not in ids


def test_file_summary_includes_flagged(client, db):
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(db, file_id=f.id, confidence=0.9)
    db.commit()
    client.patch(f"/api/files/{f.id}", json={"flagged": True})

    resp = client.get(f"/api/files/list-for-verify?project_id={p.id}")
    rows = {row["id"]: row for row in resp.json()}
    assert rows[f.id]["flagged"] is True


def test_list_for_verify_user_min_does_not_or_verify(client, db):
    """User-set min_confidence is applied LITERALLY.

    The project floor uses `(confidence >= floor OR verified)` to keep
    verified low-confidence detections visible. The user's slider must
    NOT inherit that OR-verified clause — otherwise narrowing the slider
    to 0.95-1.0 would still surface a verified detection at 0.87.
    """
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    verified_low = make_file(db, deployment_id=d.id, file_type="image")
    high = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(
        db, file_id=verified_low.id, confidence=0.87, verified=True,
    )
    make_detection(db, file_id=high.id, confidence=0.97)
    db.commit()

    # Floor alone (default request): both files visible. Verified-low
    # passes via the OR clause; high passes literally.
    resp = client.get(f"/api/files/list-for-verify?project_id={p.id}")
    ids = [row["id"] for row in resp.json()]
    assert verified_low.id in ids
    assert high.id in ids

    # User narrows: 0.95-1.0. The literal min excludes 0.87 even though
    # the detection is verified.
    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}"
        "&min_confidence=0.95&max_confidence=1.0"
    )
    ids = [row["id"] for row in resp.json()]
    assert high.id in ids
    assert verified_low.id not in ids


def _setup_three_files(db):
    """Three files with distinct timestamps for sort tests."""
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    older = make_file(
        db, deployment_id=d.id, file_type="image",
        captured_at_local=datetime(2024, 1, 1, 12, 0, 0),
    )
    middle = make_file(
        db, deployment_id=d.id, file_type="image",
        captured_at_local=datetime(2024, 2, 1, 12, 0, 0),
    )
    newer = make_file(
        db, deployment_id=d.id, file_type="image",
        captured_at_local=datetime(2024, 3, 1, 12, 0, 0),
    )
    make_detection(db, file_id=older.id, confidence=0.9)
    make_detection(db, file_id=middle.id, confidence=0.9)
    make_detection(db, file_id=newer.id, confidence=0.9)
    db.commit()
    return p, older, middle, newer


def test_files_sort_oldest(client, db):
    p, older, middle, newer = _setup_three_files(db)

    resp = client.get(f"/api/files/list-for-verify?project_id={p.id}&sort=oldest")
    ids = [row["id"] for row in resp.json()]
    assert ids == [older.id, middle.id, newer.id]


def test_files_sort_random_stable_with_seed(client, db):
    p, *_ = _setup_three_files(db)

    a = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&sort=random&seed=42"
    ).json()
    b = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&sort=random&seed=42"
    ).json()
    assert [r["id"] for r in a] == [r["id"] for r in b]

    # Three files have 6 permutations, so any specific seed pair has a
    # 1/6 chance of producing the same order. Probe seeds until we find
    # one that genuinely differs from seed 42, rather than asserting a
    # hard-coded pair and hoping for the best.
    base_ids = [r["id"] for r in a]
    for trial in range(43, 100):
        other = client.get(
            f"/api/files/list-for-verify?project_id={p.id}&sort=random&seed={trial}",
        ).json()
        if [r["id"] for r in other] != base_ids:
            break
    else:
        raise AssertionError(
            "Seeds 43-99 all produced the same order as seed 42; "
            "seeded_hash UDF is likely broken.",
        )


def test_files_sort_random_paginates_consistently(client, db):
    p, *_ = _setup_three_files(db)

    page0 = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&sort=random&seed=7&limit=2&skip=0"
    ).json()
    page1 = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&sort=random&seed=7&limit=2&skip=2"
    ).json()
    full = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&sort=random&seed=7"
    ).json()
    paginated = [r["id"] for r in page0] + [r["id"] for r in page1]
    assert paginated == [r["id"] for r in full]


def test_files_sort_cls_low_pushes_nulls_last(client, db):
    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    low = make_file(db, deployment_id=d.id, file_type="image")
    high = make_file(db, deployment_id=d.id, file_type="image")
    null = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(db, file_id=low.id, confidence=0.9, label_confidence=0.2)
    make_detection(db, file_id=high.id, confidence=0.9, label_confidence=0.7)
    make_detection(db, file_id=null.id, confidence=0.9, label_confidence=None)
    db.commit()

    resp = client.get(f"/api/files/list-for-verify?project_id={p.id}&sort=cls_low")
    ids = [row["id"] for row in resp.json()]
    assert ids == [low.id, high.id, null.id]


def test_files_adjacent_respects_sort(client, db):
    p, older, middle, newer = _setup_three_files(db)

    # In oldest-first display, opening "older" should show "middle" as next.
    resp = client.get(
        f"/api/files/{older.id}/adjacent?project_id={p.id}&sort=oldest"
    ).json()
    assert resp["next_id"] == middle.id
    assert resp["previous_id"] is None

    # In newest-first display (default), opening "older" should show no next.
    resp = client.get(
        f"/api/files/{older.id}/adjacent?project_id={p.id}"
    ).json()
    assert resp["next_id"] is None
    assert resp["previous_id"] == middle.id


def test_files_sort_invalid_value_returns_400(client, db):
    p = make_project(db, detection_threshold=0.5)
    db.commit()

    resp = client.get(
        f"/api/files/list-for-verify?project_id={p.id}&sort=bogus"
    )
    assert resp.status_code == 400
