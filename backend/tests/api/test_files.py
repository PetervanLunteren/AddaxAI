"""Tests for the /api/files endpoints."""

import tempfile
import uuid
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


def test_a_video_that_never_decoded_says_so_instead_of_serving_the_container(
    client, db
):
    """A video with no best frame must not fall through to its own bytes.

    `best_frame.py` skips a clip it cannot open, so `best_frame_number`
    and `best_frame_path` both stay NULL, and such a clip always lands in
    the Empties tab (no visible surface means no passing detection). Its
    tile then asks this endpoint for a picture. Serving the container
    answered `image/avi` for the full size and, for `size=thumb`, handed
    the file to PIL, which cannot open an AVI: an unhandled
    `UnidentifiedImageError` and a 500 per tile.

    The file exists on disk here on purpose. The existing missing-file
    test would pass against the old code too, since that path never got
    as far as opening anything.
    """
    d, _ = _setup_deployment(db)
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp.write(b"RIFF\x00\x00\x00\x00AVI ")
        tmp_path = tmp.name
    f = make_file(
        db,
        deployment_id=d.id,
        file_path=tmp_path,
        file_type="video",
        file_format="avi",
        best_frame_number=None,
        best_frame_path=None,
    )
    try:
        for url in (
            f"/api/files/{f.id}/image",
            f"/api/files/{f.id}/image?size=thumb",
        ):
            resp = client.get(url)
            assert resp.status_code == 404, url
            assert "could not be decoded" in resp.json()["detail"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_label_tree_count_by_file(client, db):
    p = make_project(db, counting_threshold=0.5)
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


def _bulk_verify(client, ids, verified=True):
    return client.post(
        "/api/files/bulk-verify", json={"file_ids": ids, "verified": verified}
    )


def test_bulk_verify_signs_off_every_file_in_one_request(client, db):
    """The Files tab's bulk action. Same rule per file as the PATCH: the
    visible boxes are verified, the weak ones rejected, the file flagged."""
    from app.models import Detection

    d, _ = _setup_deployment(db)
    a = make_file(db, deployment_id=d.id)
    b = make_file(db, deployment_id=d.id)
    strong_id = make_detection(db, file_id=a.id, confidence=0.9).id
    weak_id = make_detection(db, file_id=a.id, confidence=0.05).id
    db.commit()

    resp = _bulk_verify(client, [a.id, b.id])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"updated_count": 2}

    db.expire_all()
    assert a.verified is True
    assert b.verified is True
    assert db.get(Detection, strong_id).verified is True
    weak = db.get(Detection, weak_id)
    assert weak.label == "false detection"
    assert weak.verified is True


def test_bulk_unverify_takes_the_sign_off_back(client, db):

    d, _ = _setup_deployment(db)
    a = make_file(db, deployment_id=d.id)
    det = make_detection(db, file_id=a.id, confidence=0.9)
    db.commit()
    assert _bulk_verify(client, [a.id]).status_code == 200

    resp = _bulk_verify(client, [a.id], verified=False)
    assert resp.status_code == 200, resp.text

    db.expire_all()
    assert a.verified is False
    assert det.verified is False


def test_bulk_verify_of_nothing_is_a_no_op_and_of_unknown_ids_a_404(client, db):
    assert _bulk_verify(client, []).json() == {"updated_count": 0}
    assert _bulk_verify(client, [str(uuid.uuid4())]).status_code == 404
