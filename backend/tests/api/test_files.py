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
