"""Tests for label_taxonomy_id FK behavior in label tree and delete endpoint."""

from datetime import datetime

from app.api.crud.label_tree import build_label_filter_tree
from app.ml.taxonomy_db import link_detections_to_taxonomy
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)

MODEL_ID = "EUR-DF-v1-3"


def _add_taxonomy(db, name, level, model_id=MODEL_ID, **kw):
    row = LabelTaxonomy(
        classification_model_id=model_id,
        name=name,
        level=level,
        **kw,
    )
    db.add(row)
    db.flush()
    return row


def _setup_project_with_linked_detections(db, label_list):
    """Create project with detections linked via FK to taxonomy rows."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)

    detections = []
    for sp in label_list:
        ev = make_event_with_files(
            db, deployment_id=d.id, event_start_local=datetime(2024, 6, 1, 12, 0),
        )
        from app.models.event import event_files as ef_table
        file_row = db.execute(
            ef_table.select().where(ef_table.c.event_id == ev.id)
        ).first()
        det = make_detection(db, file_id=file_row.file_id, label=sp, label_confidence=0.8)
        detections.append(det)

    db.flush()
    return p, detections


# ---------- Label tree with FK-linked detections ----------


def test_label_tree_uses_fk_linked_detections(db):
    """Tree correctly resolves taxonomy via FK join when detections are linked."""
    p, dets = _setup_project_with_linked_detections(db, ["leopard", "lion"])

    leopard_tax = _add_taxonomy(
        db, "leopard", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera",
        taxon_species="pardus",
    )
    lion_tax = _add_taxonomy(
        db, "lion", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera",
        taxon_species="leo",
    )

    # Link detections to taxonomy
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    # Leaf IDs are now taxonomy UUIDs
    assert leopard_tax.id in result["all_leaf_ids"]
    assert lion_tax.id in result["all_leaf_ids"]


def test_label_tree_mixed_linked_and_unlinked(db):
    """Tree includes FK-linked detections; unlinked are excluded from tree."""
    p, dets = _setup_project_with_linked_detections(db, ["leopard", "mystery_animal"])

    leopard_tax = _add_taxonomy(
        db, "leopard", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera",
        taxon_species="pardus",
    )

    # Only link leopard; mystery_animal has no taxonomy row and no FK
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    # leopard resolved via FK
    assert leopard_tax.id in result["all_leaf_ids"]


def test_label_tree_all_linked(db):
    """All detections linked via FK appear in the tree."""
    p, dets = _setup_project_with_linked_detections(db, ["leopard"])

    leopard_tax = _add_taxonomy(
        db, "leopard", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera",
        taxon_species="pardus",
    )

    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    assert leopard_tax.id in result["all_leaf_ids"]


# ---------- Delete custom label sets FK to NULL ----------


def test_delete_custom_label_nullifies_fk(client, db):
    """Deleting a custom label sets label_taxonomy_id to NULL on detections."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, captured_at_local=datetime(2024, 6, 1, 12, 0))

    # Create custom taxonomy entry
    custom_tax = LabelTaxonomy(
        classification_model_id="",
        name="my_bird",
        level="unknown",
        is_custom=True,
        project_id=p.id,
    )
    db.add(custom_tax)
    db.flush()

    # Create detection linked to the custom taxonomy
    det = make_detection(db, file_id=f.id, label="my_bird",
                         label_confidence=0.8,
                         label_taxonomy_id=custom_tax.id)
    db.flush()

    # Delete via API
    resp = client.delete(f"/api/projects/{p.id}/custom-labels/{custom_tax.id}")
    assert resp.status_code == 204

    db.expire_all()
    # Detection's label string preserved, FK nullified
    assert det.label == "my_bird"
    assert det.label_taxonomy_id is None


def test_delete_custom_label_preserves_other_detections(client, db):
    """Deleting a custom label only nullifies FK on its own detections."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, captured_at_local=datetime(2024, 6, 1, 12, 0))

    # Create two custom taxonomy entries
    tax_a = LabelTaxonomy(
        classification_model_id="", name="bird_a", level="unknown",
        is_custom=True, project_id=p.id,
    )
    tax_b = LabelTaxonomy(
        classification_model_id="", name="bird_b", level="unknown",
        is_custom=True, project_id=p.id,
    )
    db.add_all([tax_a, tax_b])
    db.flush()

    det_a = make_detection(db, file_id=f.id, label="bird_a",
                           label_confidence=0.8, label_taxonomy_id=tax_a.id)
    det_b = make_detection(db, file_id=f.id, label="bird_b",
                           label_confidence=0.8, label_taxonomy_id=tax_b.id)
    db.flush()

    # Delete only bird_a
    resp = client.delete(f"/api/projects/{p.id}/custom-labels/{tax_a.id}")
    assert resp.status_code == 204

    db.expire_all()
    assert det_a.label_taxonomy_id is None
    assert det_b.label_taxonomy_id == tax_b.id


def test_rename_custom_label_relinks_fk(client, db):
    """Renaming a custom label updates both Detection.label and the FK."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, captured_at_local=datetime(2024, 6, 1, 12, 0))

    # Create custom taxonomy and a detection pointing to a *different* taxonomy row
    old_tax = _add_taxonomy(db, "cow", "species", taxon_class="mammalia")
    custom_tax = LabelTaxonomy(
        classification_model_id="", name="old_name", level="unknown",
        is_custom=True, project_id=p.id,
    )
    db.add(custom_tax)
    db.flush()

    det = make_detection(db, file_id=f.id, label="old_name",
                         label_confidence=0.8, label_taxonomy_id=old_tax.id)
    db.flush()

    # Rename via PATCH
    resp = client.patch(
        f"/api/projects/{p.id}/custom-labels/{custom_tax.id}",
        json={"name": "new_name"},
    )
    assert resp.status_code == 200

    db.expire_all()
    assert det.label == "new_name"
    assert det.label_taxonomy_id == custom_tax.id


def test_update_custom_label_relinks_stale_fk(client, db):
    """Updating taxonomy fields re-links detections that have stale FKs."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, captured_at_local=datetime(2024, 6, 1, 12, 0))

    stale_tax = _add_taxonomy(db, "cow", "species")
    custom_tax = LabelTaxonomy(
        classification_model_id="", name="my_animal", level="unknown",
        is_custom=True, project_id=p.id,
    )
    db.add(custom_tax)
    db.flush()

    # Detection has label="my_animal" but FK points to "cow" taxonomy (stale)
    det = make_detection(db, file_id=f.id, label="my_animal",
                         label_confidence=0.8, label_taxonomy_id=stale_tax.id)
    db.flush()

    # Update taxonomy fields (no name change)
    resp = client.patch(
        f"/api/projects/{p.id}/custom-labels/{custom_tax.id}",
        json={"name": "my_animal", "taxon_class": "mammalia"},
    )
    assert resp.status_code == 200

    db.expire_all()
    assert det.label_taxonomy_id == custom_tax.id
