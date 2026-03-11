"""Tests for species_taxonomy_id FK behavior in species tree and delete endpoint."""

from datetime import datetime

from app.api.crud.species_tree import build_species_filter_tree
from app.ml.taxonomy_db import link_detections_to_taxonomy
from app.models.species_taxonomy import SpeciesTaxonomy
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
    row = SpeciesTaxonomy(
        classification_model_id=model_id,
        name=name,
        level=level,
        **kw,
    )
    db.add(row)
    db.flush()
    return row


def _setup_project_with_linked_detections(db, species_list):
    """Create project with detections linked via FK to taxonomy rows."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)

    detections = []
    for sp in species_list:
        ev = make_event_with_files(
            db, deployment_id=d.id, start_time=datetime(2024, 6, 1, 12, 0),
        )
        from app.models.event import event_files as ef_table
        file_row = db.execute(
            ef_table.select().where(ef_table.c.event_id == ev.id)
        ).first()
        det = make_detection(db, file_id=file_row.file_id, species=sp, species_confidence=0.8)
        detections.append(det)

    db.flush()
    return p, detections


# ---------- Species tree with FK-linked detections ----------


def test_species_tree_uses_fk_linked_detections(db):
    """Tree correctly resolves taxonomy via FK join when detections are linked."""
    p, dets = _setup_project_with_linked_detections(db, ["leopard", "lion"])

    leo_tax = _add_taxonomy(db, "leopard", "species",
                            taxon_class="mammalia", taxon_order="carnivora",
                            taxon_family="felidae", taxon_genus="panthera",
                            taxon_species="pardus")
    _add_taxonomy(db, "lion", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera",
                  taxon_species="leo")

    # Link detections to taxonomy
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    result = build_species_filter_tree(p.id, db)
    assert result is not None
    assert "leopard" in result["all_leaf_ids"]
    assert "lion" in result["all_leaf_ids"]


def test_species_tree_mixed_linked_and_unlinked(db):
    """Tree works with both FK-linked and unlinked detections."""
    p, dets = _setup_project_with_linked_detections(db, ["leopard", "mystery_animal"])

    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera",
                  taxon_species="pardus")

    # Only link leopard; mystery_animal has no taxonomy row
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    result = build_species_filter_tree(p.id, db)
    assert result is not None
    # leopard resolved via FK, mystery_animal falls to "other"
    assert "leopard" in result["all_leaf_ids"]
    assert "mystery_animal" in result["all_leaf_ids"]


def test_species_tree_fallback_for_unlinked(db):
    """Unlinked detections still match via string fallback."""
    p, dets = _setup_project_with_linked_detections(db, ["leopard"])
    # Don't link — leave species_taxonomy_id NULL
    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera",
                  taxon_species="pardus")

    result = build_species_filter_tree(p.id, db)
    assert result is not None
    assert "leopard" in result["all_leaf_ids"]


# ---------- Delete custom species sets FK to NULL ----------


def test_delete_custom_species_nullifies_fk(client, db):
    """Deleting a custom species sets species_taxonomy_id to NULL on detections."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, timestamp=datetime(2024, 6, 1, 12, 0))

    # Create custom taxonomy entry
    custom_tax = SpeciesTaxonomy(
        classification_model_id="",
        name="my_bird",
        level="unknown",
        is_custom=True,
        project_id=p.id,
    )
    db.add(custom_tax)
    db.flush()

    # Create detection linked to the custom taxonomy
    det = make_detection(db, file_id=f.id, species="my_bird",
                         species_confidence=0.8,
                         species_taxonomy_id=custom_tax.id)
    db.flush()

    # Delete via API
    resp = client.delete(f"/api/projects/{p.id}/custom-species/{custom_tax.id}")
    assert resp.status_code == 204

    db.expire_all()
    # Detection's species string preserved, FK nullified
    assert det.species == "my_bird"
    assert det.species_taxonomy_id is None


def test_delete_custom_species_preserves_other_detections(client, db):
    """Deleting a custom species only nullifies FK on its own detections."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, timestamp=datetime(2024, 6, 1, 12, 0))

    # Create two custom taxonomy entries
    tax_a = SpeciesTaxonomy(
        classification_model_id="", name="bird_a", level="unknown",
        is_custom=True, project_id=p.id,
    )
    tax_b = SpeciesTaxonomy(
        classification_model_id="", name="bird_b", level="unknown",
        is_custom=True, project_id=p.id,
    )
    db.add_all([tax_a, tax_b])
    db.flush()

    det_a = make_detection(db, file_id=f.id, species="bird_a",
                           species_confidence=0.8, species_taxonomy_id=tax_a.id)
    det_b = make_detection(db, file_id=f.id, species="bird_b",
                           species_confidence=0.8, species_taxonomy_id=tax_b.id)
    db.flush()

    # Delete only bird_a
    resp = client.delete(f"/api/projects/{p.id}/custom-species/{tax_a.id}")
    assert resp.status_code == 204

    db.expire_all()
    assert det_a.species_taxonomy_id is None
    assert det_b.species_taxonomy_id == tax_b.id


def test_rename_custom_species_relinks_fk(client, db):
    """Renaming a custom species updates both Detection.species and the FK."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, timestamp=datetime(2024, 6, 1, 12, 0))

    # Create custom taxonomy and a detection pointing to a *different* taxonomy row
    old_tax = _add_taxonomy(db, "cow", "species", taxon_class="mammalia")
    custom_tax = SpeciesTaxonomy(
        classification_model_id="", name="old_name", level="unknown",
        is_custom=True, project_id=p.id,
    )
    db.add(custom_tax)
    db.flush()

    det = make_detection(db, file_id=f.id, species="old_name",
                         species_confidence=0.8, species_taxonomy_id=old_tax.id)
    db.flush()

    # Rename via PATCH
    resp = client.patch(
        f"/api/projects/{p.id}/custom-species/{custom_tax.id}",
        json={"name": "new_name"},
    )
    assert resp.status_code == 200

    db.expire_all()
    assert det.species == "new_name"
    assert det.species_taxonomy_id == custom_tax.id


def test_update_custom_species_relinks_stale_fk(client, db):
    """Updating taxonomy fields re-links detections that have stale FKs."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, timestamp=datetime(2024, 6, 1, 12, 0))

    stale_tax = _add_taxonomy(db, "cow", "species")
    custom_tax = SpeciesTaxonomy(
        classification_model_id="", name="my_animal", level="unknown",
        is_custom=True, project_id=p.id,
    )
    db.add(custom_tax)
    db.flush()

    # Detection has species="my_animal" but FK points to "cow" taxonomy (stale)
    det = make_detection(db, file_id=f.id, species="my_animal",
                         species_confidence=0.8, species_taxonomy_id=stale_tax.id)
    db.flush()

    # Update taxonomy fields (no name change)
    resp = client.patch(
        f"/api/projects/{p.id}/custom-species/{custom_tax.id}",
        json={"name": "my_animal", "taxon_class": "mammalia"},
    )
    assert resp.status_code == 200

    db.expire_all()
    assert det.species_taxonomy_id == custom_tax.id
