"""Tests for label_taxonomy_id FK linking and builtin labels."""

from datetime import datetime

from app.ml.taxonomy_db import (
    BUILTIN_MODEL_ID,
    ensure_builtin_labels,
    link_detections_to_taxonomy,
)
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
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


def _make_project_with_detections(db, label_list, model_id=MODEL_ID):
    """Create project -> site -> deployment -> file -> detections."""
    p = make_project(db, classification_model_id=model_id)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, timestamp=datetime(2024, 6, 1, 12, 0))

    detections = []
    for lbl in label_list:
        det = make_detection(db, file_id=f.id, label=lbl, label_confidence=0.8)
        detections.append(det)

    db.flush()
    return p, detections


# ---------- ensure_builtin_labels ----------


def test_ensure_builtin_labels_creates_rows(db):
    """Seeds person and vehicle rows."""
    count = ensure_builtin_labels(db)
    assert count == 2

    rows = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID)
        .all()
    )
    names = {r.name for r in rows}
    assert names == {"person", "vehicle"}

    for r in rows:
        assert r.level == "none"
        assert r.is_custom is False
        assert r.taxon_class is None


def test_ensure_builtin_labels_idempotent(db):
    """Calling twice doesn't duplicate rows."""
    count1 = ensure_builtin_labels(db)
    count2 = ensure_builtin_labels(db)
    assert count1 == 2
    assert count2 == 0

    total = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID)
        .count()
    )
    assert total == 2


# ---------- link_detections_to_taxonomy ----------


def test_link_detections_basic(db):
    """Links detections to model-level taxonomy rows."""
    p, dets = _make_project_with_detections(db, ["leopard", "lion"])

    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_family="felidae")
    leo_tax = _add_taxonomy(db, "lion", "species",
                            taxon_class="mammalia", taxon_family="felidae")

    count = link_detections_to_taxonomy(p.id, db)
    assert count == 2

    db.expire_all()
    for det in dets:
        assert det.label_taxonomy_id is not None

    # Verify correct mapping
    lion_det = [d for d in dets if d.label == "lion"][0]
    assert lion_det.label_taxonomy_id == leo_tax.id


def test_link_detections_idempotent(db):
    """Calling twice doesn't re-link already linked detections."""
    p, dets = _make_project_with_detections(db, ["leopard"])
    _add_taxonomy(db, "leopard", "species")

    count1 = link_detections_to_taxonomy(p.id, db)
    count2 = link_detections_to_taxonomy(p.id, db)
    assert count1 == 1
    assert count2 == 0


def test_link_detections_no_taxonomy(db):
    """Returns 0 when no taxonomy rows exist for the label."""
    p, _ = _make_project_with_detections(db, ["unknown_animal"])
    count = link_detections_to_taxonomy(p.id, db)
    assert count == 0


def test_link_detections_builtin_labels(db):
    """Links person/vehicle detections to builtin taxonomy rows."""
    ensure_builtin_labels(db)
    p, dets = _make_project_with_detections(db, ["person", "vehicle"])

    count = link_detections_to_taxonomy(p.id, db)
    assert count == 2

    db.expire_all()
    for det in dets:
        assert det.label_taxonomy_id is not None


def test_link_detections_custom_label(db):
    """Links detections to custom label when no model-level match exists."""
    p, dets = _make_project_with_detections(db, ["my_custom_bird"])

    custom_tax = LabelTaxonomy(
        classification_model_id="",
        name="my_custom_bird",
        level="unknown",
        is_custom=True,
        project_id=p.id,
    )
    db.add(custom_tax)
    db.flush()

    count = link_detections_to_taxonomy(p.id, db)
    assert count == 1

    db.expire_all()
    assert dets[0].label_taxonomy_id == custom_tax.id


def test_link_detections_model_priority_over_custom(db):
    """Model-level taxonomy takes priority over custom for the same name."""
    p, dets = _make_project_with_detections(db, ["leopard"])

    model_tax = _add_taxonomy(db, "leopard", "species")
    custom_tax = LabelTaxonomy(
        classification_model_id="",
        name="leopard",
        level="species",
        is_custom=True,
        project_id=p.id,
    )
    db.add(custom_tax)
    db.flush()

    link_detections_to_taxonomy(p.id, db)

    db.expire_all()
    assert dets[0].label_taxonomy_id == model_tax.id


def test_link_detections_cross_project_isolation(db):
    """Linking in one project doesn't affect detections in another."""
    p1, dets1 = _make_project_with_detections(db, ["leopard"])
    p2, dets2 = _make_project_with_detections(db, ["leopard"])

    _add_taxonomy(db, "leopard", "species")

    link_detections_to_taxonomy(p1.id, db)

    db.expire_all()
    assert dets1[0].label_taxonomy_id is not None
    assert dets2[0].label_taxonomy_id is None


def test_link_detections_null_label_skipped(db):
    """Detections with label=NULL are not linked."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, timestamp=datetime(2024, 6, 1, 12, 0))

    det_with = make_detection(db, file_id=f.id, label="leopard", label_confidence=0.8)
    det_without = make_detection(db, file_id=f.id, label=None)

    _add_taxonomy(db, "leopard", "species")

    count = link_detections_to_taxonomy(p.id, db)
    assert count == 1

    db.expire_all()
    assert det_with.label_taxonomy_id is not None
    assert det_without.label_taxonomy_id is None


def test_link_detections_no_model(db):
    """Works for projects without a classification model (custom labels only)."""
    p = make_project(db, classification_model_id=None)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id, timestamp=datetime(2024, 6, 1, 12, 0))

    det = make_detection(db, file_id=f.id, label="person")
    db.flush()

    ensure_builtin_labels(db)
    count = link_detections_to_taxonomy(p.id, db)
    assert count == 1

    db.expire_all()
    assert det.label_taxonomy_id is not None
