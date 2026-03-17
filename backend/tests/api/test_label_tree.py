"""Tests for the /api/events/label-tree endpoint and build_label_filter_tree."""

from datetime import datetime

from app.api.crud.label_tree import build_label_filter_tree
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_project,
    make_site,
)

MODEL_ID = "EUR-DF-v1-3"


def _add_taxonomy(db, name, level, **kw):
    """Helper to insert a LabelTaxonomy row."""
    row = LabelTaxonomy(
        classification_model_id=MODEL_ID,
        name=name,
        level=level,
        **kw,
    )
    db.add(row)
    db.flush()
    return row


def _setup_project_with_detections(db, label_list):
    """Create project -> site -> deployment -> events with detections for each label."""
    p = make_project(db, classification_model_id=MODEL_ID)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)

    for sp in label_list:
        ev = make_event_with_files(
            db,
            deployment_id=d.id,
            start_time=datetime(2024, 6, 1, 12, 0),
        )
        # Get a file from the event to attach detection
        from app.models.event import event_files as ef_table
        file_row = db.execute(
            ef_table.select().where(ef_table.c.event_id == ev.id)
        ).first()
        make_detection(db, file_id=file_row.file_id, label=sp, label_confidence=0.8)

    db.flush()
    return p


def test_build_tree_with_labels_and_rollups(db):
    """Verifies tree structure with both normal labels and rolled-up taxa."""
    p = _setup_project_with_detections(db, ["leopard", "lion", "felidae"])

    # Add taxonomy rows
    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus")
    _add_taxonomy(db, "lion", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="leo")
    _add_taxonomy(db, "felidae", "family",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae")

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    assert "tree" in result
    assert "all_leaf_ids" in result

    # Should have leaf IDs for leopard, lion, and felidae:unspecified
    leaf_ids = result["all_leaf_ids"]
    assert "leopard" in leaf_ids
    assert "lion" in leaf_ids
    assert "felidae:unspecified" in leaf_ids

    # Check tree has hierarchy
    tree = result["tree"]
    assert len(tree) > 0

    # Find the felidae:unspecified leaf in the tree
    def find_node(nodes, target_id):
        for n in nodes:
            if n["id"] == target_id:
                return n
            found = find_node(n.get("children", []), target_id)
            if found:
                return found
        return None

    felidae_node = find_node(tree, "felidae:unspecified")
    assert felidae_node is not None
    # Rolled-up leaf should have clean name and annotation
    assert felidae_node["name"] == "Felidae"
    assert felidae_node.get("annotation") == "unspecified"

    # felidae:unspecified should be nested inside "family Felidae", not at root
    root_ids = [n["id"] for n in tree]
    assert "felidae:unspecified" not in root_ids

    # Navigate hierarchy: class Mammalia -> order Carnivora -> family Felidae
    mammalia_node = find_node(tree, "class:mammalia")
    assert mammalia_node is not None

    family_felidae = find_node([mammalia_node], "class:mammalia|order:carnivora|family:felidae")
    assert family_felidae is not None
    # felidae:unspecified should be a child of family Felidae
    felidae_child_ids = [c["id"] for c in family_felidae.get("children", [])]
    assert "felidae:unspecified" in felidae_child_ids


def test_build_tree_no_model_no_detections(db):
    """Returns None when project has no model and no detections."""
    p = make_project(db, classification_model_id=None)
    result = build_label_filter_tree(p.id, db)
    assert result is None


def test_build_tree_detection_only(db):
    """Detection-only project gets a tree from categories."""
    from app.ml.taxonomy_db import ensure_builtin_labels

    ensure_builtin_labels(db)

    p = make_project(db, classification_model_id=None)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)

    for cat in ["animal", "person", "vehicle"]:
        ev = make_event_with_files(
            db,
            deployment_id=d.id,
            start_time=datetime(2024, 6, 1, 12, 0),
        )
        from app.models.event import event_files as ef_table

        file_row = db.execute(
            ef_table.select().where(ef_table.c.event_id == ev.id)
        ).first()
        make_detection(
            db,
            file_id=file_row.file_id,
            label=None,
            category=cat,
        )

    db.flush()

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    assert "tree" in result

    leaf_ids = result["all_leaf_ids"]
    assert "animal" in leaf_ids
    assert "person" in leaf_ids
    assert "vehicle" in leaf_ids

    # All three should be under the "other" node
    tree = result["tree"]
    other_node = next(
        (n for n in tree if n["id"] == "other"), None
    )
    assert other_node is not None
    other_child_ids = [c["id"] for c in other_node["children"]]
    assert "animal" in other_child_ids
    assert "person" in other_child_ids
    assert "vehicle" in other_child_ids


def test_build_tree_no_detections(db):
    """Returns None when project has no detections."""
    p = make_project(db, classification_model_id=MODEL_ID)
    result = build_label_filter_tree(p.id, db)
    assert result is None


def test_build_tree_no_taxonomy_rows(db):
    """Labels without taxonomy rows appear under 'other'."""
    p = _setup_project_with_detections(db, ["leopard"])
    # Don't add any taxonomy rows
    result = build_label_filter_tree(p.id, db)
    assert result is not None
    assert "leopard" in result["all_leaf_ids"]
    other_node = next(
        (n for n in result["tree"] if n["id"] == "other"), None
    )
    assert other_node is not None


def test_build_tree_with_event_counts(db):
    """Event counts appear in the label_event_counts dict."""
    p = _setup_project_with_detections(db, ["leopard", "leopard", "lion"])
    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus")
    _add_taxonomy(db, "lion", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="leo")

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    counts = result["label_event_counts"]
    assert counts["leopard"] >= 1
    assert counts["lion"] >= 1


def test_unmatched_label_in_other(db):
    """Labels not in taxonomy go to 'other' group."""
    p = _setup_project_with_detections(db, ["leopard", "blank"])
    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus")
    # "blank" has no taxonomy row

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    assert "blank" in result["all_leaf_ids"]

    # Find the "other" group
    other_group = None
    for node in result["tree"]:
        if node["id"] == "other":
            other_group = node
            break

    assert other_group is not None
    # "blank" should be a leaf under "other"
    child_ids = [c["id"] for c in other_group.get("children", [])]
    assert "blank" in child_ids


def test_unspecified_suffix_stripping(client, db):
    """Filter parsing strips :unspecified suffix from label IDs."""
    p = _setup_project_with_detections(db, ["felidae"])
    _add_taxonomy(db, "felidae", "family",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae")

    # Query with :unspecified suffix — should still match
    resp = client.get(
        f"/api/events?project_id={p.id}&labels=felidae:unspecified"
    )
    assert resp.status_code == 200


def test_label_tree_endpoint(client, db):
    """The /label-tree endpoint returns tree or null."""
    p = make_project(db, classification_model_id=MODEL_ID)
    resp = client.get(f"/api/events/label-tree?project_id={p.id}")
    assert resp.status_code == 200
    # No detections -> null
    assert resp.json() is None


def test_label_tree_endpoint_with_data(client, db):
    """The /label-tree endpoint returns tree when data exists."""
    p = _setup_project_with_detections(db, ["leopard"])
    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus")

    resp = client.get(f"/api/events/label-tree?project_id={p.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data is not None
    assert "tree" in data
    assert "all_leaf_ids" in data
    assert "leopard" in data["all_leaf_ids"]
    assert data["count_unit"] == "event"


def test_leaf_has_structured_fields(db):
    """Label leaf nodes have annotation and count fields."""
    p = _setup_project_with_detections(db, ["leopard"])
    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus")

    result = build_label_filter_tree(p.id, db)
    assert result is not None

    def find_node(nodes, target_id):
        for n in nodes:
            if n["id"] == target_id:
                return n
            found = find_node(n.get("children", []), target_id)
            if found:
                return found
        return None

    leaf = find_node(result["tree"], "leopard")
    assert leaf is not None
    assert leaf["name"] == "Panthera pardus"
    assert leaf["annotation"] == "leopard"
    assert leaf["count"] >= 1

    # Parent should have child_count and count
    mammalia = find_node(result["tree"], "class:mammalia")
    assert mammalia is not None
    assert mammalia.get("child_count") is not None
    assert mammalia.get("count") is not None
