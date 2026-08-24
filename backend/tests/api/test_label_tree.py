"""Tests for the /api/events/label-tree endpoint and build_label_filter_tree."""

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


def _add_taxonomy(db, name, level, **kw):
    """Helper to insert a LabelTaxonomy row."""
    from app.ml.taxonomic_rollup import format_scientific_name_from_taxonomy_row

    scientific_name = kw.pop("scientific_name", None)
    if scientific_name is None:
        scientific_name = format_scientific_name_from_taxonomy_row(
            name,
            kw.get("taxon_genus"),
            kw.get("taxon_species"),
            kw.get("taxon_family"),
            kw.get("taxon_order"),
            kw.get("taxon_class"),
            kw.get("taxon_variant"),
        )
    row = LabelTaxonomy(
        classification_model_id=MODEL_ID,
        name=name,
        level=level,
        scientific_name=scientific_name,
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
            event_start_local=datetime(2024, 6, 1, 12, 0),
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
    leopard_tax = _add_taxonomy(
        db, "leopard", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus",
    )
    lion_tax = _add_taxonomy(
        db, "lion", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera", taxon_species="leo",
    )
    felidae_tax = _add_taxonomy(
        db, "felidae", "family",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae",
    )

    # Link detections to taxonomy via FK
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    assert "tree" in result
    assert "all_leaf_ids" in result

    # Leaf IDs are now taxonomy UUIDs
    leaf_ids = result["all_leaf_ids"]
    assert leopard_tax.id in leaf_ids
    assert lion_tax.id in leaf_ids
    assert felidae_tax.id in leaf_ids

    # Check tree has hierarchy
    tree = result["tree"]
    assert len(tree) > 0

    def find_node(nodes, target_id):
        for n in nodes:
            if n["id"] == target_id:
                return n
            found = find_node(n.get("children", []), target_id)
            if found:
                return found
        return None

    felidae_node = find_node(tree, felidae_tax.id)
    assert felidae_node is not None
    # Rolled-up leaf should have clean name and annotation
    assert felidae_node["name"] == "Felidae"
    assert felidae_node.get("annotation") == "family"

    # Leaf should be nested (not at root)
    root_ids = [n["id"] for n in tree]
    assert felidae_tax.id not in root_ids

    # Navigate hierarchy: class Mammalia -> order Carnivora -> family Felidae
    mammalia_node = find_node(tree, "class:mammalia")
    assert mammalia_node is not None

    family_felidae = find_node(
        [mammalia_node], "class:mammalia|order:carnivora|family:felidae"
    )
    assert family_felidae is not None
    # felidae leaf should be a child of family Felidae
    felidae_child_ids = [c["id"] for c in family_felidae.get("children", [])]
    assert felidae_tax.id in felidae_child_ids


def test_build_tree_no_model_no_detections(db):
    """Returns None when project has no model and no detections."""
    p = make_project(db, classification_model_id=None)
    result = build_label_filter_tree(p.id, db)
    assert result is None


def test_build_tree_detection_only(db):
    """Detection-only project gets a tree from categories."""
    from app.ml.taxonomy_db import ensure_builtin_labels

    builtin_ids = ensure_builtin_labels(db)

    p = make_project(db, classification_model_id=None)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)

    for cat in ["animal", "person", "vehicle"]:
        ev = make_event_with_files(
            db,
            deployment_id=d.id,
            event_start_local=datetime(2024, 6, 1, 12, 0),
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

    # Link detections to builtin taxonomy
    link_detections_to_taxonomy(p.id, db)
    db.flush()

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    assert "tree" in result

    # Leaf IDs are taxonomy UUIDs for builtins
    leaf_ids = result["all_leaf_ids"]
    assert builtin_ids["animal"] in leaf_ids
    assert builtin_ids["person"] in leaf_ids
    assert builtin_ids["vehicle"] in leaf_ids


def test_build_tree_no_detections(db):
    """Returns None when project has no detections."""
    p = make_project(db, classification_model_id=MODEL_ID)
    result = build_label_filter_tree(p.id, db)
    assert result is None


def test_build_tree_no_taxonomy_rows(db):
    """Detections without taxonomy links don't appear in the tree."""
    p = _setup_project_with_detections(db, ["leopard"])
    # Don't add any taxonomy rows and don't link
    result = build_label_filter_tree(p.id, db)
    # No linked detections, so tree should be None
    assert result is None


def test_build_tree_with_event_counts(db):
    """Event counts appear in the label_event_counts dict."""
    p = _setup_project_with_detections(db, ["leopard", "leopard", "lion"])
    _add_taxonomy(db, "leopard", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus")
    _add_taxonomy(db, "lion", "species",
                  taxon_class="mammalia", taxon_order="carnivora",
                  taxon_family="felidae", taxon_genus="panthera", taxon_species="leo")

    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    counts = result["label_event_counts"]
    assert counts["leopard"] >= 1
    assert counts["lion"] >= 1


def test_unmatched_label_in_other(db):
    """Only linked detections appear. Unlinked detections are excluded."""
    p = _setup_project_with_detections(db, ["leopard", "blank"])
    leopard_tax = _add_taxonomy(
        db, "leopard", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus",
    )
    # "blank" has no taxonomy row

    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    # Only leopard should appear (linked), blank has no taxonomy
    assert leopard_tax.id in result["all_leaf_ids"]


def test_filter_by_taxonomy_id(client, db):
    """Event filter uses taxonomy UUIDs for label matching."""
    p = _setup_project_with_detections(db, ["felidae"])
    felidae_tax = _add_taxonomy(
        db, "felidae", "family",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae",
    )
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    # Query with taxonomy UUID
    resp = client.get(
        f"/api/events?project_id={p.id}&labels={felidae_tax.id}"
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
    leopard_tax = _add_taxonomy(
        db, "leopard", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus",
    )
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

    resp = client.get(f"/api/events/label-tree?project_id={p.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data is not None
    assert "tree" in data
    assert "all_leaf_ids" in data
    assert leopard_tax.id in data["all_leaf_ids"]
    assert data["count_unit"] == "event"


def test_leaf_has_structured_fields(db):
    """Label leaf nodes have annotation and count fields."""
    p = _setup_project_with_detections(db, ["leopard"])
    leopard_tax = _add_taxonomy(
        db, "leopard", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae", taxon_genus="panthera", taxon_species="pardus",
    )
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

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

    leaf = find_node(result["tree"], leopard_tax.id)
    assert leaf is not None
    assert leaf["name"] == "P. pardus"
    assert leaf["annotation"] == "leopard"
    assert leaf["count"] >= 1

    # Parent should have child_count and count
    mammalia = find_node(result["tree"], "class:mammalia")
    assert mammalia is not None
    assert mammalia.get("child_count") is not None
    assert mammalia.get("count") is not None


def test_rollup_leaf_annotation_falls_back_to_the_rank(db):
    """A leaf whose model label matches its rank-derived display has no
    second name to show, so the annotation names the rank: "Felidae
    (family)". Same rule as the species picker, see
    tests/ml/test_taxonomy_parser.py::test_leaf_naming_rule."""
    p = _setup_project_with_detections(db, ["felidae"])
    felidae_tax = _add_taxonomy(
        db, "felidae", "family",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="felidae",
    )
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

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

    leaf = find_node(result["tree"], felidae_tax.id)
    assert leaf is not None
    assert leaf["name"] == "Felidae"
    assert leaf["annotation"] == "family"


def test_rollup_leaf_annotation_disambiguates_collisions(db):
    """When a model label differs from its rank-derived display name
    (e.g. "micromammal" mapped to class Mammalia), the annotation shows
    the underlying label so sibling leaves under the same parent can be
    told apart visually."""
    p = _setup_project_with_detections(db, ["micromammal", "mammalia"])

    # Both rows are class-level under Mammalia but represent distinct
    # classifier outputs. Their scientific_names collide ("Mammalia").
    micromammal_tax = _add_taxonomy(
        db, "micromammal", "class",
        taxon_class="mammalia",
        scientific_name="Mammalia",
    )
    mammalia_tax = _add_taxonomy(
        db, "mammalia", "class",
        taxon_class="mammalia",
        scientific_name="Mammalia",
    )
    link_detections_to_taxonomy(p.id, db)
    db.expire_all()

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

    micromammal_leaf = find_node(result["tree"], micromammal_tax.id)
    assert micromammal_leaf is not None
    assert micromammal_leaf["name"] == "Mammalia"
    # name "micromammal" doesn't match display "Mammalia" → underlying
    # label surfaces in the italic annotation.
    assert micromammal_leaf["annotation"] == "micromammal"

    mammalia_leaf = find_node(result["tree"], mammalia_tax.id)
    assert mammalia_leaf is not None
    assert mammalia_leaf["name"] == "Mammalia"
    # name "mammalia" matches display "Mammalia" → the rank annotates it.
    assert mammalia_leaf["annotation"] == "class"

# ── Video counts follow what the grid can show ───────────────────────


def _video_with_frames(db, project, frames):
    """One video with best_frame_number=10 and a `deer` detection on each
    of `frames`. Returns (file, taxonomy_row)."""
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    tax = _add_taxonomy(
        db, "deer", "species", taxon_genus="cervus", taxon_species="elaphus"
    )
    f = make_file(
        db,
        deployment_id=dep.id,
        file_type="video",
        file_format="mp4",
        best_frame_number=10,
    )
    for frame in frames:
        make_detection(
            db,
            file_id=f.id,
            confidence=0.9,
            label="deer",
            label_taxonomy_id=tax.id,
            frame_number=frame,
        )
    db.flush()
    return f, tax


def test_video_counts_only_the_best_frame(db):
    """The filter promised counts the grid could not deliver. A 3-frame
    video with a deer on every frame counted 3, while the Labels grid
    (best-frame gated) showed 1. On a real 30-second clip that was
    "person 62" over a grid holding 4."""
    p = make_project(db, classification_model_id=MODEL_ID)
    _f, tax = _video_with_frames(db, p, [0, 10, 20])

    result = build_label_filter_tree(p.id, db, count_by="detection")
    assert result["label_event_counts"]["deer"] == 1


def test_offbestframe_only_label_leaves_the_tree(db):
    """A label living only on frames nobody can open offered a branch
    that led to a blank grid. It must not be listed at all: the tree is
    also the universe of labels the Save step can exclude, so a
    listed-but-unreachable label is worse than a missing one."""
    p = make_project(db, classification_model_id=MODEL_ID)
    f, _tax = _video_with_frames(db, p, [10])
    ghost = _add_taxonomy(
        db, "chimpanzee", "species", taxon_genus="pan",
        taxon_species="troglodytes",
    )
    make_detection(
        db,
        file_id=f.id,
        confidence=0.9,
        label="chimpanzee",
        label_taxonomy_id=ghost.id,
        frame_number=150,
    )
    db.flush()

    result = build_label_filter_tree(p.id, db, count_by="detection")
    # Counts are keyed by label name; leaf ids by taxonomy id.
    assert result["label_event_counts"]["deer"] == 1
    assert "chimpanzee" not in result["label_event_counts"]
    assert ghost.id not in result["all_leaf_ids"]


def test_verified_offbestframe_label_stays_visible(db):
    """A human looked at that box, so it must stay reachable. Same
    escape hatch calculate_max_n_for_event uses."""
    p = make_project(db, classification_model_id=MODEL_ID)
    f, _tax = _video_with_frames(db, p, [10])
    serval = _add_taxonomy(
        db, "serval", "species", taxon_genus="leptailurus",
        taxon_species="serval",
    )
    make_detection(
        db,
        file_id=f.id,
        confidence=0.9,
        label="serval",
        label_taxonomy_id=serval.id,
        frame_number=150,
        verified=True,
    )
    db.flush()

    result = build_label_filter_tree(p.id, db, count_by="detection")
    assert result["label_event_counts"]["serval"] == 1


def test_image_detections_are_never_gated(db):
    """Images have no frames, so every image detection is visible and
    nothing about this change may touch them."""
    p = make_project(db, classification_model_id=MODEL_ID)
    site = make_site(db, project_id=p.id)
    dep = make_deployment(db, site_id=site.id)
    tax = _add_taxonomy(
        db, "fox", "species", taxon_genus="vulpes", taxon_species="vulpes"
    )
    f = make_file(db, deployment_id=dep.id, file_type="image")
    for _ in range(3):
        make_detection(
            db, file_id=f.id, confidence=0.9, label="fox",
            label_taxonomy_id=tax.id,
        )
    db.flush()

    result = build_label_filter_tree(p.id, db, count_by="detection")
    assert result["label_event_counts"]["fox"] == 3


# ---------------------------------------------------------------------------
# The "No taxonomy" branch
# ---------------------------------------------------------------------------


def _unranked_taxonomy(db, name, scientific_name, model_id):
    """A taxonomy row with no rank columns at all, so it lands under
    "No taxonomy" rather than in the class > order > family chain."""
    row = LabelTaxonomy(
        classification_model_id=model_id,
        name=name,
        level="none",
        scientific_name=scientific_name,
        common_name=name.capitalize(),
    )
    db.add(row)
    db.flush()
    return row


def test_unranked_branch_is_named_no_taxonomy(db):
    """The bucket for rank-less labels is called "No taxonomy", the same
    name the insights pages use for it, not the older bare "Other"."""
    from app.ml.taxonomic_rank import NO_TAXONOMY

    p = make_project(db, classification_model_id=MODEL_ID)
    site = make_site(db, project_id=p.id)
    dep = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=dep.id, file_type="image")
    tax = _unranked_taxonomy(db, "unknown", "Unknown", MODEL_ID)
    make_detection(
        db, file_id=f.id, confidence=0.9, label="unknown",
        label_taxonomy_id=tax.id,
    )
    db.flush()

    result = build_label_filter_tree(p.id, db, count_by="detection")
    names = [n["name"] for n in result["tree"]]
    assert NO_TAXONOMY in names
    assert "Other" not in names


def test_two_rank_less_rows_sharing_a_name_stay_separate(db):
    """Two taxonomy rows can share a name: the builtin "animal" that
    stands for an unclassified detector box, and a model's kingdom-level
    rollup row, also called "animal". Grouping the branch by name
    collapsed them into one leaf that carried the summed count but only
    one of the two ids, so selecting it filtered on half the detections
    the count promised. One leaf per row, each with its own count."""
    from app.ml.taxonomy_db import BUILTIN_MODEL_ID

    p = make_project(db, classification_model_id=MODEL_ID)
    site = make_site(db, project_id=p.id)
    dep = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=dep.id, file_type="image")

    builtin = _unranked_taxonomy(db, "animal", "Animal", BUILTIN_MODEL_ID)
    rollup = _unranked_taxonomy(db, "animal", "Animalia", MODEL_ID)

    # 3 unclassified boxes, 1 rolled up to kingdom.
    for _ in range(3):
        make_detection(
            db, file_id=f.id, confidence=0.9, label_taxonomy_id=builtin.id,
        )
    make_detection(
        db, file_id=f.id, confidence=0.9, label="animal",
        label_taxonomy_id=rollup.id,
    )
    db.flush()

    result = build_label_filter_tree(p.id, db, count_by="detection")

    # Both rows are reachable as their own filter value.
    assert builtin.id in result["all_leaf_ids"]
    assert rollup.id in result["all_leaf_ids"]

    leaves = {
        leaf["id"]: leaf
        for node in result["tree"]
        for leaf in node["children"]
    }
    # Each leaf carries its own count, not the sum of both (which would
    # be 4 on either leaf).
    assert leaves[builtin.id]["count"] == 3
    assert leaves[rollup.id]["count"] == 1
    # And they are tellable apart, because the leaf is named for its own
    # scientific name rather than the name the two rows share.
    assert leaves[builtin.id]["name"] == "Animal"
    assert leaves[rollup.id]["name"] == "Animalia"


# --- variant rank ---


def _find_child(nodes, name):
    for node in nodes:
        if node["name"] == name:
            return node
    raise AssertionError(
        f"no node named {name!r} in {[n['name'] for n in nodes]}"
    )


def _variant_taxonomy(db):
    canidae = dict(
        taxon_class="mammalia", taxon_order="carnivora", taxon_family="canidae"
    )
    adult = _add_taxonomy(
        db, "red fox adult", "variant",
        taxon_genus="vulpes", taxon_species="vulpes", taxon_variant="adult",
        **canidae,
    )
    juvenile = _add_taxonomy(
        db, "red fox juvenile", "variant",
        taxon_genus="vulpes", taxon_species="vulpes", taxon_variant="juvenile",
        **canidae,
    )
    wolf = _add_taxonomy(
        db, "wolf", "species",
        taxon_genus="canis", taxon_species="lupus",
        **canidae,
    )
    return adult, juvenile, wolf


def test_variants_nest_under_a_species_node(db):
    """Variant leaves sit under a species node named for the binomial;
    a plain species class keeps its leaf directly under the genus."""
    adult, juvenile, wolf = _variant_taxonomy(db)
    p = _setup_project_with_detections(
        db, ["red fox adult", "red fox juvenile", "wolf"]
    )
    link_detections_to_taxonomy(p.id, db)

    result = build_label_filter_tree(p.id, db)
    assert result is not None

    mammalia = _find_child(result["tree"], "Mammalia")
    carnivora = _find_child(mammalia["children"], "Carnivora")
    canidae = _find_child(carnivora["children"], "Canidae")

    # Plain species: leaf directly under its genus, unchanged shape.
    canis = _find_child(canidae["children"], "Canis")
    wolf_leaf = _find_child(canis["children"], "C. lupus")
    assert wolf_leaf["children"] == []

    # Variants: species node "V. vulpes" holding the two variant leaves.
    vulpes = _find_child(canidae["children"], "Vulpes")
    species_node = _find_child(vulpes["children"], "V. vulpes")
    assert species_node["children"]
    assert species_node["child_count"] == 2
    assert species_node["count"] == 2

    adult_leaf = _find_child(species_node["children"], "Adult")
    assert adult_leaf["id"] == adult.id
    assert adult_leaf["annotation"] == "red fox adult"
    assert adult_leaf["count"] == 1
    juvenile_leaf = _find_child(species_node["children"], "Juvenile")
    assert juvenile_leaf["id"] == juvenile.id

    # Only real leaves are selectable; the species node is not one.
    assert set(result["all_leaf_ids"]) == {adult.id, juvenile.id, wolf.id}


def test_variant_tree_counts_by_detection(db):
    """Detection count mode carries through the deeper tree."""
    adult, juvenile, _wolf = _variant_taxonomy(db)
    p = _setup_project_with_detections(
        db, ["red fox adult", "red fox adult", "red fox juvenile"]
    )
    link_detections_to_taxonomy(p.id, db)

    result = build_label_filter_tree(p.id, db, count_by="detection")
    assert result is not None
    mammalia = _find_child(result["tree"], "Mammalia")
    carnivora = _find_child(mammalia["children"], "Carnivora")
    canidae = _find_child(carnivora["children"], "Canidae")
    vulpes = _find_child(canidae["children"], "Vulpes")
    species_node = _find_child(vulpes["children"], "V. vulpes")
    assert species_node["count"] == 3
    assert _find_child(species_node["children"], "Adult")["count"] == 2


def test_species_level_row_nests_inside_its_variant_species_node(db):
    """A species-level row on a chain that has variants (a rollup row
    like "vulpes vulpes") nests inside the species parent node instead
    of sitting next to it as a same-named sibling."""
    adult, juvenile, _wolf = _variant_taxonomy(db)
    rollup = _add_taxonomy(
        db, "vulpes vulpes", "species",
        taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="canidae", taxon_genus="vulpes", taxon_species="vulpes",
    )
    p = _setup_project_with_detections(
        db, ["red fox adult", "red fox juvenile", "vulpes vulpes", "wolf"]
    )
    link_detections_to_taxonomy(p.id, db)

    result = build_label_filter_tree(p.id, db)
    assert result is not None
    mammalia = _find_child(result["tree"], "Mammalia")
    carnivora = _find_child(mammalia["children"], "Carnivora")
    canidae = _find_child(carnivora["children"], "Canidae")
    vulpes = _find_child(canidae["children"], "Vulpes")

    # Exactly one "V. vulpes" row under the genus: the species node.
    named = [n for n in vulpes["children"] if n["name"] == "V. vulpes"]
    assert len(named) == 1
    species_node = named[0]

    # It holds the two variant leaves plus the species-level leaf.
    child_ids = {c["id"] for c in species_node["children"]}
    assert child_ids == {adult.id, juvenile.id, rollup.id}
    rollup_leaf = next(
        c for c in species_node["children"] if c["id"] == rollup.id
    )
    assert rollup_leaf["annotation"] == "vulpes vulpes"

    # A plain species on a chain without variants keeps its flat leaf.
    canis = _find_child(canidae["children"], "Canis")
    assert _find_child(canis["children"], "C. lupus")["children"] == []
