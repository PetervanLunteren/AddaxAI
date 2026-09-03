"""
CRUD for building the label filter tree from the label_taxonomy table.

Returns a pre-built taxonomy tree containing only labels with actual detections,
annotated with event counts. Replaces the frontend's two-query + client-side
pruning approach.
"""

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.detection_visibility import on_visible_frame
from app.ml.label_exclusion import threshold_or_verified
from app.ml.taxonomic_rank import NO_TAXONOMY, species_binomial, to_display_case
from app.ml.taxonomic_rollup import format_leaf_annotation
from app.models import Deployment, Detection, Event, File, Project
from app.models.event import event_files
from app.models.label_taxonomy import LabelTaxonomy

logger = get_logger(__name__)

LEVEL_ORDER = ["class", "order", "family", "genus"]


def build_label_filter_tree(
    project_id: str,
    db: Session,
    count_by: str = "event",
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict | None:
    """
    Build the label filter tree for a project from label_taxonomy + detections.

    Args:
        count_by: "event" (default) counts distinct events per label;
                  "file" counts distinct files per label (videos resolved
                  via Detection.file_id's source_video_id on frame rows);
                  "detection" counts individual detections per label.
        site_ids: optional list of site IDs to scope counts to.
        date_from / date_to: optional ISO date strings to scope counts by
                  File.captured_at_local.

    Returns:
        Dict with tree, all_leaf_ids, label_event_counts, count_unit; or None if no taxonomy.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return None

    threshold = project.counting_threshold

    # Scope the counts to the active site + date filters so the tree matches
    # the slice the user has narrowed to (see SIMON_FEEDBACK B11). Both clauses
    # ride on joins every count query already has (Deployment, File). Date
    # compares against File.captured_at_local with raw ISO strings, matching
    # the dashboard stats in crud/statistics.py.
    def _apply_scope(query):
        if site_ids:
            query = query.filter(Deployment.site_id.in_(site_ids))
        if date_from:
            query = query.filter(File.captured_at_local >= date_from)
        if date_to:
            query = query.filter(File.captured_at_local <= date_to)
        return query

    # Count by label_taxonomy_id (authoritative FK).
    # Only count detections at or above the project's confidence threshold
    # so the tree matches what the verify page actually displays.
    if count_by == "detection":
        query = (
            db.query(
                Detection.label_taxonomy_id,
                func.count(Detection.id),
            )
            .join(File, File.id == Detection.file_id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .filter(Deployment.project_id == project_id)
            .filter(Detection.label_taxonomy_id.isnot(None))
            .filter(threshold_or_verified(threshold))
            # Only detections the user can actually reach. Without
            # this the tree promised counts the grid cannot show.
            .filter(on_visible_frame())
        )
        label_count_rows = (
            _apply_scope(query).group_by(Detection.label_taxonomy_id).all()
        )
    elif count_by == "file":
        # Count distinct media items (image/video), resolving frame rows up
        # to their parent video so a video isn't undercounted once per frame.
        media_id = func.coalesce(File.source_video_id, File.id)
        query = (
            db.query(
                Detection.label_taxonomy_id,
                func.count(func.distinct(media_id)),
            )
            .join(File, File.id == Detection.file_id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .filter(Deployment.project_id == project_id)
            .filter(Detection.label_taxonomy_id.isnot(None))
            .filter(threshold_or_verified(threshold))
            # Only detections the user can actually reach. Without
            # this the tree promised counts the grid cannot show.
            .filter(on_visible_frame())
        )
        label_count_rows = (
            _apply_scope(query).group_by(Detection.label_taxonomy_id).all()
        )
    else:
        query = (
            db.query(
                Detection.label_taxonomy_id,
                func.count(func.distinct(Event.id)),
            )
            .join(File, File.id == Detection.file_id)
            .join(event_files, event_files.c.file_id == File.id)
            .join(Event, Event.id == event_files.c.event_id)
            .join(Deployment, Deployment.id == Event.deployment_id)
            .filter(Deployment.project_id == project_id)
            .filter(Detection.label_taxonomy_id.isnot(None))
            .filter(threshold_or_verified(threshold))
            # Only detections the user can actually reach. Without
            # this the tree promised counts the grid cannot show.
            .filter(on_visible_frame())
        )
        label_count_rows = (
            _apply_scope(query).group_by(Detection.label_taxonomy_id).all()
        )

    if not label_count_rows:
        return None

    taxonomy_id_counts = {
        tid: count for tid, count in label_count_rows if tid
    }
    detected_taxonomy_ids = set(taxonomy_id_counts.keys())

    # Load taxonomy rows for all detected taxonomy IDs
    taxonomy_rows_raw = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.id.in_(detected_taxonomy_ids))
        .all()
    )

    # Build name-based counts from taxonomy_id counts
    # (needed for the tree builder which indexes by name)
    label_event_counts: dict[str, int] = {}
    tid_to_name: dict[str, str] = {}
    for row in taxonomy_rows_raw:
        tid_to_name[row.id] = row.name
    for tid, count in taxonomy_id_counts.items():
        name = tid_to_name.get(tid)
        if name:
            label_event_counts[name] = (
                label_event_counts.get(name, 0) + count
            )

    # Rows with no taxonomy fields go under "No taxonomy" instead of root.
    #
    # Both lists hold taxonomy *rows*, never names. Two rows can share a
    # name (the builtin "animal" that stands for an unclassified detector
    # box, and a model's kingdom-level rollup row also called "animal"),
    # and grouping by name collapsed them into one leaf whose count was
    # the sum of both but whose id was whichever row the dict happened to
    # write last. Selecting that leaf then filtered on one of the two ids
    # while the count promised both. One leaf per row, keyed by id, is the
    # only shape that cannot drift from the counts it was built from.
    has_taxonomy = []
    unranked_rows = []
    for row in taxonomy_rows_raw:
        if any([
            row.taxon_class, row.taxon_order,
            row.taxon_family, row.taxon_genus,
        ]):
            has_taxonomy.append(row)
        else:
            unranked_rows.append(row)
    taxonomy_rows = has_taxonomy

    if not taxonomy_rows and not unranked_rows:
        return None

    # Build hierarchical tree
    root: dict = {}
    other_key = "__other__"

    # Chains that get a species parent node because a variant sits below
    # the species. Species-level rows on such a chain (a rollup row like
    # "vulpes vulpes", or a custom label) nest inside that node instead
    # of sitting next to it as a same-named sibling.
    variant_chains = {
        (row.taxon_genus, row.taxon_species)
        for row in taxonomy_rows
        if row.level == "variant" and row.taxon_species
    }

    def _ensure_parent(
        current: dict, level_name: str, taxon_value: str, path_parts: list[str],
        display: str,
    ) -> dict:
        """Ensure a parent node exists and return its children dict."""
        path_parts.append(f"{level_name}:{taxon_value.lower()}")
        node_id = "|".join(path_parts)
        if node_id not in current:
            current[node_id] = {
                "id": node_id,
                "name": display,
                "children": {},
                "is_leaf": False,
            }
        return current[node_id]["children"]

    for row in taxonomy_rows:
        path_parts: list[str] = []
        current = root

        # Walk through taxonomy levels to build parent chain: the ranks
        # above the row's leaf tier. Species becomes a parent only on
        # chains where a variant exists (variant leaves below it, and
        # species-level rows of that chain inside it), so trees without
        # variants are unchanged.
        levels = [
            ("class", row.taxon_class),
            ("order", row.taxon_order),
            ("family", row.taxon_family),
            ("genus", row.taxon_genus),
        ]
        if row.level == "variant" or (
            row.level == "species"
            and (row.taxon_genus, row.taxon_species) in variant_chains
        ):
            levels.append(("species", row.taxon_species))

        for level_name, taxon_value in levels:
            if not taxon_value:
                continue
            if level_name == "species":
                # The species parent shows the binomial, matching how
                # species leaves read elsewhere in the tree.
                parent_display = (
                    species_binomial(row.taxon_genus, taxon_value)
                    or taxon_value.title()
                )
            else:
                parent_display = taxon_value.title()
            current = _ensure_parent(
                current, level_name, taxon_value, path_parts, parent_display
            )
            if row.level == level_name:
                break

        # Add leaf node. Counted by taxonomy id, which is what the leaf
        # is and what the count query grouped on; going via the name
        # sums rows that share one (see the unranked note above).
        count = taxonomy_id_counts.get(row.id, 0)

        # One rule for every rank, shared with the model taxonomy tree in
        # ml.taxonomy_parser: the leaf is named for the taxon and annotated
        # with the model's own label, or with the rank when the label is
        # itself the taxon name. That last case covers rollup rows
        # ("Numididae (family)") and model classes named after their taxon
        # ("Gorilla (genus)"), which the old literal "unspecified" conflated.
        # A variant leaf sits under its species node and shows only the
        # variant ("Adult"), with the model's own label as annotation.
        leaf_id = row.id
        if row.level == "variant" and row.taxon_variant:
            display = to_display_case(row.taxon_variant)
        else:
            display = row.scientific_name or row.name.replace("_", " ").capitalize()
        leaf_node = {
            "id": leaf_id,
            "name": display,
            "annotation": format_leaf_annotation(row.name, display, row.level),
            "count": count,
            "children": {},
            "is_leaf": True,
            "_event_count": count,
        }

        if leaf_id not in current:
            current[leaf_id] = leaf_node

    # Add the rank-less rows to the "No taxonomy" group. One leaf per row,
    # named the same way the ranked leaves are, so two rows sharing a name
    # stay tellable apart ("Animal" for the detector's own category,
    # "Animalia" for a kingdom-level rollup).
    #
    # The node id stays "other": only leaf ids reach `all_leaf_ids` and the
    # filter, so this one is never user-visible, and changing it would
    # break saved filter URLs for no gain.
    if unranked_rows:
        other_children: dict = {}
        for row in unranked_rows:
            count = taxonomy_id_counts.get(row.id, 0)
            display = (
                row.scientific_name or row.name.replace("_", " ").capitalize()
            )
            other_children[row.id] = {
                "id": row.id,
                "name": display,
                "count": count,
                "children": {},
                "is_leaf": True,
                "_event_count": count,
            }
        root[other_key] = {
            "id": "other",
            "name": NO_TAXONOMY,
            "children": other_children,
            "is_leaf": False,
        }

    # Convert to TaxonomyNode format with sorting and count annotation
    def _to_tree(nodes: dict, level: int) -> list[dict]:
        leaves = []
        parents = []

        for node in nodes.values():
            children = _to_tree(node["children"], level + 1) if node["children"] else []
            tree_node = {
                "id": node["id"],
                "name": node["name"],
                "level": level,
                "children": children,
                "selected": True,
            }
            if children:
                # Annotate parent with descendant counts
                leaf_count, event_total = _count_descendants(children)
                tree_node["child_count"] = leaf_count
                tree_node["count"] = event_total
                parents.append(tree_node)
            else:
                tree_node["_event_count"] = node.get("_event_count", 0)
                if "annotation" in node:
                    tree_node["annotation"] = node["annotation"]
                if "count" in node:
                    tree_node["count"] = node["count"]
                leaves.append(tree_node)

        # Sort: leaves first (alphabetically), then parents (alphabetically)
        leaves.sort(key=lambda n: n["name"].lower())
        parents.sort(key=lambda n: n["name"].lower())
        return leaves + parents

    def _count_descendants(children: list[dict]) -> tuple[int, int]:
        """Return (leaf_count, total_events) for a list of child nodes."""
        leaf_count = 0
        event_total = 0
        for child in children:
            if child["children"]:
                lc, et = _count_descendants(child["children"])
                leaf_count += lc
                event_total += et
            else:
                leaf_count += 1
                event_total += child.get("_event_count", 0)
        return leaf_count, event_total

    tree = _to_tree(root, 1)

    # Collect all leaf IDs
    all_leaf_ids: list[str] = []

    def _collect_leaves(nodes: list[dict]) -> None:
        for node in nodes:
            if node["children"]:
                _collect_leaves(node["children"])
            else:
                all_leaf_ids.append(node["id"])

    _collect_leaves(tree)

    return {
        "tree": tree,
        "all_leaf_ids": all_leaf_ids,
        "label_event_counts": label_event_counts,
        "count_unit": count_by,
    }
