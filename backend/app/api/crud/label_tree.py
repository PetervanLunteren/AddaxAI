"""
CRUD for building the label filter tree from the label_taxonomy table.

Returns a pre-built taxonomy tree containing only labels with actual detections,
annotated with event counts. Replaces the frontend's two-query + client-side
pruning approach.
"""

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, Event, File, Project, Site
from app.models.event import event_files
from app.models.label_taxonomy import LabelTaxonomy

logger = get_logger(__name__)

LEVEL_ORDER = ["class", "order", "family", "genus"]


def build_label_filter_tree(
    project_id: str, db: Session, count_by: str = "event",
) -> dict | None:
    """
    Build the label filter tree for a project from label_taxonomy + detections.

    Args:
        count_by: "event" (default) counts distinct events per label;
                  "detection" counts individual detections per label.

    Returns:
        Dict with tree, all_leaf_ids, label_event_counts, count_unit; or None if no taxonomy.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return None

    model_id = project.classification_model_id
    threshold = project.detection_threshold

    # Use COALESCE so detection-only projects (label=NULL) fall back
    # to category ("animal", "person", "vehicle").
    effective_label = func.coalesce(Detection.label, Detection.category)

    # Get detected labels + counts (events or detections).
    # Only count detections at or above the project's confidence threshold
    # so the tree matches what the verify page actually displays.
    if count_by == "detection":
        label_count_rows = (
            db.query(effective_label, func.count(Detection.id))
            .join(File, File.id == Detection.file_id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .join(Site, Site.id == Deployment.site_id)
            .filter(Site.project_id == project_id)
            .filter(effective_label.isnot(None))
            .filter(or_(Detection.confidence >= threshold, Detection.verified == True))
            .group_by(effective_label)
            .all()
        )
    else:
        label_count_rows = (
            db.query(
                effective_label, func.count(func.distinct(Event.id))
            )
            .join(File, File.id == Detection.file_id)
            .join(event_files, event_files.c.file_id == File.id)
            .join(Event, Event.id == event_files.c.event_id)
            .join(Deployment, Deployment.id == Event.deployment_id)
            .join(Site, Site.id == Deployment.site_id)
            .filter(Site.project_id == project_id)
            .filter(effective_label.isnot(None))
            .filter(or_(Detection.confidence >= threshold, Detection.verified == True))
            .group_by(effective_label)
            .all()
        )

    if not label_count_rows:
        return None

    label_event_counts = {name: count for name, count in label_count_rows}
    detected_labels = set(label_event_counts.keys())

    # Get taxonomy rows via FK join (preferred) + string match fallback for unlinked.
    # FK-linked: query distinct taxonomy rows referenced by detections in this project.
    linked_taxonomy_ids = (
        db.query(func.distinct(Detection.label_taxonomy_id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .join(Site, Site.id == Deployment.site_id)
        .filter(
            Site.project_id == project_id,
            Detection.label_taxonomy_id.isnot(None),
        )
        .subquery()
    )
    fk_rows = [
        r for r in (
            db.query(LabelTaxonomy)
            .filter(LabelTaxonomy.id.in_(
                db.query(linked_taxonomy_ids.c[0])
            ))
            .all()
        )
        if r.name in detected_labels
    ]
    fk_label_names = {r.name for r in fk_rows}

    # String-match fallback for unlinked detections (label_taxonomy_id IS NULL)
    unlinked_labels = detected_labels - fk_label_names
    fallback_rows: list[LabelTaxonomy] = []
    if unlinked_labels:
        if model_id:
            model_rows = (
                db.query(LabelTaxonomy)
                .filter(
                    LabelTaxonomy.classification_model_id == model_id,
                    LabelTaxonomy.project_id.is_(None),
                    LabelTaxonomy.name.in_(unlinked_labels),
                )
                .all()
            )
            model_label_names = {r.name.lower() for r in model_rows}
        else:
            model_rows = []
            model_label_names = set()

        custom_rows = (
            db.query(LabelTaxonomy)
            .filter(
                LabelTaxonomy.project_id == project_id,
                LabelTaxonomy.is_custom == True,  # noqa: E712
                LabelTaxonomy.name.in_(unlinked_labels),
            )
            .all()
        )
        custom_rows = [
            r for r in custom_rows
            if r.name.lower() not in model_label_names
        ]
        fallback_rows = model_rows + custom_rows

    taxonomy_rows = fk_rows + fallback_rows

    # Rows with no taxonomy fields go to "Other" instead of root
    has_taxonomy = []
    no_taxonomy_names: set[str] = set()
    for row in taxonomy_rows:
        if any([
            row.taxon_class, row.taxon_order,
            row.taxon_family, row.taxon_genus,
        ]):
            has_taxonomy.append(row)
        else:
            no_taxonomy_names.add(row.name)
    taxonomy_rows = has_taxonomy

    # Build sets for matched vs unmatched labels
    matched_labels = {row.name for row in taxonomy_rows}
    unmatched_labels = (detected_labels - matched_labels) | no_taxonomy_names

    if not taxonomy_rows and not unmatched_labels:
        return None

    # Build hierarchical tree
    root: dict = {}
    other_key = "__other__"

    def _ensure_parent(
        current: dict, level_name: str, taxon_value: str, path_parts: list[str],
    ) -> dict:
        """Ensure a parent node exists and return its children dict."""
        path_parts.append(f"{level_name}:{taxon_value.lower()}")
        node_id = "|".join(path_parts)
        if node_id not in current:
            display = taxon_value if level_name == "species" else taxon_value.title()
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

        # Walk through taxonomy levels to build parent chain
        levels = [
            ("class", row.taxon_class),
            ("order", row.taxon_order),
            ("family", row.taxon_family),
            ("genus", row.taxon_genus),
        ]

        for level_name, taxon_value in levels:
            if not taxon_value:
                continue
            current = _ensure_parent(current, level_name, taxon_value, path_parts)
            if row.level == level_name:
                break

        # Add leaf node
        count = label_event_counts.get(row.name, 0)

        if row.level == "species":
            # Build binomial display name.
            # Model-native: taxon_species is the epithet -> prepend genus.
            # Custom (GBIF): taxon_species is already the full binomial.
            if row.is_custom:
                display_label = row.taxon_species or row.name
            elif row.taxon_species and row.taxon_genus:
                genus = row.taxon_genus.strip()
                display_label = f"{genus[0].upper()}. {row.taxon_species}"
            else:
                display_label = row.taxon_species or row.name
            display_name = row.name.replace("_", " ")
            leaf_id = row.name
            leaf_node = {
                "id": leaf_id,
                "name": display_label,
                "annotation": display_name,
                "count": count,
                "children": {},
                "is_leaf": True,
                "_event_count": count,
            }
        else:
            leaf_id = f"{row.name}:unspecified"
            # Check if the raw model name matches its taxon field for this level.
            # If yes (e.g. "Bovidae" == taxon_family "Bovidae"), it's a proper
            # taxon name and gets the level prefix.  If no (e.g. "Bird" !=
            # taxon_class "Aves"), it's a raw model label -- show without prefix.
            taxon_value_for_level = {
                "class": row.taxon_class,
                "order": row.taxon_order,
                "family": row.taxon_family,
                "genus": row.taxon_genus,
            }.get(row.level)
            is_formal_taxon = (
                taxon_value_for_level is not None
                and taxon_value_for_level.lower() == row.name.lower()
            )
            display = (
                row.name.replace("_", " ").title()
                if is_formal_taxon
                else row.name.replace("_", " ").capitalize()
            )
            leaf_node = {
                "id": leaf_id,
                "name": display,
                "annotation": "unspecified",
                "count": count,
                "children": {},
                "is_leaf": True,
                "_event_count": count,
            }

        if leaf_id not in current:
            current[leaf_id] = leaf_node

    # Add unmatched labels to "other" group
    if unmatched_labels:
        other_children: dict = {}
        for label_name in sorted(unmatched_labels):
            count = label_event_counts.get(label_name, 0)
            leaf_id = label_name
            other_children[leaf_id] = {
                "id": leaf_id,
                "name": label_name.replace("_", " ").capitalize(),
                "count": count,
                "children": {},
                "is_leaf": True,
                "_event_count": count,
            }
        root[other_key] = {
            "id": "other",
            "name": "Other",
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
