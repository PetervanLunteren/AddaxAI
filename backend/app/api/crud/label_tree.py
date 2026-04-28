"""
CRUD for building the label filter tree from the label_taxonomy table.

Returns a pre-built taxonomy tree containing only labels with actual detections,
annotated with event counts. Replaces the frontend's two-query + client-side
pruning approach.
"""

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, Event, File, Project
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
                  "file" counts distinct files per label (videos resolved
                  via Detection.file_id's source_video_id on frame rows);
                  "detection" counts individual detections per label.

    Returns:
        Dict with tree, all_leaf_ids, label_event_counts, count_unit; or None if no taxonomy.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return None

    threshold = project.detection_threshold

    # Count by label_taxonomy_id (authoritative FK).
    # Only count detections at or above the project's confidence threshold
    # so the tree matches what the verify page actually displays.
    if count_by == "detection":
        label_count_rows = (
            db.query(
                Detection.label_taxonomy_id,
                func.count(Detection.id),
            )
            .join(File, File.id == Detection.file_id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .filter(Deployment.project_id == project_id)
            .filter(Detection.label_taxonomy_id.isnot(None))
            .filter(or_(Detection.confidence >= threshold, Detection.verified == True))
            .group_by(Detection.label_taxonomy_id)
            .all()
        )
    elif count_by == "file":
        # Count distinct media items (image/video), resolving frame rows up
        # to their parent video so a video isn't undercounted once per frame.
        media_id = func.coalesce(File.source_video_id, File.id)
        label_count_rows = (
            db.query(
                Detection.label_taxonomy_id,
                func.count(func.distinct(media_id)),
            )
            .join(File, File.id == Detection.file_id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .filter(Deployment.project_id == project_id)
            .filter(Detection.label_taxonomy_id.isnot(None))
            .filter(or_(Detection.confidence >= threshold, Detection.verified == True))
            .group_by(Detection.label_taxonomy_id)
            .all()
        )
    else:
        label_count_rows = (
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
            .filter(or_(Detection.confidence >= threshold, Detection.verified == True))
            .group_by(Detection.label_taxonomy_id)
            .all()
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

    detected_labels = set(label_event_counts.keys())
    taxonomy_rows = taxonomy_rows_raw

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
            display_label = row.display_name or row.name
            display_name = row.name.replace("_", " ")
            leaf_id = row.id
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
            leaf_id = row.id
            display = (
                row.display_name
                or row.name.replace("_", " ").capitalize()
            )
            # Annotation: show the underlying model label when it differs
            # from the rank-derived display, otherwise the literal
            # "unspecified". This matches the species annotation rule
            # (model label in italics) and lets users tell apart sibling
            # rollup leaves that share a display name (e.g. "micromammal"
            # vs "mammalia" both rendering as "Mammalia").
            if display and row.name.lower() != display.lower():
                annotation = row.name.replace("_", " ")
            else:
                annotation = "unspecified"
            leaf_node = {
                "id": leaf_id,
                "name": display,
                "annotation": annotation,
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
        # Resolve unmatched names to taxonomy IDs for leaf IDs.
        # Include all taxonomy rows (with and without taxonomy fields).
        name_to_tid: dict[str, str] = {
            r.name: r.id for r in taxonomy_rows_raw
        }
        for label_name in sorted(unmatched_labels):
            count = label_event_counts.get(label_name, 0)
            leaf_id = name_to_tid.get(label_name, label_name)
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
