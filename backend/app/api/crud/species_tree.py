"""
CRUD for building the species filter tree from the species_taxonomy table.

Returns a pre-built taxonomy tree containing only species with actual detections,
annotated with event counts. Replaces the frontend's two-query + client-side
pruning approach.
"""

import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, Event, File, Project, Site
from app.models.event import event_files
from app.models.species_taxonomy import SpeciesTaxonomy

logger = get_logger(__name__)

LEVEL_ORDER = ["class", "order", "family", "genus"]


def build_species_filter_tree(
    project_id: str, db: Session, count_by: str = "event",
) -> dict | None:
    """
    Build the species filter tree for a project from species_taxonomy + detections.

    Args:
        count_by: "event" (default) counts distinct events per species;
                  "detection" counts individual detections per species.

    Returns:
        Dict with tree, all_leaf_ids, species_event_counts, count_unit; or None if no taxonomy.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project or not project.classification_model_id:
        return None

    model_id = project.classification_model_id

    # Get detected species + counts (events or detections)
    if count_by == "detection":
        species_count_rows = (
            db.query(Detection.species, func.count(Detection.id))
            .join(File, File.id == Detection.file_id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .join(Site, Site.id == Deployment.site_id)
            .filter(Site.project_id == project_id)
            .filter(Detection.species.isnot(None))
            .group_by(Detection.species)
            .all()
        )
    else:
        species_count_rows = (
            db.query(Detection.species, func.count(func.distinct(Event.id)))
            .join(File, File.id == Detection.file_id)
            .join(event_files, event_files.c.file_id == File.id)
            .join(Event, Event.id == event_files.c.event_id)
            .join(Deployment, Deployment.id == Event.deployment_id)
            .join(Site, Site.id == Deployment.site_id)
            .filter(Site.project_id == project_id)
            .filter(Detection.species.isnot(None))
            .group_by(Detection.species)
            .all()
        )

    if not species_count_rows:
        return None

    species_event_counts = {name: count for name, count in species_count_rows}
    detected_species = set(species_event_counts.keys())

    # Get taxonomy rows via FK join (preferred) + string match fallback for unlinked.
    # FK-linked: query distinct taxonomy rows referenced by detections in this project.
    linked_taxonomy_ids = (
        db.query(func.distinct(Detection.species_taxonomy_id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .join(Site, Site.id == Deployment.site_id)
        .filter(
            Site.project_id == project_id,
            Detection.species_taxonomy_id.isnot(None),
        )
        .subquery()
    )
    fk_rows = (
        db.query(SpeciesTaxonomy)
        .filter(SpeciesTaxonomy.id.in_(db.query(linked_taxonomy_ids.c[0])))
        .all()
    )
    fk_species_names = {r.name for r in fk_rows}

    # String-match fallback for unlinked detections (species_taxonomy_id IS NULL)
    unlinked_species = detected_species - fk_species_names
    fallback_rows: list[SpeciesTaxonomy] = []
    if unlinked_species:
        model_rows = (
            db.query(SpeciesTaxonomy)
            .filter(
                SpeciesTaxonomy.classification_model_id == model_id,
                SpeciesTaxonomy.project_id.is_(None),
                SpeciesTaxonomy.name.in_(unlinked_species),
            )
            .all()
        )
        model_species_names = {r.name.lower() for r in model_rows}

        custom_rows = (
            db.query(SpeciesTaxonomy)
            .filter(
                SpeciesTaxonomy.project_id == project_id,
                SpeciesTaxonomy.is_custom == True,  # noqa: E712
                SpeciesTaxonomy.name.in_(unlinked_species),
            )
            .all()
        )
        custom_rows = [r for r in custom_rows if r.name.lower() not in model_species_names]
        fallback_rows = model_rows + custom_rows

    taxonomy_rows = fk_rows + fallback_rows

    if not taxonomy_rows:
        return None

    # Build sets for matched vs unmatched species
    matched_species = {row.name for row in taxonomy_rows}
    unmatched_species = detected_species - matched_species

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
        count = species_event_counts.get(row.name, 0)

        if row.level == "species":
            # Build binomial display name.
            # Model-native: taxon_species is the epithet → prepend genus.
            # Custom (GBIF): taxon_species is already the full binomial.
            if row.is_custom:
                species_label = row.taxon_species or row.name
            elif row.taxon_species and row.taxon_genus:
                species_label = f"{row.taxon_genus.strip().capitalize()} {row.taxon_species}"
            else:
                species_label = row.taxon_species or row.name
            display_name = row.name.replace("_", " ")
            leaf_id = row.name
            leaf_node = {
                "id": leaf_id,
                "name": species_label,
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
            # taxon_class "Aves"), it's a raw model label — show without prefix.
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
            display = row.name.replace("_", " ").title() if is_formal_taxon else row.name.replace("_", " ")
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

    # Add unmatched species to "other" group
    if unmatched_species:
        other_children: dict = {}
        for sp in sorted(unmatched_species):
            count = species_event_counts.get(sp, 0)
            leaf_id = sp
            other_children[leaf_id] = {
                "id": leaf_id,
                "name": sp.replace("_", " "),
                "count": count,
                "children": {},
                "is_leaf": True,
                "_event_count": count,
            }
        root[other_key] = {
            "id": "other",
            "name": "other",
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
        "species_event_counts": species_event_counts,
        "count_unit": count_by,
    }
