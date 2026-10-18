"""
CRUD for building the label filter tree from the label_taxonomy table.

Returns a pre-built taxonomy tree containing only labels with actual detections,
annotated with event counts. Replaces the frontend's two-query + client-side
pruning approach.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.taxonomic_rank import NO_TAXONOMY, species_binomial, to_display_case
from app.ml.taxonomic_rollup import format_leaf_annotation
from app.models import Project
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
    *,
    verification: str | None = None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
) -> dict | None:
    """
    Build the label filter tree for a project from label_taxonomy + detections.

    The leaves are every label present in the project (`present_label_rows`,
    the same query the species colours use). The count beside a leaf
    follows every other filter of the page the tree is on: it is what that
    page would show if the label were ticked alone, the label filter itself
    left out, as a faceted search does. A label the filters leave nothing
    of stays in the tree at 0, so a fully verified species is still there
    to pick under "Unverified". Each page's count comes from that page's
    own query (`count_detections_by_label`, `count_labels_files_by_label`,
    `count_events_by_label`), so the tree cannot promise what the page
    cannot show; this was the mismatch a tester hit with "20 detections"
    beside an empty grid, all twenty verified under the Unverified filter.

    Args:
        count_by: "event" (default) counts distinct events per label (the
                  Counts page); "file" distinct files (the Files tab);
                  "detection" boxes (the Labels grid).
        site_ids, date_from, date_to, and the keyword filters: the page's
                  active filters, in the list endpoints' vocabulary.

    Returns:
        Dict with tree, all_leaf_ids, label_event_counts, count_unit; or None if no taxonomy.
    """
    from app.api.crud.event import count_events_by_label, present_label_rows
    from app.api.crud.file import count_labels_files_by_label
    from app.api.schemas.label import LabelFilters
    from app.services.label_service import count_detections_by_label

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return None

    threshold = project.counting_threshold
    present_ids = {
        row[0] for row in present_label_rows(db, project_id, threshold)
    }
    if not present_ids:
        return None

    def _dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    if count_by == "detection":
        counts = count_detections_by_label(
            db,
            project_id,
            LabelFilters(
                site_ids=site_ids,
                date_from=_dt(date_from),
                date_to=_dt(date_to),
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                min_label_confidence=min_label_confidence,
                max_label_confidence=max_label_confidence,
                verified=(
                    None if verification in (None, "all")
                    else verification == "verified"
                ),
                flagged=flagged if flagged in ("flagged", "not_flagged") else None,
                favorited=(
                    favorited if favorited in ("favorited", "not_favorited") else None
                ),
            ),
        )
    elif count_by == "file":
        counts = count_labels_files_by_label(
            db,
            project_id,
            # The Files tab never lowers its floor below the project
            # threshold (routers/labels.py), unlike the Detections tab.
            floor=threshold,
            empty=empty or "all",
            site_ids=site_ids,
            date_from=_dt(date_from),
            date_to=_dt(date_to),
            verification=verification,
            flagged=flagged,
            favorited=favorited,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            min_label_confidence=min_label_confidence,
            max_label_confidence=max_label_confidence,
        )
    else:
        counts = count_events_by_label(
            db,
            project_id,
            project_floor=threshold,
            site_ids=site_ids,
            date_from=_dt(date_from),
            date_to=_dt(date_to),
            verification=verification,
            flagged=flagged,
            favorited=favorited,
            empty=empty,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            min_label_confidence=min_label_confidence,
            max_label_confidence=max_label_confidence,
        )

    taxonomy_id_counts = {tid: counts.get(tid, 0) for tid in present_ids}
    detected_taxonomy_ids = present_ids

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
