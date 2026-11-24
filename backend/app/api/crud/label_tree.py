"""
The label filter, in two parts that change at different times.

- **The tree** (`build_label_filter_tree`): which labels the project has,
  nested by taxonomy. It changes only when labels change, so it travels
  with the rest of what the project has in `GET /api/events/filter-options`.
- **The counts** (`count_labels_under_filters`): how many of each label
  the page would show under its active filters. They change with every
  filter, so they are their own request, `GET /api/events/label-counts`.

Keeping them apart is the point: a filter change costs one pass over the
project's boxes for the counts, not a second one to rebuild the tree.
The front end puts the counts on the leaves (`withCounts` in
`lib/taxonomy-utils.ts`). A count never travels with the tree.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.taxonomic_rank import NO_TAXONOMY, species_binomial, to_display_case
from app.ml.taxonomic_rollup import format_leaf_annotation
from app.ml.taxonomy_db import BUILTIN_MODEL_ID
from app.models import Project
from app.models.label_taxonomy import LabelTaxonomy

logger = get_logger(__name__)

LEVEL_ORDER = ["class", "order", "family", "genus"]
COUNT_UNITS = ("event", "file", "detection")


def count_labels_under_filters(
    db: Session,
    project_id: str,
    count_by: str,
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
) -> dict[str, int]:
    """How many of each label the page would show if that label were
    ticked alone, under every other filter of the page, the label filter
    itself left out, as a faceted search does. Keyed by
    ``label_taxonomy_id``; a label missing from the result counts 0.

    Each page's count comes from that page's own query, so a number can
    never promise what the page cannot show. That was the mismatch a
    tester hit with "20 detections" beside an empty grid, all twenty
    verified under the Unverified filter.

    Args:
        count_by: "event" counts distinct events (the Counts page), "file"
            distinct files (the Files tab), "detection" boxes (the Labels
            grid).
        site_ids, date_from, date_to, and the keyword filters: the page's
            active filters, in the list endpoints' vocabulary.
    """
    from app.api.crud.event import count_events_by_label
    from app.api.crud.file import count_labels_files_by_label
    from app.api.schemas.label import LabelFilters
    from app.services.label_service import count_detections_by_label

    if count_by not in COUNT_UNITS:
        raise ValueError(f"count_by must be one of {COUNT_UNITS}, not {count_by!r}")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return {}
    threshold = project.counting_threshold

    def _dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    if count_by == "detection":
        return count_detections_by_label(
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
    if count_by == "file":
        return count_labels_files_by_label(
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
    return count_events_by_label(
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


def build_label_filter_tree(db: Session, present_ids: set[str]) -> dict | None:
    """The label filter tree for the labels a project has.

    ``present_ids`` are the project's labels (`present_label_rows`, the
    same set the species colours and the flat list use), so the filter
    can never offer a label without a colour or the other way round.
    Every one is a leaf, nested under its ranks. Leaves carry no count:
    a label the filters leave nothing of is still there, at 0 once the
    counts arrive, so a fully verified species can still be picked under
    "Unverified".

    Returns ``{"tree", "all_leaf_ids"}``, or None when the project has no
    labels (the front end then shows the flat list).
    """
    if not present_ids:
        return None

    taxonomy_rows_raw = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.id.in_(present_ids))
        .all()
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
    # only shape that cannot drift from the counts, which are keyed by id.
    #
    # Detector classes ("Animal", "Fish", "Person": the builtin rows a
    # box without a species points at) are leaves at the root, so
    # "detected, species not set" is one click away. Keyed on the builtin
    # model id, not on the missing ranks: a model's or a custom label
    # without ranks is still "No taxonomy".
    has_taxonomy = []
    unranked_rows = []
    class_rows = []
    for row in taxonomy_rows_raw:
        if row.classification_model_id == BUILTIN_MODEL_ID:
            class_rows.append(row)
        elif any([
            row.taxon_class, row.taxon_order,
            row.taxon_family, row.taxon_genus,
        ]):
            has_taxonomy.append(row)
        else:
            unranked_rows.append(row)
    taxonomy_rows = has_taxonomy

    if not taxonomy_rows and not unranked_rows and not class_rows:
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
            "children": {},
            "is_leaf": True,
        }

        if leaf_id not in current:
            current[leaf_id] = leaf_node

    # Detector classes at the root, the other rank-less rows in the "No
    # taxonomy" group. One leaf per row, named the same way the ranked
    # leaves are, so two rows sharing a name stay tellable apart ("Animal"
    # for the detector's own class, "Animalia" for a kingdom-level rollup).
    #
    # The node id stays "other": only leaf ids reach `all_leaf_ids` and the
    # filter, so this one is never user-visible, and changing it would
    # break saved filter URLs for no gain.
    def _unranked_leaf(row: LabelTaxonomy) -> dict:
        return {
            "id": row.id,
            "name": row.scientific_name or row.name.replace("_", " ").capitalize(),
            "children": {},
            "is_leaf": True,
        }

    for row in class_rows:
        root[row.id] = _unranked_leaf(row)

    if unranked_rows:
        other_children = {row.id: _unranked_leaf(row) for row in unranked_rows}
        root[other_key] = {
            "id": "other",
            "name": NO_TAXONOMY,
            "children": other_children,
            "is_leaf": False,
        }

    # Convert to TaxonomyNode format, sorted: leaves first, then parents,
    # each alphabetically.
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
                tree_node["child_count"] = _count_leaves(children)
                parents.append(tree_node)
            else:
                if "annotation" in node:
                    tree_node["annotation"] = node["annotation"]
                leaves.append(tree_node)

        leaves.sort(key=lambda n: n["name"].lower())
        parents.sort(key=lambda n: n["name"].lower())
        return leaves + parents

    def _count_leaves(children: list[dict]) -> int:
        return sum(
            _count_leaves(child["children"]) if child["children"] else 1
            for child in children
        )

    tree = _to_tree(root, 1)

    all_leaf_ids: list[str] = []

    def _collect_leaves(nodes: list[dict]) -> None:
        for node in nodes:
            if node["children"]:
                _collect_leaves(node["children"])
            else:
                all_leaf_ids.append(node["id"])

    _collect_leaves(tree)

    return {"tree": tree, "all_leaf_ids": all_leaf_ids}
