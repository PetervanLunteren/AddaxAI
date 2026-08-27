"""Species colours: one palette, assigned so related species differ most.

The Labels grid, the Counts page and the annotated JPEG export all
colour a detection by its species. The colours exist so a person
scanning a block of look-alike crops notices the odd one out, and that
only works when the species most likely to be confused for each other
sit far apart in colour. Two shades of green on two rodents is exactly
the failure this module prevents.

The rule:

1. Collect the species present in the project, the same population the
   label filter offers (threshold-or-verified, visible frame).
2. Sort them by taxonomy: class, order, family, genus, species, variant,
   then name. Siblings end up next to each other.
3. Walk ``SPECIES_PALETTE``: rank ``i`` gets ``SPECIES_PALETTE[i % 12]``.

The palette is ordered farthest-first, so any two consecutive entries
are far apart perceptually. Sorting siblings next to each other and then
walking that order is what gives them the most contrasting colours.
Species 13 onwards share a colour with the species twelve ranks away,
which is almost never a relative.

Why not hash the label, as before: with ten species present and any
fixed number of colours, two of them land on the same or a neighbouring
colour almost every time (the birthday problem). Only looking at which
species are present avoids that. The cost is that colours are per
project and can shift when a new species appears.

This is the only implementation. The frontend fetches the map from
``GET /api/projects/{id}/label-colors`` and the export reads it through
``_visualisation_style.detection_color``, so the JPEG on disk always
matches the grid on screen.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.api.crud.event import present_label_rows
from app.ml.taxonomy_db import BUILTIN_MODEL_ID
from app.models import Project
from app.models.label_taxonomy import LabelTaxonomy

# Twelve colours, ordered farthest-first by CIEDE2000 starting from the
# brand dark red, generated in OKLCH (lightness 0.45 to 0.74) and then
# fixed as literals. The three category colours (#0f6064 animal,
# #ff8945 person, #71b7ba vehicle) were excluded from the candidate pool
# so a species never looks like an unlabelled box. Consecutive entries
# are at least 30 apart; the closest pair overall (16) is the last entry
# against the first, which only meet at the wrap from rank 11 to 12.
# Unclassified detections carry a ``__builtin__`` taxonomy row named after
# their category (see ``taxonomy_db.ensure_builtin_taxonomy``). Those are
# not species: they keep the category colours of ``getCategoryColor`` in
# ``frontend/src/lib/detection-utils.ts`` and take no palette slot, so a
# person box is orange on screen and in the export alike.
CATEGORY_COLORS: dict[str, str] = {
    "animal": "#0f6064",
    "person": "#ff8945",
    "vehicle": "#71b7ba",
}
# Unknown categories, mirroring the "bad" red fallback of category_color.
DEFAULT_CATEGORY_COLOR = "#882000"

SPECIES_PALETTE: tuple[str, ...] = (
    "#882000",
    "#73c076",
    "#d48dd8",
    "#17559b",
    "#326402",
    "#82326c",
    "#79abfc",
    "#cba63a",
    "#db6371",
    "#8059bb",
    "#849b11",
    "#8f2e3d",
)


def _taxonomic_sort_key(row: LabelTaxonomy) -> tuple[str, ...]:
    """Class > order > family > genus > species > variant > name.

    Missing ranks sort as empty strings, so a family-level rollup row
    sits directly in front of the species of that family, next to the
    labels it is most likely confused with.
    """
    return tuple(
        (value or "").lower()
        for value in (
            row.taxon_class,
            row.taxon_order,
            row.taxon_family,
            row.taxon_genus,
            row.taxon_species,
            row.taxon_variant,
            row.name,
        )
    )


def _fnv1a(text: str) -> int:
    hash_value = 2166136261
    for ch in text:
        hash_value ^= ord(ch)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value


def fallback_color(key: str) -> str:
    """Colour for a label the project map does not know.

    Reached by the export for a label that passes the media threshold
    but not the project's counting threshold. Deterministic so the
    same label always draws the same, but with no guarantee against
    matching a present species; the map is the real answer.
    """
    return SPECIES_PALETTE[_fnv1a(key.strip().lower()) % len(SPECIES_PALETTE)]


def assign_label_colors(db: Session, project_id: str) -> dict[str, str]:
    """Colour per species present in the project.

    Keyed by both the ``label_taxonomy`` id and the lowercased label
    name, because the frontend colours by whichever it has at hand.
    Empty when the project has no labelled detections yet.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    present_ids = [
        row[0]
        for row in present_label_rows(db, project_id, project.counting_threshold)
    ]
    if not present_ids:
        return {}

    rows = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.id.in_(present_ids))
        .all()
    )

    colors: dict[str, str] = {}
    species: list[LabelTaxonomy] = []
    for row in rows:
        if row.classification_model_id == BUILTIN_MODEL_ID:
            color = CATEGORY_COLORS.get(row.name.lower(), DEFAULT_CATEGORY_COLOR)
            colors[row.id] = color
            colors[row.name.lower()] = color
        else:
            species.append(row)

    species.sort(key=_taxonomic_sort_key)
    for rank, row in enumerate(species):
        color = SPECIES_PALETTE[rank % len(SPECIES_PALETTE)]
        colors[row.id] = color
        colors[row.name.lower()] = color
    return colors
