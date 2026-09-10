"""Species colours: one palette, assigned so related species differ most.

The Labels grid, the Counts page and the annotated JPEG export all
colour a detection by its species. The colours exist so a person
scanning a block of look-alike crops notices the odd one out, and that
only works when the species most likely to be confused for each other
sit far apart in colour. Two shades of green on two rodents is exactly
the failure this module prevents.

The rule:

1. Collect the classes present in the project, the same population the
   label filter offers (threshold-or-verified, visible frame). A class
   is whatever names the box: its species when a classifier named one,
   otherwise the detector's category. Where that category comes from
   makes no difference, so "animal", "person", "vehicle" and
   "elasmobranch" are classes like any other.
2. Sort them by taxonomy: class, order, family, genus, species, variant,
   then name. Siblings end up next to each other. A detector category
   carries no taxonomy, so those sort first as a group, by name.
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

from app.api.crud.event import present_category_rows, present_label_rows
from app.ml.label_exclusion import is_non_label
from app.models import Project
from app.models.label_taxonomy import LabelTaxonomy

# Twelve colours, ordered farthest-first by CIEDE2000 starting from the
# brand dark red, generated in OKLCH (lightness 0.45 to 0.74) and then
# fixed as literals. Consecutive entries are at least 30 apart; the
# closest pair overall (16) is the last entry against the first, which
# only meet at the wrap from rank 11 to 12.
#
# The three MegaDetector categories used to hold fixed colours here and
# take no palette slot. They were folded in: a class is a class, whether
# a classifier or a detector named it, and the special case had grown a
# second copy in the frontend and a third in the export, which disagreed
# (an "elasmobranch" box drew teal on screen and brand red in the JPEG).

# A box a person rejected (X, or a relabel to a non-label class such as
# "false detection" or a model's "non-animal") still passes the scope
# rule when it sat above the threshold, so it is "present". It is not a
# class of its own: giving it a palette slot shifted every real class by
# one colour the moment someone pressed X. It keeps the same neutral grey
# the frontend uses for a label the map does not know
# (UNKNOWN_SPECIES_COLOR in frontend/src/utils/species-colors.ts).
REJECTED_LABEL_COLOR = "#6b7280"

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


def _category_sort_key(name: str) -> tuple[str, ...]:
    """The sort key of a class that has no taxonomy row.

    Same shape as ``_taxonomic_sort_key``, with every rank empty, which
    is what a ``__builtin__`` row already produces. So a detector's
    categories sort together in front of the classified species, by
    name, and the two kinds need no separate handling.
    """
    return ("",) * 6 + (name,)


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
    """Colour per class present in the project.

    Keyed by the ``label_taxonomy`` id, the lowercased label name and the
    lowercased category name, because the frontend colours by whichever
    it has at hand. Empty when the project has nothing to draw yet.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    threshold = project.counting_threshold
    colors: dict[str, str] = {}
    # (sort key, every key the frontend or the export might look this
    # class up by). One list, so classified species and bare detector
    # categories are ranked together by the same rule.
    ranked: list[tuple[tuple[str, ...], list[str]]] = []
    seen: set[str] = set()

    present_ids = [row[0] for row in present_label_rows(db, project_id, threshold)]
    rows = (
        db.query(LabelTaxonomy).filter(LabelTaxonomy.id.in_(present_ids)).all()
        if present_ids
        else []
    )
    for row in rows:
        name = row.name.lower()
        seen.add(name)
        if is_non_label(row.name):
            colors[row.id] = REJECTED_LABEL_COLOR
            colors[name] = REJECTED_LABEL_COLOR
            continue
        ranked.append((_taxonomic_sort_key(row), [row.id, name]))

    for category in present_category_rows(db, project_id, threshold):
        name = category.lower()
        # A category that also has a taxonomy row (MegaDetector's three)
        # is already ranked under that row's name key, and the lookup by
        # category name lands on it. A second slot would only spend a
        # colour on the same class twice.
        if name in seen:
            continue
        seen.add(name)
        ranked.append((_category_sort_key(name), [name]))

    ranked.sort(key=lambda item: item[0])
    for rank, (_sort_key, keys) in enumerate(ranked):
        color = SPECIES_PALETTE[rank % len(SPECIES_PALETTE)]
        for key in keys:
            colors[key] = color
    return colors
