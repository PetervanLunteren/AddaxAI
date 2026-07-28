"""
Shared taxonomic rank constants and a Python-side resolver.

Dashboard stats use a SQL CASE expression (statistics._rank_display_label)
and the performance matrix uses a Python resolver. Both live here so the
bucket names, the rank column mapping, and the ordered option list stay
in one place; a future plot can pick whichever side fits its query shape.
"""

from typing import Literal, Protocol

RANK_COLUMNS = {
    "class": "taxon_class",
    "order": "taxon_order",
    "family": "taxon_family",
    "genus": "taxon_genus",
    "species": "taxon_species",
}

TaxonomicRank = Literal["all", "class", "order", "family", "genus", "species"]

HIGHER_LEVEL_TAXA = "Higher-level taxa"
NO_TAXONOMY = "No taxonomy"
MOST_SPECIFIC: TaxonomicRank = "all"

# Ordered list of (value, label) pairs. The frontend mirrors this.
# Strings are the contract; keep in sync with frontend/src/lib/taxonomic-rank.ts.
RANK_OPTIONS: list[tuple[TaxonomicRank, str]] = [
    ("all", "Most specific"),
    ("species", "Species"),
    ("genus", "Genus"),
    ("family", "Family"),
    ("order", "Order"),
    ("class", "Class"),
]


class _TaxonRow(Protocol):
    """Minimal shape a taxonomy row must satisfy for resolve_rank."""

    name: str
    scientific_name: str | None
    taxon_class: str | None
    taxon_order: str | None
    taxon_family: str | None
    taxon_genus: str | None
    taxon_species: str | None


def resolve_rank(
    *,
    category: str,
    label: str | None,
    scientific_name: str | None,
    taxonomy_row: _TaxonRow | None,
    rank: TaxonomicRank | None,
) -> str:
    """
    Python-side mirror of statistics._rank_display_label.

    Non-animal categories return the category. "Most specific" (None or
    "all") falls back `scientific_name -> label -> category`. A specific
    rank returns the rank value when the row has it, or buckets to
    "Higher-level taxa" / "No taxonomy" using the same rules as the
    dashboard's SQL CASE.
    """
    if category in ("person", "vehicle"):
        return category
    if rank is None or rank == "all":
        return scientific_name or label or category
    if rank not in RANK_COLUMNS:
        # Unknown rank string: do the safest fallback rather than crashing.
        return label or category
    if taxonomy_row is None:
        return NO_TAXONOMY
    value = _rank_value(taxonomy_row, rank)
    if value is not None:
        return value
    if taxonomy_row.taxon_class is not None:
        return HIGHER_LEVEL_TAXA
    return NO_TAXONOMY


def _rank_value(taxonomy_row: _TaxonRow, rank: TaxonomicRank) -> str | None:
    if rank == "species":
        # Species rank uses scientific_name to avoid species-epithet
        # collisions across genera (e.g. two "pardus" species).
        if taxonomy_row.taxon_species is None:
            return None
        return taxonomy_row.scientific_name or taxonomy_row.name
    raw = getattr(taxonomy_row, RANK_COLUMNS[rank], None)
    return to_display_case(raw)


def to_display_case(value: str | None) -> str | None:
    """
    Capitalise the first letter of a stored taxon value for display.

    The CSV importer normalises taxon_* columns to lowercase, but
    family / genus / order / class names are conventionally written
    with an initial capital. This is the single point where that
    normalisation happens, used on both the matrix (Python) and the
    dashboard (via a SQL equivalent) so views stay consistent.
    """
    if not value:
        return value
    return value[0].upper() + value[1:]
