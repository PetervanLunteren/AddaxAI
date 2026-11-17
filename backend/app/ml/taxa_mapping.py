"""
Custom SpeciesNet taxa list (.csv).

The file MegaDetector's `restrict_to_taxa_list` reads. Each row maps a
SpeciesNet taxon to a label the user picks, so it can rename (`aepyceros`
to "impala") and group (`accipitridae` and `strigiformes` both to "bird of
prey"). A project with a taxa list runs it instead of the country geofence.
AddaxAI only reads these files; Dan Morris's taxonomy mapping skill makes
them.

A file loads completely or is refused with the list of bad rows, the same
rows `restrict_to_taxa_list` would refuse.
"""

import csv
import io
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.taxonomic_rollup import (
    TAXONOMY_LEVELS,
    format_common_name,
    format_scientific_name_from_taxonomy_row,
)

logger = get_logger(__name__)

# Column names, here and nowhere else. These are the names from Dan's
# rename in AddaxAI#136. The other two columns of that format,
# input_common and user_alt_name, are notes for the reader and not used.
INPUT_LATIN = "input_latin"
USER_TAXON_NAME = "user_taxon_name"

# The names the pinned megadetector (10.0.28) reads, for the file
# `write_md_taxa_list` hands it. Once the pin moves to the release with
# the rename, these become INPUT_LATIN and USER_TAXON_NAME.
MD_LATIN = "latin"
MD_COMMON = "common"

# The cap on a custom label's name (LabelName in schemas/project.py).
MAX_LABEL_LENGTH = 200


class TaxaListError(ValueError):
    """A taxa list that cannot be used, with one line per problem."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class SharedTaxon:
    """The lowest taxon all of a label's rows have in common."""

    level: str
    ranks: dict[str, str]


def find_taxonomy_release_file(model_dir: Path) -> Path | None:
    """The SpeciesNet taxonomy file ('taxonomy_release.*.txt') in a model dir."""
    matches = sorted(model_dir.glob("taxonomy_release.*.txt"))
    return matches[-1] if matches else None


@lru_cache(maxsize=4)
def load_speciesnet_taxa(taxonomy_path: str) -> dict[str, dict[str, str]]:
    """
    Map every SpeciesNet Latin name to its ranks, the way
    `restrict_to_taxa_list` keys its taxonomy.

    The key is the binomial for a species ("aepyceros melampus") and the
    lowest filled rank otherwise ("aepyceros", "bovidae"). Every rank that
    appears as a parent is a taxon too, also when the file has no line of
    its own for it, as MegaDetector adds those before matching.
    """
    taxa: dict[str, dict[str, str]] = {}
    with open(taxonomy_path, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split(";")
            if len(tokens) != 7:
                continue
            ranks = dict(zip(TAXONOMY_LEVELS, tokens[1:6], strict=True))
            if not ranks["class"] or ranks["class"] == "no cv result":
                continue
            filled = [lvl for lvl in TAXONOMY_LEVELS if ranks[lvl]]
            for i, lvl in enumerate(filled):
                chain = {r: ranks[r] for r in filled[: i + 1]}
                key = (
                    f"{ranks['genus']} {ranks['species']}"
                    if lvl == "species"
                    else ranks[lvl]
                )
                taxa.setdefault(key, chain)
    return taxa


def check_rows(rows: list[dict[str, str]], taxa: dict[str, dict[str, str]]) -> None:
    """
    Refuse a taxa list with any row `restrict_to_taxa_list` would refuse,
    and the rows it would silently drop or skip: an empty field, or a
    taxon listed twice. Raises TaxaListError naming every bad row.
    """
    problems: list[str] = []
    seen: set[str] = set()
    # Numbered as a spreadsheet shows the file: the header is row 1.
    # ponytail: a blank line in the file shifts the rows below it by one.
    for n, row in enumerate(rows, start=2):
        latin = row[INPUT_LATIN]
        name = row[USER_TAXON_NAME]
        where = f"Row {n} ({latin or 'empty'})"
        if not latin or not name:
            problems.append(f"{where}: {INPUT_LATIN} and {USER_TAXON_NAME} are both required")
        elif len(name) > MAX_LABEL_LENGTH:
            problems.append(
                f"{where}: {USER_TAXON_NAME} is longer than {MAX_LABEL_LENGTH} characters"
            )
        elif latin in seen:
            problems.append(f"{where}: this taxon is listed twice")
        elif latin not in taxa:
            problems.append(f"{where}: not a SpeciesNet taxon")
        seen.add(latin)
    if not rows:
        problems.append("The file has no rows")
    if problems:
        raise TaxaListError(problems)


def parse_taxa_list(text: str, taxa: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """
    Read a taxa list .csv into rows of the two columns AddaxAI uses.

    Values are stripped and lowercased, as MegaDetector does, so a row's
    user_taxon_name is the label that ends up on the detections.
    """
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    columns = [c.strip() for c in reader.fieldnames or []]
    missing = [c for c in (INPUT_LATIN, USER_TAXON_NAME) if c not in columns]
    if missing:
        raise TaxaListError([f"Missing column: {c}" for c in missing])
    reader.fieldnames = columns
    rows = normalize_rows(
        row
        for row in reader
        if any(v.strip() for v in row.values() if isinstance(v, str))
    )
    check_rows(rows, taxa)
    return rows


def normalize_rows(rows: Iterable[dict]) -> list[dict[str, str]]:
    """Keep the two columns AddaxAI uses, stripped and lowercased."""
    return [
        {
            INPUT_LATIN: (row.get(INPUT_LATIN) or "").strip().lower(),
            USER_TAXON_NAME: (row.get(USER_TAXON_NAME) or "").strip().lower(),
        }
        for row in rows
    ]


def write_md_taxa_list(rows: list[dict[str, str]], path: Path) -> None:
    """Write the rows as the .csv the pinned megadetector reads."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([MD_LATIN, MD_COMMON])
        for row in rows:
            writer.writerow([row[INPUT_LATIN], row[USER_TAXON_NAME]])


def mapped_labels(rows: list[dict[str, str]]) -> list[str]:
    """The labels a taxa list produces, in file order, each once."""
    return list(dict.fromkeys(row[USER_TAXON_NAME] for row in rows))


def shared_taxa(
    rows: list[dict[str, str]], taxa: dict[str, dict[str, str]]
) -> dict[str, SharedTaxon]:
    """
    For each label, the lowest taxon its rows share: "impala" on the genus
    `aepyceros` is a genus, "bird of prey" on a family and an order of
    birds is Aves, and a label over a mammal and a bird is the kingdom.
    """
    chains_by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        chains_by_label.setdefault(row[USER_TAXON_NAME], []).append(
            taxa[row[INPUT_LATIN]]
        )

    result: dict[str, SharedTaxon] = {}
    for label, chains in chains_by_label.items():
        ranks: dict[str, str] = {}
        for lvl in TAXONOMY_LEVELS:
            values = {chain.get(lvl) for chain in chains}
            value = values.pop() if len(values) == 1 else None
            if value is None:
                break
            ranks[lvl] = value
        level = TAXONOMY_LEVELS[len(ranks) - 1] if ranks else "kingdom"
        result[label] = SharedTaxon(level=level, ranks=ranks)
    return result


def taxa_for_model(model_id: str | None) -> dict[str, dict[str, str]]:
    """The SpeciesNet taxa of a downloaded classification model, or
    TaxaListError if it has none."""
    from app.core.config import get_settings

    if not model_id:
        raise TaxaListError(["A taxa list needs a classification model"])
    model_dir = get_settings().models_dir / "cls" / model_id
    path = find_taxonomy_release_file(model_dir) if model_dir.exists() else None
    if path is None:
        raise TaxaListError(
            ["This model has no SpeciesNet taxonomy file, so it cannot use a taxa list"]
        )
    return load_speciesnet_taxa(str(path))


def clear_location(project) -> None:
    """A taxa list replaces the country: no country, state or exclusions."""
    project.country_code = None
    project.state_code = None
    project.excluded_classes = []


def settle_project_taxa_mapping(db: Session, project) -> None:
    """
    Hold the rule of a project with a taxa list, before it is committed:
    no country and no exclusions (otherwise the export would still filter
    on a list the user can no longer see), the rows checked against the
    project's model, and every label with its taxonomy row. Without a list
    it only drops the rows an earlier one left. Raises TaxaListError on a
    bad list.
    """
    labels: list[str] = []
    if project.taxa_mapping:
        clear_location(project)
        rows = normalize_rows(project.taxa_mapping["rows"])
        project.taxa_mapping = {**project.taxa_mapping, "rows": rows}
        taxa = taxa_for_model(project.classification_model_id)
        check_rows(rows, taxa)
        sync_taxa_mapping_rows(db, project.id, project.classification_model_id, rows, taxa)
        labels = mapped_labels(rows)
    relink_taxa_list_labels(db, project, labels)


def relink_taxa_list_labels(db: Session, project, labels: list[str]) -> None:
    """
    Drop the project's rows that the current list (and model) no longer
    has, and relink the project's detections on every label a list row
    names now or named before. Otherwise a removed "impala" genus row keeps
    winning over SpeciesNet's impala species, and detections linked before
    a list named their label keep the model's row. Linking only touches
    detections without a row, so they are unlinked first.
    """
    from sqlalchemy import func, or_, select, update

    from app.ml.taxonomy_db import batch_resolve_taxonomy_ids, link_detections_to_taxonomy
    from app.models import (
        Deployment,
        Detection,
        Event,
        EventObservation,
        File,
        LabelTaxonomy,
    )

    stale = (
        db.query(LabelTaxonomy.id, LabelTaxonomy.name)
        .filter(
            LabelTaxonomy.project_id == project.id,
            or_(
                LabelTaxonomy.classification_model_id.is_distinct_from(
                    project.classification_model_id
                ),
                LabelTaxonomy.name.notin_(labels),
            ),
        )
        .all()
    )
    names = set(labels) | {name for _, name in stale}
    if not names:
        return
    stale_ids = [row_id for row_id, _ in stale]
    if stale_ids:
        for model in (Detection, EventObservation):
            db.execute(
                update(model)
                .where(model.label_taxonomy_id.in_(stale_ids))
                .values(label_taxonomy_id=None)
            )
        db.query(LabelTaxonomy).filter(LabelTaxonomy.id.in_(stale_ids)).delete(
            synchronize_session=False
        )
    project_files = (
        select(File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project.id)
    )
    db.execute(
        update(Detection)
        .where(Detection.file_id.in_(project_files), func.lower(Detection.label).in_(names))
        .values(label_taxonomy_id=None)
    )
    db.flush()
    link_detections_to_taxonomy(project.id, db)

    # Observations by the same rule, so counts and exports agree with the
    # detections before the events are next rebuilt.
    resolved = batch_resolve_taxonomy_ids(
        sorted(names), project.classification_model_id, project.id, db
    )
    project_events = (
        select(Event.id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .where(Deployment.project_id == project.id)
    )
    for name in names:
        db.execute(
            update(EventObservation)
            .where(
                EventObservation.event_id.in_(project_events),
                func.lower(EventObservation.label) == name,
            )
            .values(label_taxonomy_id=resolved[name][0] if name in resolved else None)
        )


def sync_taxa_mapping_rows(
    db: Session,
    project_id: str,
    model_id: str,
    rows: list[dict[str, str]],
    taxa: dict[str, dict[str, str]],
) -> None:
    """
    Give every label of a taxa list its own label_taxonomy row, scoped to
    the project, with the ranks of the taxon its rows share. The project's
    row wins over the model's own row of the same name, so a custom
    "impala" on the genus does not read as SpeciesNet's impala species.
    """
    from app.models.label_taxonomy import LabelTaxonomy

    existing = {
        row.name: row
        for row in db.query(LabelTaxonomy).filter(
            LabelTaxonomy.classification_model_id == model_id,
            LabelTaxonomy.project_id == project_id,
        )
    }
    for label, shared in shared_taxa(rows, taxa).items():
        entry = existing.get(label) or LabelTaxonomy(
            classification_model_id=model_id,
            name=label,
            project_id=project_id,
            is_custom=False,
        )
        entry.taxon_class = shared.ranks.get("class")
        entry.taxon_order = shared.ranks.get("order")
        entry.taxon_family = shared.ranks.get("family")
        entry.taxon_genus = shared.ranks.get("genus")
        entry.taxon_species = shared.ranks.get("species")
        entry.level = shared.level
        entry.common_name = format_common_name(label)
        entry.scientific_name = (
            "Animalia"
            if shared.level == "kingdom"
            else format_scientific_name_from_taxonomy_row(
                label,
                entry.taxon_genus,
                entry.taxon_species,
                entry.taxon_family,
                entry.taxon_order,
                entry.taxon_class,
            )
        )
        if label not in existing:
            db.add(entry)
    logger.info(
        f"Taxa list for project {project_id}: {len(mapped_labels(rows))} labels"
    )
