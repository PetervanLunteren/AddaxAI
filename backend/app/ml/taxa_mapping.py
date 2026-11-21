"""
Custom SpeciesNet taxa list (.csv).

The file MegaDetector's `restrict_to_taxa_list` reads. Each row maps a
SpeciesNet taxon to a label the user picks, so it can rename (`aepyceros`
to "impala") and group (`accipitridae` and `strigiformes` both to "bird of
prey"). A project with a taxa list runs it instead of the country geofence.
AddaxAI only reads these files; Dan Morris's taxonomy mapping skill makes
them.

A file loads completely or is refused with the list of bad rows, the same
rows `restrict_to_taxa_list` would refuse. Rows without a Latin name are
skipped, as MegaDetector skips them: Dan's files keep a row for every
category of the user's own list, also Human, Blank and species SpeciesNet
does not know.
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

# The keys of a stored row (projects.taxa_mapping), AddaxAI's own, so a
# header rename only touches FORMATS.
LATIN = "latin"
LABEL = "label"

# The headers a file can have: its Latin column, then the label columns,
# the first filled one winning. The last label column is required.
FORMATS = (
    # The names agreed in AddaxAI#136 (2026-11-19). input_common and
    # output_alt_name are notes for the reader and not used.
    ("input_latin", ("output_name",)),
    # The files that exist today, as MegaDetector reads them: the label is
    # original_common when it is filled, common otherwise.
    ("latin", ("original_common", "common")),
)

# The names the pinned megadetector (10.0.28) reads, for the file
# `write_md_taxa_list` hands it. Once the pin moves to the release with
# the rename, these become input_latin and output_name.
MD_LATIN = "latin"
MD_COMMON = "common"

# The classification_model_id prefix of a retired list row: one a list no
# longer names but a verified detection or confirmed observation still
# uses (relink_taxa_list_labels). The row's id follows it.
RETIRED = "retired-taxa-list:"

# The cap on a custom label's name (LabelName in schemas/project.py).
MAX_LABEL_LENGTH = 200

# A refusal lists every bad row up to this many, each with at most this
# much of its Latin name: the form shows them all, and a shared file can
# be any size.
MAX_PROBLEMS = 100
MAX_SHOWN_LATIN = 80


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


def check_rows(
    rows: list[dict[str, str]],
    taxa: dict[str, dict[str, str]],
    line_numbers: list[int] | None = None,
) -> None:
    """
    Refuse a taxa list with any row `restrict_to_taxa_list` would refuse,
    and the rows it would silently drop: a row without a label, or a taxon
    listed twice. Raises TaxaListError naming every bad row, by the line
    of the file when `line_numbers` has it (the header is line 1).
    """
    problems: list[str] = []
    seen: set[str] = set()
    for n, row in zip(line_numbers or range(2, len(rows) + 2), rows, strict=True):
        latin = row[LATIN]
        label = row[LABEL]
        where = f"Row {n} ({latin[:MAX_SHOWN_LATIN] or 'empty'})"
        if not latin or not label:
            problems.append(f"{where}: needs both a Latin name and a label")
        elif len(label) > MAX_LABEL_LENGTH:
            problems.append(f"{where}: the label is longer than {MAX_LABEL_LENGTH} characters")
        elif latin in seen:
            problems.append(f"{where}: this taxon is listed twice")
        elif latin not in taxa:
            problems.append(f"{where}: not a SpeciesNet taxon")
        seen.add(latin)
    if not rows:
        problems.append("The file has no rows with a Latin name")
    if len(problems) > MAX_PROBLEMS:
        problems[MAX_PROBLEMS:] = [f"and {len(problems) - MAX_PROBLEMS} more"]
    if problems:
        raise TaxaListError(problems)


def parse_taxa_list(text: str, taxa: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """
    Read a taxa list .csv, in either format of FORMATS, into stored rows.

    Values are stripped and lowercased, as MegaDetector does, so a row's
    label is the label that ends up on the detections. Rows without a
    Latin name are skipped.
    """
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    columns = [c.strip() for c in reader.fieldnames or []]
    fmt = next((f for f in FORMATS if f[0] in columns), None)
    if fmt is None:
        raise TaxaListError(
            [f"Missing column: {FORMATS[0][0]} (or {FORMATS[1][0]} in the older format)"]
        )
    latin_column, label_columns = fmt
    if label_columns[-1] not in columns:
        raise TaxaListError([f"Missing column: {label_columns[-1]}"])
    reader.fieldnames = columns

    rows: list[dict[str, str]] = []
    line_numbers: list[int] = []
    for raw in reader:
        latin = (raw.get(latin_column) or "").strip().lower()
        if not latin:
            continue
        labels = ((raw.get(c) or "").strip() for c in label_columns)
        rows.append({LATIN: latin, LABEL: next((v for v in labels if v), "").lower()})
        line_numbers.append(reader.line_num)
    check_rows(rows, taxa, line_numbers)
    return rows


def normalize_rows(rows: Iterable[dict]) -> list[dict[str, str]]:
    """Keep the two keys of a stored row, stripped and lowercased."""
    return [
        {
            LATIN: (row.get(LATIN) or "").strip().lower(),
            LABEL: (row.get(LABEL) or "").strip().lower(),
        }
        for row in rows
    ]


def write_md_taxa_list(rows: list[dict[str, str]], path: Path) -> None:
    """Write the rows as the .csv the pinned megadetector reads."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([MD_LATIN, MD_COMMON])
        for row in rows:
            writer.writerow([row[LATIN], row[LABEL]])


def mapped_labels(rows: list[dict[str, str]]) -> list[str]:
    """The labels a taxa list produces, in file order, each once."""
    return list(dict.fromkeys(row[LABEL] for row in rows))


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
        chains_by_label.setdefault(row[LABEL], []).append(
            taxa[row[LATIN]]
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

    A verified detection (or a track box under a verified card) and a
    confirmed observation (signed off, or with
    a count a person set) stay as they are, as on every reprocess: they
    keep their row, and a stale row one of them uses is retired instead of
    dropped.
    """
    from sqlalchemy import and_, func, or_, select, union, update

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
    # Not human-decided (label_exclusion.is_human_decided): a track box
    # carries the label of a verified card.
    open_detection = and_(
        Detection.verified.is_(False),
        Detection.classification_method.is_distinct_from("track"),
    )
    open_observation = and_(
        EventObservation.human_count.is_(None),
        EventObservation.event_id.notin_(select(Event.id).where(Event.confirmed.is_(True))),
    )
    stale_ids = [row_id for row_id, _ in stale]
    if stale_ids:
        for model, is_open in ((Detection, open_detection), (EventObservation, open_observation)):
            db.execute(
                update(model)
                .where(model.label_taxonomy_id.in_(stale_ids), is_open)
                .values(label_taxonomy_id=None)
            )
        in_use = union(
            select(Detection.label_taxonomy_id).where(Detection.label_taxonomy_id.in_(stale_ids)),
            select(EventObservation.label_taxonomy_id).where(
                EventObservation.label_taxonomy_id.in_(stale_ids)
            ),
        )
        stale_rows = db.query(LabelTaxonomy).filter(LabelTaxonomy.id.in_(stale_ids))
        stale_rows.filter(LabelTaxonomy.id.notin_(in_use)).delete(synchronize_session=False)
        # A row still in use is retired: out of every name lookup
        # (model_scope matches the model id), so the model's own row of
        # the name wins again, and unique per row. A list that names the
        # label again takes it back (sync_taxa_mapping_rows).
        stale_rows.update(
            {LabelTaxonomy.classification_model_id: RETIRED + LabelTaxonomy.id},
            synchronize_session=False,
        )
    project_files = (
        select(File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project.id)
    )
    db.execute(
        update(Detection)
        .where(
            Detection.file_id.in_(project_files),
            func.lower(Detection.label).in_(names),
            open_detection,
        )
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
                open_observation,
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
    A retired row of the label is taken back, so what a person verified on
    it is under the label again.
    """
    from sqlalchemy import or_

    from app.models.label_taxonomy import LabelTaxonomy

    project_rows = db.query(LabelTaxonomy).filter(
        LabelTaxonomy.project_id == project_id,
        or_(
            LabelTaxonomy.classification_model_id == model_id,
            LabelTaxonomy.classification_model_id.startswith(RETIRED, autoescape=True),
        ),
    )
    # The model's row last, so it wins over a retired row of the name.
    existing = {
        row.name: row
        for row in sorted(project_rows, key=lambda r: r.classification_model_id == model_id)
    }
    for label, shared in shared_taxa(rows, taxa).items():
        entry = existing.get(label) or LabelTaxonomy(
            name=label,
            project_id=project_id,
            is_custom=False,
        )
        entry.classification_model_id = model_id
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
    # Written now: relink reads the rows back in SQL (no autoflush), and
    # a revived row must not look retired there.
    db.flush()
    logger.info(
        f"Taxa list for project {project_id}: {len(mapped_labels(rows))} labels"
    )
