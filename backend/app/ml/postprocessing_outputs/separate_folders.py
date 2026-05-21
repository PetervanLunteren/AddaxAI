"""Reorganise a project's media files into ``<output_root>/<label>/``.

The legacy-AddaxAI mental model for postprocessing: browse the run's
output as ``output_root/dog/``, ``output_root/leopard/``,
``output_root/blank/``, …, in the file manager. This module re-creates
that experience on top of the modern DB pipeline.

Multi-species behaviour (hardcoded, no opt-out):

A file with detections of multiple distinct species lands in every
matching folder. So an image with ``dog`` + ``wolf`` appears in both
``output_root/dog/`` and ``output_root/wolf/``. This is the convention
the wider camera-trap ecosystem uses (camtrapR's ``getSpeciesImages``,
the Data Carpentry guide) — the alternative of a single "top label
wins" placement is the headline UX complaint from beta testers because
information gets dropped at the file system level. We pay the disk
cost; the user finds their image when they look for any of its
species.

Modes:

- ``copy``: every matching folder gets its own copy of the source.
- ``move``: the file moves to the **primary** label folder (top
  confidence) and gets copied into the others. The DB's
  ``File.file_path`` is rewritten to the new primary location so the
  verify UI keeps working. Source folder loses the media.

Grouping:

- ``taxonomic`` (default): a nested chain
  ``Class/Order/Family/Genus/species/`` derived from each label's
  ``LabelTaxonomy`` entry. Labels with no taxonomy mapping fall to
  ``Other/<label>/``.
- ``flat``: one folder per species label at the root (``dog/``,
  ``leopard/`` …). Best when only a few species are in play.

Non-animal observation types (human / vehicle / blank / unknown /
unclassified) route to a single fixed folder in both modes.

Collision handling: ``output_root/<label>/IMG_001.jpg`` already
exists → append ``_2``, ``_3``, … per destination folder until the
name is unique. Original files are never overwritten.

Videos: the source video file is reorganised. The per-video best
frame JPEG (under ``.addaxai/projects/<pid>/video_frames/``) is NOT
moved here; it stays in the project cache and is consumed by
``annotated_copies`` if the user asked for visualised / anonymised
output.

Each successful placement is recorded on the shared ``OutputContext``
so downstream modules (``annotated_copies``, the CSV / XLSX writers)
write into / reference the same path the separated file landed at.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.logging_config import get_logger
from app.models import Deployment, File, LabelTaxonomy, Project

from ._exif_writer import ExifBatch, build_tag_set, is_image_path
from ._label_filter import (
    file_is_dropped_by_filter,
    passing_labels_for_file,
)
from ._output_context import OutputContext

logger = get_logger(__name__)


SeparateMode = Literal["copy", "move"]
# ``taxonomic`` nests Class > Order > Family > Genus > species; labels
# with no taxonomy mapping fall to ``Other/<label>/``. ``flat`` collapses
# to one folder per species label at the root.
SeparateGroupBy = Literal["taxonomic", "flat", "none"]
# Bucket name for labels with no taxonomy mapping under ``taxonomic``.
UNRANKED_FOLDER = "Other"
# Order of taxonomic ranks. Path construction walks these in order and
# stops at the label's own rank; missing intermediate ancestors pad
# with ``UNRANKED_FOLDER`` so the depth stays consistent.
_TAXONOMIC_RANK_ORDER: tuple[str, ...] = (
    "class",
    "order",
    "family",
    "genus",
    "species",
)


@dataclass
class SeparateFoldersResult:
    """Summary of a separate-into-folders run.

    Counters reflect placements, not source files: a file that ends up
    in three folders counts three times. ``multi_placement_count`` is
    the count of distinct source files that appeared in more than one
    folder, so the UI can call that out when ``copy`` mode inflates
    the placement total beyond the file total.

    ``skipped_excluded`` counts animal files whose every passing
    species label was in the exclusion set — the user asked for the
    species to disappear from outputs, so the file is skipped entirely.
    """

    copied_count: int = 0
    moved_count: int = 0
    skipped_missing_source: int = 0
    skipped_excluded: int = 0
    renamed_count: int = 0
    multi_placement_count: int = 0
    by_label: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    @property
    def written_count(self) -> int:
        """Total placements across copy and move modes."""
        return self.copied_count + self.moved_count

    def to_dict(self) -> dict:
        return {
            "copied_count": self.copied_count,
            "moved_count": self.moved_count,
            "written_count": self.written_count,
            "skipped_missing_source": self.skipped_missing_source,
            "skipped_excluded": self.skipped_excluded,
            "renamed_count": self.renamed_count,
            "multi_placement_count": self.multi_placement_count,
            "by_label": dict(self.by_label),
            "errors": list(self.errors),
        }


# Non-animal observation types route to a fixed folder name at the
# root of the output. Species-level taxonomy doesn't apply to
# detector-level categories (person / vehicle / blank).
_OBSERVATION_TYPE_FOLDER: dict[str, str] = {
    "animal": "animal",
    "human": "person",  # match the rest of the UI's wording
    "vehicle": "vehicle",
    "blank": "blank",
    "unknown": "unknown",
    "unclassified": "unclassified",
}

# Fallback folder name when nothing else fits.
_FALLBACK_FOLDER = "animal"


@dataclass(frozen=True)
class _LabelPlan:
    """Destination folders for one file: a primary plus zero or more
    additional folders the file should also appear in.

    ``move`` mode places the file at the primary destination and copies
    to the others (the file can only move once). ``copy`` mode treats
    primary and others identically — N placements of the same source.
    """

    primary: str
    others: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return (self.primary, *self.others)


def _build_taxonomy_map(
    db: Session, project: Project
) -> dict[str, LabelTaxonomy]:
    """Preload the ``LabelTaxonomy`` rows for the project's classifier
    so per-file plans resolve the taxonomic chain without a query per
    file. Empty dict when the project has no classifier — every label
    then falls back to ``Other/<label>``.
    """
    model_id = project.classification_model_id
    if not model_id:
        return {}
    rows = db.execute(
        select(LabelTaxonomy).where(
            LabelTaxonomy.classification_model_id == model_id
        )
    ).scalars().all()
    return {row.name: row for row in rows}


def _taxonomic_path_for_label(
    label: str,
    taxonomy_map: dict[str, LabelTaxonomy],
) -> str:
    """Build the nested taxonomic path for one label.

    Returns a slash-separated path of the form ``<ancestors>/<label>``.
    Ancestors are the ``LabelTaxonomy`` columns above the label's own
    rank; missing intermediate ranks are padded with
    ``UNRANKED_FOLDER`` so the on-disk depth is consistent across
    siblings. The label itself is always the leaf segment (what the
    classifier emitted, not ``taxon_species``). Labels with no
    taxonomy row fall back to ``Other/<label>``.
    """
    taxon = taxonomy_map.get(label)
    if taxon is None:
        return f"{UNRANKED_FOLDER}/{label}"

    label_level = (taxon.level or "").lower()
    if label_level not in _TAXONOMIC_RANK_ORDER:
        return label or UNRANKED_FOLDER

    label_level_idx = _TAXONOMIC_RANK_ORDER.index(label_level)

    parts: list[str] = []
    for rank in _TAXONOMIC_RANK_ORDER[:label_level_idx]:
        value = getattr(taxon, f"taxon_{rank}", None)
        parts.append(value if value else UNRANKED_FOLDER)
    parts.append(label)
    return "/".join(parts)


def _label_plan_for_file(
    db: Session,
    file: File,
    threshold: float,
    taxonomy_map: dict[str, LabelTaxonomy],
    group_by: SeparateGroupBy = "taxonomic",
    excluded_label_ids: frozenset[str] | None = None,
) -> _LabelPlan:
    """Compute the set of destination paths for a single file.

    Non-animal files route to their fixed observation-type folder
    under both grouping modes. Animal files build a slash-path per
    passing label (``taxonomic``) or a single-segment folder per
    label (``flat``). Multi-species files end up in multiple distinct
    destinations either way. Animal files with no surviving labels
    fall back to ``animal/``.
    """
    # "none": no subfolders. Every file lands flat in the output root
    # (a single copy each, regardless of species). Primary "" joins to
    # the output root unchanged.
    if group_by == "none":
        return _LabelPlan(primary="", others=())

    obs_type = file.observation_type
    fixed_folder = _OBSERVATION_TYPE_FOLDER.get(obs_type)

    if obs_type != "animal":
        return _LabelPlan(
            primary=fixed_folder or _FALLBACK_FOLDER, others=()
        )

    labels = passing_labels_for_file(
        db, file, threshold, excluded_label_ids
    )

    if not labels:
        # Animal file with no species-labelled detections. Caller
        # already skipped files where every label was excluded, so
        # this is the "we know it's an animal but not what kind"
        # case — lands in the top-level animal/ folder.
        return _LabelPlan(primary=_FALLBACK_FOLDER, others=())

    folder_names: list[str] = []
    seen: set[str] = set()
    for label in labels:
        path = (
            _taxonomic_path_for_label(label, taxonomy_map)
            if group_by == "taxonomic"
            else label
        )
        if path not in seen:
            folder_names.append(path)
            seen.add(path)

    primary = folder_names[0]
    others = tuple(folder_names[1:])
    return _LabelPlan(primary=primary, others=others)


def _unique_destination(target_dir: Path, source_name: str) -> tuple[Path, bool]:
    """Return a path inside ``target_dir`` that doesn't already exist.

    Appends ``_2``, ``_3``, … before the suffix until the name is free.
    Returns the path plus a flag indicating whether a rename happened.
    """
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix
    candidate = target_dir / source_name
    if not candidate.exists():
        return candidate, False
    counter = 2
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate, True
        counter += 1


def _place_primary(
    source: Path, destination: Path, mode: SeparateMode
) -> None:
    """Execute the copy / move for the primary placement.

    Raises ``OSError`` on failure; the caller records and moves on.
    """
    if mode == "copy":
        # copy2 preserves mtime. Important for downstream tools that
        # read modification time as a proxy for capture time.
        shutil.copy2(source, destination)
    elif mode == "move":
        # shutil.move falls back to copy+delete across filesystems.
        # The DB rewrite happens after a successful move so partial
        # failures leave the DB consistent with what's on disk.
        shutil.move(str(source), str(destination))
    else:
        raise ValueError(f"unsupported separate_folders mode: {mode!r}")


def _place_extra(
    source: Path,
    primary_dest: Path,
    extra_dest: Path,
    mode: SeparateMode,
) -> None:
    """Execute an additional placement for a multi-species file.

    ``source`` is the original on-disk file (still present under
    ``copy``, gone under ``move``). ``primary_dest`` is where the
    file's primary placement was just written, used as the copy
    source under ``move`` (which can only move once).
    """
    copy_source = primary_dest if mode == "move" else source
    shutil.copy2(copy_source, extra_dest)


def separate_into_folders(
    db: Session,
    project_id: str,
    ctx: OutputContext,
    *,
    mode: SeparateMode = "copy",
    group_by: SeparateGroupBy = "taxonomic",
    include_empty: bool = True,
    excluded_label_ids: frozenset[str] | None = None,
) -> SeparateFoldersResult:
    """Reorganise every file in the project into subdirectories under
    ``ctx.output_root``.

    Under ``group_by="taxonomic"`` (default) animal files land in a
    nested chain ``<Class>/<Order>/<Family>/<Genus>/<species>/``;
    under ``group_by="flat"`` each animal file lands in a
    single-segment folder named after the species label. Non-animal
    files always land in their fixed observation-type folder. Files
    with multiple species labels appear in every matching destination.

    ``mode`` controls placement at the primary destination
    (``copy`` / ``move``); extras are always copies (a file can only
    move once and the primary already holds the moved bytes).

    ``excluded_label_ids`` filters labelled detections: animal files
    where every passing label is in the set are skipped entirely
    (counted in ``skipped_excluded``). Files with mixed inclusion
    still go through but only land in folders for their non-excluded
    labels.

    Each successful placement is recorded on ``ctx`` so downstream
    modules can find the file on disk without re-reading
    ``File.file_path``.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir = ctx.output_root
    target_dir.mkdir(parents=True, exist_ok=True)
    threshold = project.detection_threshold
    taxonomy_map = _build_taxonomy_map(db, project)

    files = db.execute(
        select(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    ).scalars().all()

    result = SeparateFoldersResult()

    # One ExifToolHelper for the whole batch so the underlying
    # exiftool process stays alive across writes — start-up is the
    # expensive part.
    with ExifBatch() as exif_batch:
        for file in files:
            source = Path(file.file_path)
            if not source.exists():
                result.skipped_missing_source += 1
                result.errors.append(
                    f"Source file no longer on disk: {file.file_path}"
                )
                continue

            if file_is_dropped_by_filter(
                db, file, threshold, excluded_label_ids
            ):
                result.skipped_excluded += 1
                continue

            # Empties are skipped from the copies unless opted in.
            if not include_empty and file.observation_type == "blank":
                continue

            plan = _label_plan_for_file(
                db,
                file,
                threshold,
                taxonomy_map,
                group_by,
                excluded_label_ids,
            )

            # Primary placement.
            primary_dir = target_dir / plan.primary
            primary_dir.mkdir(parents=True, exist_ok=True)
            primary_dest, renamed_primary = _unique_destination(
                primary_dir, source.name
            )
            try:
                _place_primary(source, primary_dest, mode)
            except OSError as e:
                result.errors.append(f"Failed to {mode} {source}: {e}")
                logger.exception(
                    f"separate_folders: {mode} failed for {source}"
                )
                continue

            if mode == "move":
                file.file_path = str(primary_dest)
                db.commit()
                result.moved_count += 1
            else:
                result.copied_count += 1
            result.by_label[plan.primary] += 1
            if renamed_primary:
                result.renamed_count += 1
            ctx.record(file.id, primary_dest)

            # Silent EXIF on the primary placement when the file is an
            # image format that carries EXIF. ``annotated_copies`` will
            # overwrite this if the user also asked for visualised /
            # anonymised output, which is fine — the tag set is the
            # same.
            if is_image_path(primary_dest):
                tag_set = build_tag_set(
                    db,
                    file,
                    project,
                    APP_VERSION,
                    excluded_label_ids=excluded_label_ids,
                )
                if tag_set is not None:
                    try:
                        exif_batch.write(primary_dest, tag_set)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"separate_folders: EXIF write failed for "
                            f"{primary_dest}: {e}"
                        )

            # Extra placements for multi-species files.
            extras_written = 0
            for extra_label in plan.others:
                extra_dir = target_dir / extra_label
                extra_dir.mkdir(parents=True, exist_ok=True)
                extra_dest, renamed_extra = _unique_destination(
                    extra_dir, source.name
                )
                try:
                    _place_extra(
                        source, primary_dest, extra_dest, mode
                    )
                except OSError as e:
                    result.errors.append(
                        f"Failed extra placement {extra_dest}: {e}"
                    )
                    logger.exception(
                        f"separate_folders: extra placement failed for "
                        f"{extra_dest}"
                    )
                    continue

                result.copied_count += 1
                result.by_label[extra_label] += 1
                if renamed_extra:
                    result.renamed_count += 1
                extras_written += 1
                ctx.record(file.id, extra_dest)

                if is_image_path(extra_dest):
                    # Reuse the same tag set; multi-species files
                    # carry the same detection summary in every folder.
                    tag_set = build_tag_set(
                        db,
                        file,
                        project,
                        APP_VERSION,
                        excluded_label_ids=excluded_label_ids,
                    )
                    if tag_set is not None:
                        try:
                            exif_batch.write(extra_dest, tag_set)
                        except Exception as e:  # noqa: BLE001
                            logger.warning(
                                f"separate_folders: EXIF write failed "
                                f"for {extra_dest}: {e}"
                            )

            if extras_written > 0:
                result.multi_placement_count += 1

    logger.info(
        f"separate_folders: project={project_id} mode={mode} "
        f"group_by={group_by} written={result.written_count} "
        f"multi={result.multi_placement_count} "
        f"renamed={result.renamed_count} "
        f"missing={result.skipped_missing_source} "
        f"excluded={result.skipped_excluded}"
    )
    return result
