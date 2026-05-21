"""Compute a preview of what the Save outputs step will produce.

The Save step renders a folder-tree preview next to the option
controls so the user can see, before committing, the taxonomic
nesting that the run will produce, how multi-species placement
inflates the total, and how their species exclusion filter trims
it.

This module mirrors the placement logic in
``separate_folders._label_plan_for_file`` exactly: same threshold,
same dedupe, same fallback folder, same observation-type folder
names, same exclusion semantics. The numbers it returns are the
same numbers the real postprocess run will produce.

Computation is three SELECTs: files, labelled detections, the
LabelTaxonomy chain for the project's classification model. The
Python loop after them is linear in (files + detections). Cheap
enough to call on every form change.

Disk size comes from ``File.size_bytes`` which is populated by the
ingestion step. Files with NULL size are excluded from the byte
total; the response surfaces both the byte total and the count of
files contributing to it so the UI can clarify when the estimate
is partial.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import (
    Deployment,
    Detection,
    File,
    LabelTaxonomy,
    Project,
)

from .separate_folders import (
    _FALLBACK_FOLDER,
    _OBSERVATION_TYPE_FOLDER,
    _taxonomic_path_for_label,
)


@dataclass
class OutputPreviewResult:
    """Aggregated counts used by the Save step's live preview.

    All counts reflect what the actual postprocess run would write —
    the multi-species inflation in ``by_taxonomic_tree`` is real,
    not an estimate, because the placement rules are deterministic
    from the DB state alone.

    The "dropped" counters reflect files that the user's species
    exclusion filter removes from the output. Surface separately so
    the UI can say "120 of 200 files will be in scope" honestly.
    """

    total_files: int = 0
    image_count: int = 0
    video_count: int = 0
    total_bytes: int = 0
    files_with_known_size: int = 0
    # File-level filter outcome — animal files dropped because every
    # passing label is in the exclusion set.
    dropped_by_filter: int = 0
    # Files that survive the filter. For non-animal files this is
    # always the file itself; for animal files it's the file when at
    # least one of its passing labels is included.
    in_scope_files: int = 0
    in_scope_image_count: int = 0
    in_scope_video_count: int = 0
    in_scope_bytes: int = 0
    # Slash-separated taxonomic paths to placement counts. Keys look
    # like "Mammalia/Carnivora/Canidae/Canis/dog". The frontend
    # parses these into a real nested tree. Non-animal files
    # contribute single-segment paths ("person", "vehicle", ...).
    by_taxonomic_tree: Counter = field(default_factory=Counter)
    # Flat single-segment placements: one folder per species label
    # (or per non-animal observation type, or the animal/ fallback).
    # Used when the user picks ``group_by="flat"`` on the Separate
    # card; keys are label strings, no slashes.
    by_flat: Counter = field(default_factory=Counter)
    # Distinct source files placed into more than one leaf of the
    # tree (multi-species shots). Each one inflates the total
    # placements by at least one above the in-scope file count.
    multi_species_files: int = 0

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "image_count": self.image_count,
            "video_count": self.video_count,
            "total_bytes": self.total_bytes,
            "files_with_known_size": self.files_with_known_size,
            "dropped_by_filter": self.dropped_by_filter,
            "in_scope_files": self.in_scope_files,
            "in_scope_image_count": self.in_scope_image_count,
            "in_scope_video_count": self.in_scope_video_count,
            "in_scope_bytes": self.in_scope_bytes,
            "by_taxonomic_tree": dict(self.by_taxonomic_tree),
            "by_flat": dict(self.by_flat),
            "multi_species_files": self.multi_species_files,
        }


def build_output_preview(
    db: Session,
    project_id: str,
    *,
    excluded_label_ids: frozenset[str] | None = None,
    include_empty: bool = True,
) -> OutputPreviewResult:
    """Aggregate the counts the Save step needs for its live preview.

    ``excluded_label_ids`` filters labelled detections out before
    bucketing. Animal files where every passing label is in the
    set are counted in ``dropped_by_filter`` and contribute no
    placements at any rank.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    threshold = project.detection_threshold
    excluded = excluded_label_ids or frozenset()

    file_rows = db.execute(
        select(
            File.id,
            File.observation_type,
            File.file_type,
            File.size_bytes,
        )
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    ).all()

    detection_rows = db.execute(
        select(
            Detection.file_id,
            Detection.label,
            Detection.label_taxonomy_id,
        )
        .join(File, Detection.file_id == File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        .where(File.observation_type == "animal")
        .where(Detection.label.isnot(None))
        .where(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
    ).all()

    taxonomy_by_name = _load_taxonomy_map(db, project)

    def _row_is_excluded(row) -> bool:
        if not excluded:
            return False
        if row.label_taxonomy_id and row.label_taxonomy_id in excluded:
            return True
        if row.label and row.label in excluded:
            return True
        return False

    # Aggregate labels per animal file, applying the exclusion filter.
    labels_per_file: dict[str, set[str]] = {}
    for det_row in detection_rows:
        if _row_is_excluded(det_row):
            continue
        labels_per_file.setdefault(det_row.file_id, set()).add(
            det_row.label
        )

    # File-level filter: animal files where EVERY passing detection
    # row is in the exclusion set should be marked dropped. Track
    # both totals and excluded counts per file so we can detect "all
    # detections excluded" without rebuilding the label set.
    det_total_per_file: dict[str, int] = {}
    det_excluded_per_file: dict[str, int] = {}
    if excluded:
        for det_row in detection_rows:
            det_total_per_file[det_row.file_id] = (
                det_total_per_file.get(det_row.file_id, 0) + 1
            )
            if _row_is_excluded(det_row):
                det_excluded_per_file[det_row.file_id] = (
                    det_excluded_per_file.get(det_row.file_id, 0) + 1
                )

    result = OutputPreviewResult()
    result.total_files = len(file_rows)

    for row in file_rows:
        if row.file_type == "image":
            result.image_count += 1
        elif row.file_type == "video":
            result.video_count += 1

        if row.size_bytes is not None:
            result.total_bytes += row.size_bytes
            result.files_with_known_size += 1

        # Determine whether the species filter drops this file:
        # animal file with at least one labelled passing detection,
        # and every such detection is excluded.
        dropped = False
        if excluded and row.observation_type == "animal":
            total = det_total_per_file.get(row.id, 0)
            blocked = det_excluded_per_file.get(row.id, 0)
            if total > 0 and total == blocked:
                dropped = True

        if dropped:
            result.dropped_by_filter += 1
            continue

        # Empties are skipped from the copies unless opted in, so they
        # add no placements or in-scope counts in the preview either.
        if not include_empty and row.observation_type == "blank":
            continue

        result.in_scope_files += 1
        if row.file_type == "image":
            result.in_scope_image_count += 1
        elif row.file_type == "video":
            result.in_scope_video_count += 1
        if row.size_bytes is not None:
            result.in_scope_bytes += row.size_bytes

        if row.observation_type == "animal":
            labels = labels_per_file.get(row.id, set())
            if not labels:
                # Animal file with no labelled detection passing the
                # threshold. Lands in the fallback "animal/" folder
                # under both grouping modes.
                result.by_taxonomic_tree[_FALLBACK_FOLDER] += 1
                result.by_flat[_FALLBACK_FOLDER] += 1
            else:
                _bucket_animal_labels(
                    result, labels, taxonomy_by_name
                )
        else:
            non_animal_folder = _OBSERVATION_TYPE_FOLDER.get(
                row.observation_type, _FALLBACK_FOLDER
            )
            result.by_taxonomic_tree[non_animal_folder] += 1
            result.by_flat[non_animal_folder] += 1

    return result


def _load_taxonomy_map(
    db: Session, project: Project
) -> dict[str, LabelTaxonomy]:
    """Preload the LabelTaxonomy rows for the project's classifier,
    keyed by species name. Empty when the project has no classifier."""
    model_id = project.classification_model_id
    if not model_id:
        return {}
    rows = db.execute(
        select(LabelTaxonomy).where(
            LabelTaxonomy.classification_model_id == model_id
        )
    ).scalars().all()
    return {row.name: row for row in rows}


def _bucket_animal_labels(
    result: OutputPreviewResult,
    labels: set[str],
    taxonomy_by_name: dict[str, LabelTaxonomy],
) -> None:
    """Bucket an animal file into both the taxonomic-tree counter
    and the flat counter, deduping placements per-mode so
    multi-species inflation matches the real run for whichever
    mode the user picks.

    Two species can collapse to a single leaf under ``taxonomic``
    (rare — same full chain) but always produce two distinct
    placements under ``flat`` (each species is its own folder),
    so the two bucket totals can differ.
    """
    if len(labels) > 1:
        result.multi_species_files += 1

    tree_paths: set[str] = set()
    for label in labels:
        tree_paths.add(
            _taxonomic_path_for_label(label, taxonomy_by_name)
        )
    for path in tree_paths:
        result.by_taxonomic_tree[path] += 1

    for label in labels:
        result.by_flat[label] += 1
