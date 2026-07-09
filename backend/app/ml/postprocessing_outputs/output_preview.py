"""Compute a preview of what the Save outputs step will produce.

The Save step renders a folder-tree preview next to the option
controls so the user can see, before committing, the taxonomic
nesting that the run will produce, how multi-species placement
inflates the total, and how their species exclusion filter trims
it.

This module mirrors the placement logic in
``separate_folders._folder_for_file`` exactly: same threshold, same
main-species choice, same fallback folder, same observation-type
folder names, same exclusion semantics. The numbers it returns are
the same numbers the real postprocess run will produce.

The placement counts and byte totals are exact. ``by_media_tree`` is the
full on-disk tree for the chosen ``group_by`` + ``species_last``: the
species / observation folder and the preserved source subfolder combined
in the order the run writes them (path -> count). A capped ``root_files``
sample lists any loose files that land at the output root (only the "No
subfolders" mode with files at the source root).

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
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.observation_type import derive_observation_type
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
    NameMode,
    SeparateGroupBy,
    _leaf_name,
    _taxonomic_path_for_label,
    build_deployment_folders,
    build_event_primary_labels,
    source_subdir,
    video_still_name,
)


@dataclass
class OutputPreviewResult:
    """Aggregated counts used by the Save step's live preview.

    All counts reflect what the actual postprocess run would write —
    the placement counts in ``by_media_tree`` are real, not estimates,
    because the placement rules are deterministic from the DB state
    alone.

    The "dropped" counters reflect files that the user's species
    exclusion filter removes from the output. Surface separately so
    the UI can say "120 of 200 files will be in scope" honestly.
    """

    total_files: int = 0
    image_count: int = 0
    video_count: int = 0
    total_bytes: int = 0
    files_with_known_size: int = 0
    # File-level filter outcome — files dropped because every passing
    # identified detection (species, or builtin person / vehicle) is in
    # the exclusion set.
    dropped_by_filter: int = 0
    # Files that survive the filter. For non-animal files this is
    # always the file itself; for animal files it's the file when at
    # least one of its passing labels is included.
    in_scope_files: int = 0
    in_scope_image_count: int = 0
    in_scope_video_count: int = 0
    in_scope_bytes: int = 0
    # Slash-separated destination folder paths to placement counts, for
    # the exact ``group_by`` + ``species_last`` the user picked — i.e. the
    # real on-disk tree, species nesting AND the preserved source
    # subfolders combined in the chosen order. Keys look like
    # "mammalia/carnivora/canidae/canis/dog/cam01" (species first) or
    # "cam01/dog" (species last, flat). Non-animal files contribute their
    # observation-type folder ("person", "blank", ...). Under
    # ``group_by="none"`` the keys are just the source subfolders. The
    # frontend parses these into a nested tree and rolls counts up.
    by_media_tree: Counter = field(default_factory=Counter)
    # A capped sample of loose file names that land at the output root
    # (no folder at all): basenames, videos as their best-frame
    # "<stem>_still.jpg". Only non-empty when a file's full destination
    # path is empty — i.e. "No subfolders" mode with files at the source
    # root. Root total = in_scope_files - sum(by_media_tree).
    root_files: list[str] = field(default_factory=list)

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
            "by_media_tree": dict(self.by_media_tree),
            "root_files": list(self.root_files),
        }


# Cap on the root-file filename sample; matches the frontend's
# MAX_CHILDREN_PER_LEVEL so the list shows a few names then "… N more".
_ROOT_FILE_CAP = 4


def _output_basename(file_type: str, file_path: str) -> str:
    """The on-disk output name: images keep their name, videos become
    their best-frame ``<stem>_still.jpg``."""
    if file_type == "video":
        return video_still_name(file_path)
    return Path(file_path).name


def build_output_preview(
    db: Session,
    project_id: str,
    *,
    media_threshold: float,
    excluded_label_ids: frozenset[str] | None = None,
    include_empty: bool = True,
    name_mode: NameMode = "common",
    group_events: bool = True,
    group_by: SeparateGroupBy = "flat",
    species_last: bool = False,
) -> OutputPreviewResult:
    """Aggregate the counts the Save step needs for its live preview.

    Mirrors ``separate_folders`` exactly: every file is bucketed into its
    one destination folder; ``excluded_label_ids`` drops a file when
    every passing identified detection is excluded (counted in
    ``dropped_by_filter``); ``group_events`` buckets a whole event into
    one folder, the event's main species. ``media_threshold`` is the
    Save step's media-output confidence, applied exactly as the real run
    applies it (placement, blank skip, filter drops).

    ``group_by`` and ``species_last`` shape ``by_media_tree`` to the exact
    on-disk layout the run will write: the species / observation folder
    (nested taxonomy, a single species folder, or none) combined with the
    preserved source subfolder in the chosen order.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    threshold = media_threshold
    excluded = excluded_label_ids or frozenset()

    file_rows = db.execute(
        select(
            File.id,
            File.file_type,
            File.size_bytes,
            File.best_frame_path,
            File.file_path,
            File.deployment_id,
        )
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        # Stable order so the filename sample doesn't flicker between
        # preview refreshes.
        .order_by(File.file_path)
    ).all()

    # Source folders per deployment, to show the preserved subfolder
    # path in the "No subfolders" filename sample.
    dep_folders = build_deployment_folders(db, project_id)

    # Every detection in the project, confidence descending. One query
    # feeds two derivations: the passing identified detections (a label
    # or a builtin taxonomy id — species plus person / vehicle /
    # animal), and each file's effective observation type at the media
    # confidence (the stored ``observation_type`` column is derived at
    # the project threshold, which no longer matches media outputs).
    detection_rows = db.execute(
        select(
            Detection.file_id,
            Detection.label,
            Detection.label_taxonomy_id,
            Detection.category,
            Detection.confidence,
            Detection.verified,
        )
        .join(File, Detection.file_id == File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        .order_by(Detection.confidence.desc())
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

    def _row_passes(row) -> bool:
        return row.confidence >= threshold or row.verified

    # Passing identified detections per file, confidence-descending, and
    # the effective observation type per file (rows expose category /
    # confidence / verified, satisfying derive_observation_type).
    idents_per_file: dict[str, list] = {}
    dets_per_file: dict[str, list] = {}
    for det_row in detection_rows:
        dets_per_file.setdefault(det_row.file_id, []).append(det_row)
        if _row_passes(det_row) and (
            det_row.label or det_row.label_taxonomy_id
        ):
            idents_per_file.setdefault(det_row.file_id, []).append(det_row)
    obs_type_per_file: dict[str, str] = {
        file_id: derive_observation_type(rows, threshold)
        for file_id, rows in dets_per_file.items()
    }

    # Event-primary map, populated only when grouping is on.
    event_primary: dict[str, str] = {}
    if group_events:
        event_primary = build_event_primary_labels(
            db, project_id, threshold, excluded or None
        )

    result = OutputPreviewResult()
    result.total_files = len(file_rows)

    for row in file_rows:
        if row.file_type == "image":
            result.image_count += 1
        elif row.file_type == "video":
            result.video_count += 1

        # A video is written as its best-frame JPEG, so it contributes
        # the JPEG's size, not the full container's.
        written = _written_size(row)
        if written is not None:
            result.total_bytes += written
            result.files_with_known_size += 1

        idents = idents_per_file.get(row.id, [])

        # File-level filter: drop when every passing identified
        # detection is excluded (covers species and person / vehicle).
        if excluded and idents and all(
            _row_is_excluded(r) for r in idents
        ):
            result.dropped_by_filter += 1
            continue

        # Empties are skipped from the copies unless opted in, so they
        # add no placements or in-scope counts in the preview either.
        obs_type = obs_type_per_file.get(row.id, "blank")
        if not include_empty and obs_type == "blank":
            continue

        result.in_scope_files += 1
        if row.file_type == "image":
            result.in_scope_image_count += 1
        elif row.file_type == "video":
            result.in_scope_video_count += 1
        if written is not None:
            result.in_scope_bytes += written

        # The species / observation folder for this file, per group_by.
        if group_by == "none":
            folder = ""
        elif obs_type == "animal":
            # Main species = the event's main species when grouping, else
            # the file's most confident non-excluded label (idents is
            # confidence-descending).
            main = event_primary.get(row.id)
            if main is None:
                for r in idents:
                    if r.label and not _row_is_excluded(r):
                        main = r.label
                        break
            if main is None:
                # Animal known, species not — fallback "animal/" folder.
                folder = _FALLBACK_FOLDER
            elif group_by == "taxonomic":
                folder = _taxonomic_path_for_label(
                    main, taxonomy_by_name, name_mode
                )
            else:
                folder = _leaf_name(
                    main, taxonomy_by_name.get(main), name_mode
                )
        else:
            folder = _OBSERVATION_TYPE_FOLDER.get(
                obs_type, _FALLBACK_FOLDER
            )

        # Combine the species / observation folder with the preserved
        # source subfolder in the chosen order, exactly as
        # separate_into_folders lays it on disk. An empty full path means
        # the file lands loose at the output root.
        subdir = source_subdir(
            row.file_path, dep_folders.get(row.deployment_id)
        )
        ordered = (subdir, folder) if species_last else (folder, subdir)
        full = "/".join(p for p in ordered if p)
        if full:
            result.by_media_tree[full] += 1
        elif len(result.root_files) < _ROOT_FILE_CAP:
            result.root_files.append(
                _output_basename(row.file_type, row.file_path)
            )

    return result


def _written_size(row) -> int | None:
    """Bytes this file contributes to the output footprint.

    Videos are written as their best-frame JPEG, so use that file's size
    rather than the full container's. Best-effort: an unstattable best
    frame (slow / unmounted drive) returns ``None`` and the file just
    drops out of the byte total, like an image with no recorded size.
    """
    if row.file_type == "video":
        if not row.best_frame_path:
            return None
        try:
            return Path(row.best_frame_path).stat().st_size
        except OSError:
            return None
    return row.size_bytes


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
