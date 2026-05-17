"""Dedicated EXIF-prediction writer for the folder-run save step.

The other postprocess modules (separate / visualised / blur) silently
embed detection metadata into the copies they create. This module is
the explicit, user-facing option for the case where the user wants
EXIF metadata WITHOUT also producing reorganised / annotated / blurred
copies — or wants the original source files to carry the metadata
permanently.

Two modes:

- ``overwrite``: writes EXIF tags onto the source files in place.
  Permanent modification of the user's archive. Matches legacy
  AddaxAI and TrapTagger behaviour.

- ``copy``: copies every source image into ``target_dir/exif-tagged/``
  and writes the tags onto the copies. Originals are untouched. Use
  this when the source archive should stay clean.

Videos are skipped in both modes — exiftool can write QuickTime tags
in principle but the camera-trap community standard is to put labels
on the visualised best-frame JPEG rather than into the video
container. The visualised / blur outputs already handle videos via
their best-frame copies.

The actual tag construction and exiftool wrangling lives in
``_exif_writer``; this module is the orchestration shell.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.logging_config import get_logger
from app.models import Deployment, File, Project

from ._exif_writer import ExifBatch, build_tag_set, is_image_path
from ._label_filter import file_is_dropped_by_filter

logger = get_logger(__name__)


ExifMode = Literal["overwrite", "copy"]

EXIF_COPIES_SUBDIR = "exif-tagged"


@dataclass
class ExifMetadataResult:
    """Summary of an explicit EXIF predictions write."""

    mode: ExifMode = "copy"
    written_count: int = 0
    skipped_no_detections: int = 0
    skipped_video: int = 0
    skipped_missing_source: int = 0
    skipped_excluded: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "written_count": self.written_count,
            "skipped_no_detections": self.skipped_no_detections,
            "skipped_video": self.skipped_video,
            "skipped_missing_source": self.skipped_missing_source,
            "skipped_excluded": self.skipped_excluded,
            "errors": list(self.errors),
        }


def write_exif_predictions(
    db: Session,
    project_id: str,
    target_dir: Path,
    *,
    mode: ExifMode = "copy",
    excluded_label_ids: frozenset[str] | None = None,
) -> ExifMetadataResult:
    """Write detection EXIF tags onto image files for a project.

    `target_dir` is the run's root output directory. In `copy` mode
    the tagged copies land in `target_dir/exif-tagged/`; in `overwrite`
    mode `target_dir` is unused (we write to source paths) but kept
    in the signature for symmetry with the other postprocess modules.

    ``excluded_label_ids`` filters labelled detections out of the tag
    set and skips animal files where every passing label is
    excluded.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    threshold = project.detection_threshold

    files = db.execute(
        select(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    ).scalars().all()

    result = ExifMetadataResult(mode=mode)

    if mode == "copy":
        copies_dir = target_dir / EXIF_COPIES_SUBDIR
        copies_dir.mkdir(parents=True, exist_ok=True)
    else:
        copies_dir = None

    with ExifBatch() as exif_batch:
        for file in files:
            source = Path(file.file_path)
            if not source.exists():
                result.skipped_missing_source += 1
                result.errors.append(
                    f"Source file no longer on disk: {file.file_path}"
                )
                continue

            if not is_image_path(source):
                # Videos and other formats — see module docstring for
                # the reasoning. The visualised / blur outputs cover
                # the per-video best frame separately.
                result.skipped_video += 1
                continue

            if file_is_dropped_by_filter(
                db, file, threshold, excluded_label_ids
            ):
                result.skipped_excluded += 1
                continue

            tag_set = build_tag_set(
                db,
                file,
                project,
                APP_VERSION,
                excluded_label_ids=excluded_label_ids,
            )
            if tag_set is None:
                result.skipped_no_detections += 1
                continue

            if mode == "overwrite":
                target_path = source
            else:
                assert copies_dir is not None
                target_path = _unique_destination(copies_dir, source.name)
                try:
                    shutil.copy2(source, target_path)
                except OSError as e:
                    result.errors.append(
                        f"Could not copy {source} to {target_path}: {e}"
                    )
                    logger.exception(
                        f"exif_metadata: copy failed for {source}",
                    )
                    continue

            try:
                exif_batch.write(target_path, tag_set)
            except Exception as e:  # noqa: BLE001
                result.errors.append(
                    f"EXIF write failed for {target_path}: {e}"
                )
                logger.warning(
                    f"exif_metadata: EXIF write failed for "
                    f"{target_path}: {e}"
                )
                continue

            result.written_count += 1

    logger.info(
        f"exif_metadata: project={project_id} mode={mode} "
        f"written={result.written_count} "
        f"no_detections={result.skipped_no_detections} "
        f"video={result.skipped_video} "
        f"missing={result.skipped_missing_source} "
        f"excluded={result.skipped_excluded}"
    )
    return result


def _unique_destination(target_dir: Path, source_name: str) -> Path:
    """Same collision-suffix logic as the other modules. Local so we
    can return just a path (without the renamed flag we don't track
    in this module's result schema)."""
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix
    candidate = target_dir / source_name
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
