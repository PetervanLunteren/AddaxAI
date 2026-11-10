"""Folder-run tabular CSV outputs.

A folder run writes five tables: a per-species summary (what was found
and how much of it), a per-event counts table (one row per species per
event, with the count the user confirmed on the Counts step or else the
AI's MaxN), a per-file files table (the complete media list, including
empties), a detections table (one row per card: a photo box or a video
track) and a frames table (every box of those tracks, one row per
sampled frame, videos only). The deployments table
(location and effort) lives in projects mode only: a folder run has no
sites and no real deployment.

All five wrap the shared ``export_crud`` builders, so a column added
there shows up in projects-mode exports and here automatically. The
columns that say nothing in a folder run are then trimmed by
``_table_columns.folder_run_table``; see that module for which and why.

Writes ``addaxai-summary.csv``, ``addaxai-counts.csv``,
``addaxai-files.csv``, ``addaxai-detections.csv`` and, when the run
holds a video, ``addaxai-frames.csv`` under ``target_dir``
(the user's output dir, which defaults to the source folder; the prefix
keeps the run's files grouped between the user's own).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.crud import export as export_crud
from app.api.crud import export_formats
from app.core.logging_config import get_logger
from app.ml.postprocessing_outputs._table_columns import folder_run_table
from app.models import Project

logger = get_logger(__name__)

SUMMARY_FILENAME = "addaxai-summary.csv"
COUNTS_FILENAME = "addaxai-counts.csv"
FILES_FILENAME = "addaxai-files.csv"
DETECTIONS_FILENAME = "addaxai-detections.csv"
FRAMES_FILENAME = "addaxai-frames.csv"


@dataclass
class TablesCsvResult:
    """Summary of the CSV writes (all five tables)."""

    output_paths: list[str] = field(default_factory=list)
    row_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_paths": list(self.output_paths),
            "row_count": self.row_count,
            "errors": list(self.errors),
        }


def write_tables_csv(
    db: Session,
    project_id: str,
    target_dir: Path,
) -> TablesCsvResult:
    """Write ``addaxai-summary.csv``, ``addaxai-counts.csv``,
    ``addaxai-files.csv``, ``addaxai-detections.csv`` and, when the run
    holds a video, ``addaxai-frames.csv``.

    All tables derive from the same project scope (no per-call species
    exclusion), so they stay join-consistent.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir.mkdir(parents=True, exist_ok=True)

    # One scope for every table and for both modes: the threshold plus
    # the verified override, and only boxes on a video's visible frame.
    # So these tables hold what the Labels step showed, which is what the
    # user could actually check and correct. ``addaxai-recognitions.json``
    # stays the complete record of the run.
    #
    # Every table is streamed from its own query straight into its file,
    # so the save job holds one batch of rows at a time, whatever the size
    # of the run.
    detections_path = target_dir / DETECTIONS_FILENAME
    detection_count = export_formats.write_csv(
        detections_path,
        *folder_run_table(*export_crud.build_detection_rows(db, project)),
    )
    summary_path = target_dir / SUMMARY_FILENAME
    summary_count = export_formats.write_csv(
        summary_path,
        *folder_run_table(*export_crud.build_summary_rows(db, project)),
    )
    # One row per species per event, the count a person confirmed on the
    # Counts step or else the AI's MaxN.
    counts_path = target_dir / COUNTS_FILENAME
    counts_count = export_formats.write_csv(
        counts_path,
        *folder_run_table(*export_crud.build_observation_rows(db, project)),
    )
    # Same scope as the detections table above, so the two agree: a file
    # whose species columns are empty has no rows in addaxai-detections.csv
    # either.
    files_path = target_dir / FILES_FILENAME
    files_count = export_formats.write_csv(
        files_path,
        *folder_run_table(*export_crud.build_files_rows(db, project)),
    )

    # Every box of the tracks in the detections table, one row per
    # sampled frame. On a marine run it is an order of magnitude bigger
    # than every other table together. A run without videos gets no
    # frames file at all, rather than an empty one.
    frames_paths: list[Path] = []
    frame_count = 0
    if export_crud.scope_has_videos(db, project):
        frames_path = target_dir / FRAMES_FILENAME
        frame_count = export_formats.write_csv(
            frames_path,
            *folder_run_table(
                export_crud.FRAMES_HEADERS, export_crud.iter_frames_rows(db, project)
            ),
        )
        frames_paths.append(frames_path)

    logger.info(
        f"tables_csv: project={project_id} summary={summary_count} "
        f"counts={counts_count} files={files_count} "
        f"detections={detection_count} frames={frame_count}"
    )

    return TablesCsvResult(
        output_paths=[
            str(summary_path),
            str(counts_path),
            str(files_path),
            str(detections_path),
            *(str(p) for p in frames_paths),
        ],
        row_count=(
            summary_count + counts_count + files_count + detection_count + frame_count
        ),
    )
