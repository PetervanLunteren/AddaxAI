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
from app.ml.postprocessing_outputs._table_columns import (
    folder_run_columns,
    folder_run_table,
)
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
    # Fetched and released before the files table below loads its own row
    # set, so the two biggest allocations of the save job never coexist.
    # The summary is a handful of rows built from the same fetch, so it
    # rides along without changing that peak.
    scoped = export_crud.get_scoped_detection_rows(db, project)
    det_headers, det_rows = folder_run_table(
        *export_crud.build_detection_rows(db, project, scoped)
    )
    summary_headers, summary_rows = folder_run_table(
        *export_crud.build_summary_rows(db, project, scoped)
    )
    del scoped
    detections_path = target_dir / DETECTIONS_FILENAME
    with open(detections_path, "wb") as f:
        f.write(export_formats.serialize_csv(det_headers, det_rows))
    detection_count = len(det_rows)
    del det_rows

    summary_path = target_dir / SUMMARY_FILENAME
    with open(summary_path, "wb") as f:
        f.write(export_formats.serialize_csv(summary_headers, summary_rows))

    # One row per species per event, the count a person confirmed on the
    # Counts step or else the AI's MaxN. A handful of rows next to the two
    # big tables, so it does not move the memory peak described above.
    counts_headers, counts_rows = folder_run_table(
        *export_crud.build_observation_rows(db, project)
    )
    counts_path = target_dir / COUNTS_FILENAME
    with open(counts_path, "wb") as f:
        f.write(export_formats.serialize_csv(counts_headers, counts_rows))

    # Same scope as the detections table above, so the two agree: a file
    # whose species columns are empty has no rows in addaxai-detections.csv
    # either. Its own fetch, because the detections row set was released
    # above to keep peak memory down.
    files_headers, files_rows = folder_run_table(
        *export_crud.build_files_rows(db, project)
    )
    files_path = target_dir / FILES_FILENAME
    with open(files_path, "wb") as f:
        f.write(export_formats.serialize_csv(files_headers, files_rows))

    # Every box of the tracks in the detections table, one row per
    # sampled frame. Written row by row: on a marine run it is an order of
    # magnitude bigger than every other table together, so it is never
    # held as one list. A run without videos gets no frames file at all,
    # rather than an empty one.
    frames_paths: list[Path] = []
    frame_count = 0
    if export_crud.scope_has_videos(db, project):
        frames_path = target_dir / FRAMES_FILENAME
        keep = folder_run_columns(export_crud.FRAMES_HEADERS)
        rows = export_crud.iter_frames_rows(db, project)
        with open(frames_path, "w", encoding="utf-8", newline="") as f:
            frame_count = export_formats.write_delimited(
                f,
                [export_crud.FRAMES_HEADERS[i] for i in keep],
                ([row[i] for i in keep] for row in rows),
            )
        frames_paths.append(frames_path)

    logger.info(
        f"tables_csv: project={project_id} summary={len(summary_rows)} "
        f"counts={len(counts_rows)} files={len(files_rows)} "
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
            len(summary_rows)
            + len(counts_rows)
            + len(files_rows)
            + detection_count
            + frame_count
        ),
    )
