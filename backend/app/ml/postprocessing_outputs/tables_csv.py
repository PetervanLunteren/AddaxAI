"""Folder-run tabular CSV outputs.

A folder run is "run AI without ecological interpretation", so its data
export is exactly three tables: a per-species summary (what was found
and how much of it), a per-file files table (the complete media list,
including empties) and a per-detection detections table. The
interpretation tables (deployments with location / effort, and the
event-level counts) live in projects mode only — a folder run has no
sites, no real deployment, and no confirmed counts.

All three wrap the shared ``export_crud`` builders, so a column added
there shows up in projects-mode exports and here automatically. The
columns that say nothing in a folder run are then trimmed by
``_table_columns.folder_run_table``; see that module for which and why.

Writes ``addaxai-summary.csv``, ``addaxai-files.csv`` and
``addaxai-detections.csv`` under ``target_dir`` (the user's output dir,
which defaults to the source folder — the prefix keeps the run's files
grouped between the user's own).
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
FILES_FILENAME = "addaxai-files.csv"
DETECTIONS_FILENAME = "addaxai-detections.csv"


@dataclass
class TablesCsvResult:
    """Summary of the CSV writes (all three tables)."""

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
    """Write ``addaxai-summary.csv``, ``addaxai-files.csv`` and
    ``addaxai-detections.csv``.

    The data exports are the complete record of the run (no per-call
    species exclusion), so all tables derive from the same project
    scope and stay join-consistent.
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

    logger.info(
        f"tables_csv: project={project_id} summary={len(summary_rows)} "
        f"files={len(files_rows)} detections={detection_count}"
    )

    return TablesCsvResult(
        output_paths=[
            str(summary_path),
            str(files_path),
            str(detections_path),
        ],
        row_count=len(summary_rows) + len(files_rows) + detection_count,
    )
