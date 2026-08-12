"""Folder-run tabular CSV outputs.

A folder run is "run AI without ecological interpretation", so its data
export is exactly two tables: a per-file files table (the complete
media list, including empties) and a per-detection detections table.
The interpretation tables (deployments with location / effort, and the
event-level counts) live in projects mode only — a folder run has no
sites, no real deployment, and no confirmed counts.

Both tables wrap the shared ``export_crud`` builders, so a column added
there shows up in projects-mode exports and here automatically. The
columns that say nothing in a folder run are then trimmed by
``_table_columns.folder_run_table``; see that module for which and why.

Writes ``addaxai-files.csv`` and ``addaxai-detections.csv`` under
``target_dir`` (the user's output dir, which defaults to the source
folder — the prefix keeps the run's files grouped between the user's
own).
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

FILES_FILENAME = "addaxai-files.csv"
DETECTIONS_FILENAME = "addaxai-detections.csv"


@dataclass
class TablesCsvResult:
    """Summary of the CSV writes (both tables)."""

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
    """Write ``addaxai-files.csv`` and ``addaxai-detections.csv``.

    The data exports are the complete record of the run (no per-call
    species exclusion), so both tables derive from the same project
    scope and stay join-consistent.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Complete record: folder-run data exports are never filtered by the
    # detection threshold. Thresholding is an in-app / media-output
    # concern only (beta feedback from Dan). Built first and released
    # before the files table below loads its own row set, so the two
    # biggest allocations of the save job never coexist.
    scoped = export_crud.get_scoped_detection_rows(
        db, project, apply_threshold=False
    )
    det_headers, det_rows = folder_run_table(
        *export_crud.build_detection_rows(db, project, scoped)
    )
    del scoped
    detections_path = target_dir / DETECTIONS_FILENAME
    with open(detections_path, "wb") as f:
        f.write(export_formats.serialize_csv(det_headers, det_rows))
    detection_count = len(det_rows)
    del det_rows

    # The files table is deliberately NOT unthresholded the way the
    # detections table above is. Its observation_type and species columns
    # both describe the file's strongest *passing* detection, so dropping
    # the threshold here would make a file read as an animal that the app
    # itself calls blank. The visible effect is that a file whose every box
    # sits below the threshold has empty species columns while
    # addaxai-detections.csv still lists those boxes. That is the two grains
    # answering two different questions, not a bug to fix.
    files_headers, files_rows = folder_run_table(
        *export_crud.build_files_rows(db, project)
    )
    files_path = target_dir / FILES_FILENAME
    with open(files_path, "wb") as f:
        f.write(export_formats.serialize_csv(files_headers, files_rows))

    logger.info(
        f"tables_csv: project={project_id} "
        f"files={len(files_rows)} detections={detection_count}"
    )

    return TablesCsvResult(
        output_paths=[
            str(files_path),
            str(detections_path),
        ],
        row_count=len(files_rows) + detection_count,
    )
