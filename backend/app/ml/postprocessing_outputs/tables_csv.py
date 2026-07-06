"""Folder-run tabular CSV outputs.

Writes the same tables the Research projects Export page produces, so the
two modes stay in sync: a per-deployment deployments table (location +
effort), a per-file files table (membership, including empties), a
per-detection detections table, and an event-level counts table. All
wrap the shared ``export_crud`` builders, so a column added there shows up
in both places automatically.

Writes the four ``addaxai-*.csv`` files under ``target_dir`` (the
user's output dir, which defaults to the source folder — the prefix
keeps the run's files grouped between the user's own).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.crud import export as export_crud
from app.api.crud import export_formats
from app.core.logging_config import get_logger
from app.models import Project

logger = get_logger(__name__)

DEPLOYMENTS_FILENAME = "addaxai-deployments.csv"
FILES_FILENAME = "addaxai-files.csv"
DETECTIONS_FILENAME = "addaxai-detections.csv"
COUNTS_FILENAME = "addaxai-counts.csv"


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
    """Write the four ``addaxai-*.csv`` tables into ``target_dir``.

    The data exports are the complete record of the run (no per-call
    species exclusion), so all tables derive from the same project
    scope and stay join-consistent.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir.mkdir(parents=True, exist_ok=True)

    scoped = export_crud.get_scoped_detection_rows(db, project)
    dep_headers, dep_rows = export_crud.build_deployments_rows(db, project)
    files_headers, files_rows = export_crud.build_files_rows(db, project)
    det_headers, det_rows = export_crud.build_detection_rows(db, project, scoped)
    obs_headers, obs_rows = export_crud.build_observation_rows(db, project)

    deployments_path = target_dir / DEPLOYMENTS_FILENAME
    with open(deployments_path, "wb") as f:
        f.write(export_formats.serialize_csv(dep_headers, dep_rows))
    files_path = target_dir / FILES_FILENAME
    with open(files_path, "wb") as f:
        f.write(export_formats.serialize_csv(files_headers, files_rows))
    detections_path = target_dir / DETECTIONS_FILENAME
    with open(detections_path, "wb") as f:
        f.write(export_formats.serialize_csv(det_headers, det_rows))
    counts_path = target_dir / COUNTS_FILENAME
    with open(counts_path, "wb") as f:
        f.write(export_formats.serialize_csv(obs_headers, obs_rows))

    logger.info(
        f"tables_csv: project={project_id} deployments={len(dep_rows)} "
        f"files={len(files_rows)} detections={len(det_rows)} "
        f"counts={len(obs_rows)}"
    )

    return TablesCsvResult(
        output_paths=[
            str(deployments_path),
            str(files_path),
            str(detections_path),
            str(counts_path),
        ],
        row_count=(
            len(dep_rows) + len(files_rows) + len(det_rows) + len(obs_rows)
        ),
    )
