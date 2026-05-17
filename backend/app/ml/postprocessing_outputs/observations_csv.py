"""Flat observations CSV.

Same shape the Research projects Export page produces: one row per
species per image (or per event-level observation). The user can open
it directly in Excel, R, Pandas, QGIS, etc. Wraps the existing
`export_crud` and `export_formats` modules — there is one canonical
observation-row schema in the codebase and we use it here.

Output goes to `<target_dir>/observations.csv` at the root, not in a
subfolder, mirroring how `recognition_json` writes its single file
directly into the output root.
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

CSV_FILENAME = "observations.csv"


@dataclass
class ObservationsCsvResult:
    """Summary of a CSV write."""

    output_path: str = ""
    row_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "row_count": self.row_count,
            "errors": list(self.errors),
        }


def write_observations_csv(
    db: Session,
    project_id: str,
    target_dir: Path,
    *,
    excluded_species: list[str] | None = None,
) -> ObservationsCsvResult:
    """Write the canonical observations CSV for a project.

    ``excluded_species`` augments ``project.excluded_classes`` with a
    per-call exclusion. Used by the folder-run Save step to honour
    the user's "exclude these species from outputs" filter without
    persisting it on the project.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir.mkdir(parents=True, exist_ok=True)

    scoped = export_crud.get_scoped_detection_rows(
        db, project, extra_excluded=excluded_species
    )
    headers, rows = export_crud.build_observation_rows(db, project, scoped)
    payload = export_formats.serialize_csv(headers, rows)

    output_path = target_dir / CSV_FILENAME
    with open(output_path, "wb") as f:
        f.write(payload)

    logger.info(
        f"observations_csv: project={project_id} "
        f"rows={len(rows)} path={output_path}"
    )

    return ObservationsCsvResult(
        output_path=str(output_path),
        row_count=len(rows),
    )
