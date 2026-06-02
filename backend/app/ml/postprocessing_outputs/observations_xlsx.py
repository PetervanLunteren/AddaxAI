"""Observations XLSX — Excel-native flat observation table.

Same row schema as ``observations_csv.py`` (including the canonical
``relative_path`` column). Wraps the existing
``export_crud.build_observation_rows`` +
``export_formats.serialize_xlsx`` pipeline.

Writes ``<output_root>/observations.xlsx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.api.crud import export as export_crud
from app.api.crud import export_formats
from app.core.logging_config import get_logger
from app.models import Project

from ._output_context import OutputContext

logger = get_logger(__name__)

XLSX_FILENAME = "observations.xlsx"


@dataclass
class ObservationsXlsxResult:
    """Summary of an XLSX write."""

    output_path: str = ""
    row_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "row_count": self.row_count,
            "errors": list(self.errors),
        }


def write_observations_xlsx(
    db: Session,
    project_id: str,
    ctx: OutputContext,
    *,
    excluded_species: list[str] | None = None,
) -> ObservationsXlsxResult:
    """Write the canonical observations XLSX for a project.

    ``excluded_species`` augments ``project.excluded_classes`` with a
    per-call exclusion. See ``observations_csv.write_observations_csv``
    for the rationale.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    ctx.output_root.mkdir(parents=True, exist_ok=True)

    scoped = export_crud.get_scoped_detection_rows(
        db, project, extra_excluded=excluded_species
    )
    headers, rows = export_crud.build_observation_rows(db, project, scoped)
    payload = export_formats.serialize_xlsx(
        headers, rows, sheet_title="Observations"
    )

    output_path = ctx.output_root / XLSX_FILENAME
    with open(output_path, "wb") as f:
        f.write(payload)

    logger.info(
        f"observations_xlsx: project={project_id} "
        f"rows={len(rows)} path={output_path}"
    )

    return ObservationsXlsxResult(
        output_path=str(output_path),
        row_count=len(rows),
    )
