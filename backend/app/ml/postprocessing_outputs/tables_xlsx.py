"""Folder-run tabular XLSX output.

One workbook with two sheets, Detections and Counts, matching the
Research projects Export page's combined spreadsheet. Wraps the shared
``export_crud.build_spreadsheet_sheets`` + ``export_formats.serialize_xlsx_multi``
pipeline so the two modes never drift.

Writes ``<output_root>/spreadsheet.xlsx``.
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

XLSX_FILENAME = "spreadsheet.xlsx"


@dataclass
class TablesXlsxResult:
    """Summary of the XLSX write."""

    output_path: str = ""
    row_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "row_count": self.row_count,
            "errors": list(self.errors),
        }


def write_tables_xlsx(
    db: Session,
    project_id: str,
    ctx: OutputContext,
) -> TablesXlsxResult:
    """Write the two-sheet ``spreadsheet.xlsx`` (Detections + Counts)."""
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    ctx.output_root.mkdir(parents=True, exist_ok=True)

    sheets = export_crud.build_spreadsheet_sheets(db, project)
    payload = export_formats.serialize_xlsx_multi(sheets)

    output_path = ctx.output_root / XLSX_FILENAME
    with open(output_path, "wb") as f:
        f.write(payload)

    total_rows = sum(len(rows) for _title, _headers, rows in sheets)
    logger.info(
        f"tables_xlsx: project={project_id} rows={total_rows} path={output_path}"
    )

    return TablesXlsxResult(output_path=str(output_path), row_count=total_rows)
