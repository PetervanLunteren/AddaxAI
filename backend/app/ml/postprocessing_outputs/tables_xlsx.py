"""Folder-run tabular XLSX output.

One workbook with two sheets, Detections and Counts, matching the
Research projects Export page's combined spreadsheet. Wraps the shared
``export_crud.build_spreadsheet_sheets`` + ``export_formats.serialize_xlsx_multi``
pipeline so the two modes never drift.

Writes ``<target_dir>/addaxai-spreadsheet.xlsx``.
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

XLSX_FILENAME = "addaxai-spreadsheet.xlsx"


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
    target_dir: Path,
) -> TablesXlsxResult:
    """Write the two-sheet ``addaxai-spreadsheet.xlsx`` (Detections + Counts)."""
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Complete record: folder-run data exports are never filtered by the
    # detection threshold (matches tables_csv and the recognition JSON).
    sheets = export_crud.build_spreadsheet_sheets(
        db, project, apply_threshold=False
    )
    payload = export_formats.serialize_xlsx_multi(sheets)

    output_path = target_dir / XLSX_FILENAME
    with open(output_path, "wb") as f:
        f.write(payload)

    total_rows = sum(len(rows) for _title, _headers, rows in sheets)
    logger.info(
        f"tables_xlsx: project={project_id} rows={total_rows} path={output_path}"
    )

    return TablesXlsxResult(output_path=str(output_path), row_count=total_rows)
