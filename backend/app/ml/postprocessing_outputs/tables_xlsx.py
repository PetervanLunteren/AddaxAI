"""Folder-run tabular XLSX output.

One workbook with two sheets, Files and Detections — the same two
tables as the folder-run CSV export, in one file. A folder run is "run
AI without ecological interpretation", so the projects-mode sheets
(Deployments, Counts) are intentionally absent. The sheets wrap the
shared ``export_crud`` builders and are trimmed to the same column set
as the folder-run CSVs; see ``_table_columns`` for which columns and
why.

Writes ``<target_dir>/addaxai-spreadsheet.xlsx``.
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
    """Write the two-sheet ``addaxai-spreadsheet.xlsx`` (Files + Detections)."""
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir.mkdir(parents=True, exist_ok=True)

    # One scope for both sheets and for both modes: the threshold plus the
    # verified override, and only boxes on a video's visible frame, so the
    # workbook holds what the Labels step showed (matches tables_csv).
    # ``addaxai-recognitions.json`` stays the complete record of the run.
    #
    # Built before the files sheet and released right after, so this row
    # set and build_files_rows' own never sit in memory together: on a
    # large run each is the biggest allocation in the whole save job.
    scoped = export_crud.get_scoped_detection_rows(db, project)
    det_headers, det_rows = folder_run_table(
        *export_crud.build_detection_rows(db, project, scoped)
    )
    del scoped
    files_headers, files_rows = folder_run_table(
        *export_crud.build_files_rows(db, project)
    )
    sheets = [
        ("Files", files_headers, files_rows),
        ("Detections", det_headers, det_rows),
    ]

    total_rows = sum(len(rows) for _title, _headers, rows in sheets)

    output_path = target_dir / XLSX_FILENAME
    # Saved straight to disk; the bytes-returning serializer would hold
    # the whole zipped workbook in memory first.
    #
    # A run too big for the XLSX format reports the reason and writes
    # nothing, rather than raising. Raising would fail the whole save
    # job, and by the time this module runs the expensive work
    # (separating folders, annotated copies) is already finished and on
    # disk, so the user would be told everything failed when only this
    # one output could not exist. The error travels back on the result
    # and the completion screen lists it (`collectIssues` in
    # `SaveShared.tsx`).
    #
    # Note the Save step's Format is one dropdown, CSV *or* Excel, never
    # both, so this run then writes no tables at all. That is why the
    # message has to name CSV as the way out: choosing it and saving
    # again is the user's only route to the data.
    try:
        export_formats.write_xlsx_multi(sheets, output_path)
    except export_formats.XlsxRowLimitError as e:
        logger.warning(
            f"tables_xlsx: project={project_id} rows={total_rows} skipped: {e}"
        )
        return TablesXlsxResult(row_count=total_rows, errors=[str(e)])

    logger.info(
        f"tables_xlsx: project={project_id} rows={total_rows} path={output_path}"
    )

    return TablesXlsxResult(output_path=str(output_path), row_count=total_rows)
