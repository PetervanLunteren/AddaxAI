"""Flat observations CSV.

Same shape the Research projects Export page produces: one row per
species per image (or per event-level observation). Wraps the existing
``export_crud`` and ``export_formats`` modules so the canonical
observation row schema stays single-sourced.

The folder-run variant adds one extra column right after ``filename``:
``relative_path`` is the file's path under the user's chosen
``output_root``, populated from ``OutputContext.resolved_paths`` when
separation ran. With separation off, the column is blank — every file
sits at the root and ``filename`` already identifies it.

Output goes to ``<output_root>/observations.csv``, mirroring how the
recognition JSON writes its single file at the root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.api.crud import export as export_crud
from app.api.crud import export_formats
from app.core.logging_config import get_logger
from app.models import Project

from ._output_context import OutputContext

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
    ctx: OutputContext,
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

    ctx.output_root.mkdir(parents=True, exist_ok=True)

    scoped = export_crud.get_scoped_detection_rows(
        db, project, extra_excluded=excluded_species
    )
    headers, rows = export_crud.build_observation_rows(db, project, scoped)
    headers, rows = _enrich_with_relative_path(headers, rows, ctx)
    payload = export_formats.serialize_csv(headers, rows)

    output_path = ctx.output_root / CSV_FILENAME
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


def _enrich_with_relative_path(
    headers: list[str],
    rows: list[list[Any]],
    ctx: OutputContext,
) -> tuple[list[str], list[list[Any]]]:
    """Insert a ``relative_path`` column after ``filename``.

    Each row's ``image_uuid`` (column 0) is the ``File.id`` the row
    describes; we look up the first resolved placement on the context
    and emit a forward-slash path relative to ``ctx.output_root``.
    For multi-species files (multiple placements), the primary label
    folder is what shows up here — the user can still discover the
    extras by browsing the tree.
    """
    filename_idx = headers.index("filename")
    insert_idx = filename_idx + 1
    new_headers = list(headers)
    new_headers.insert(insert_idx, "relative_path")

    new_rows: list[list[Any]] = []
    for row in rows:
        file_id = row[0]
        resolved = ctx.resolved_paths.get(file_id)
        if resolved:
            try:
                rel = resolved[0].relative_to(ctx.output_root).as_posix()
            except ValueError:
                # A path outside output_root would only happen if the
                # caller constructed the context weirdly; fall back to
                # the absolute path so the CSV still points somewhere.
                rel = Path(resolved[0]).as_posix()
        else:
            rel = ""
        new_row = list(row)
        new_row.insert(insert_idx, rel)
        new_rows.append(new_row)

    return new_headers, new_rows
