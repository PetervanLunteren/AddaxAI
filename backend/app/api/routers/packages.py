"""Importing a project package.

- ``POST /api/packages/preview``: what importing a package folder would
  do, without writing anything. The dialog shows it and the Import
  button follows its ``problems`` list.
- ``POST /api/packages/import``: the import itself, as a job.

The export side sits with the other exports under
``/api/projects/{id}/export/package``. The format and the rules are in
``app/services/project_package.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.crud import job as job_crud
from app.api.schemas.job import JobCreate
from app.api.schemas.package import (
    PackageImportRequest,
    PackagePathRequest,
    PackagePreview,
)
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.services.project_package import PackageError, inspect_package

router = APIRouter(prefix="/api/packages", tags=["Packages"])


def _preview(db: Session, path: str) -> PackagePreview:
    try:
        return inspect_package(db, Path(path))
    except PackageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/preview", response_model=PackagePreview)
def preview_package(
    body: PackagePathRequest, db: Session = Depends(get_db)
) -> PackagePreview:
    """Read the package and say what an import would create, add, skip
    or refuse. Sync on purpose: it opens files on whatever drive the
    user picked."""
    return _preview(db, body.path)


@router.post("/import", status_code=202)
async def start_package_import(
    body: PackageImportRequest, db: Session = Depends(get_db)
) -> dict[str, str]:
    """Start the import job. Refused while another job runs: the import
    holds the database's one write lock for its whole duration.

    Async because `register_start` schedules on the running loop; the
    package read goes to a thread, as the other prepare endpoints do
    with their filesystem checks.
    """
    preview = await asyncio.to_thread(_preview, db, body.path)
    if preview.problems:
        raise HTTPException(status_code=400, detail=" ".join(preview.problems))
    if job_crud.has_active_jobs(db):
        raise HTTPException(
            status_code=409,
            detail="Another job is running. Wait for it to finish, then import.",
        )
    job = job_crud.create_job(
        db,
        JobCreate(
            type="package_import",
            payload={
                "path": str(Path(body.path)),
                "replace_existing": body.replace_existing,
                "project_id": preview.project_id,
            },
        ),
    )

    from app.workers.project_package_worker import process_package_import_job

    ws_manager.register_start(
        job.id, lambda jid=job.id: process_package_import_job(jid)
    )
    return {"job_id": job.id}
