"""
Export endpoints for project data.

Streaming endpoints, one per export type:

- ``/deployments``: one row per deployment (location + effort: site,
  coordinates, date span, trap-nights) in CSV / TSV / XLSX.
- ``/files``: one row per media file, including empties (the membership
  table) in CSV / TSV / XLSX.
- ``/detections``: flat one-row-per-detection (the labels grain) in
  CSV / TSV / XLSX.
- ``/observations``: event-level one-row-per-species-per-event with the
  effective count (the Counts grain) in CSV / TSV / XLSX.
- ``/spreadsheet``: combined XLSX with Deployments, Files, Detections and
  Counts sheets for the one format that holds several tables in one file.
- ``/spatial``: GIS layers in GeoJSON / Shapefile (ZIP) / GeoPackage.
- ``/camtrap-dp``: Camera Trap Data Package v1.0 (GBIF-compatible) as a ZIP.

See ``app/api/crud/export.py`` for the data layer and
``app/api/crud/export_formats.py`` for the serializers.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.api.crud import export as export_crud
from app.api.crud import export_formats
from app.api.crud import job as job_crud
from app.api.schemas.job import JobCreate
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.models import Deployment, Project
from app.utils.datetime_serialization import set_active_project_timezone

router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["Export"])


def _resolve_project(project_id: str, db: Session) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.timezone:
        set_active_project_timezone(project.timezone)
    return project


def _filename_base(project: Project, kind: str) -> str:
    slug = export_formats.slugify(project.name)
    today = date.today().isoformat()
    return f"{kind}-{slug}-{today}"


def _attachment_headers(filename: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


def _tabular_response(
    headers: list[str],
    rows: list[list[object]],
    base: str,
    sheet_title: str,
    format: str,
) -> StreamingResponse:
    """Serialize (headers, rows) to CSV / TSV / XLSX as a download."""
    if format == "xlsx":
        payload = export_formats.serialize_xlsx(
            headers, rows, sheet_title=sheet_title
        )
        return StreamingResponse(
            BytesIO(payload),
            media_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            headers=_attachment_headers(f"{base}.xlsx"),
        )
    if format == "tsv":
        payload = export_formats.serialize_tsv(headers, rows)
        return StreamingResponse(
            BytesIO(payload),
            media_type="text/tab-separated-values",
            headers=_attachment_headers(f"{base}.tsv"),
        )
    payload = export_formats.serialize_csv(headers, rows)
    return StreamingResponse(
        BytesIO(payload),
        media_type="text/csv",
        headers=_attachment_headers(f"{base}.csv"),
    )


@router.get("/deployments")
async def export_deployments(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Location / effort table: one row per deployment (site + trap-nights)."""
    project = _resolve_project(project_id, db)
    headers, rows = export_crud.build_deployments_rows(db, project)
    base = _filename_base(project, "deployments")
    return _tabular_response(headers, rows, base, "Deployments", format)


@router.get("/files")
async def export_files(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Media / membership table: one row per file, including empties."""
    project = _resolve_project(project_id, db)
    headers, rows = export_crud.build_files_rows(db, project)
    base = _filename_base(project, "files")
    return _tabular_response(headers, rows, base, "Files", format)


@router.get("/detections")
async def export_detections(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Flat detections: one row per detection (the labels grain)."""
    project = _resolve_project(project_id, db)
    scoped = export_crud.get_scoped_detection_rows(db, project)
    headers, rows = export_crud.build_detection_rows(db, project, scoped)
    base = _filename_base(project, "detections")
    return _tabular_response(headers, rows, base, "Detections", format)


@router.get("/observations")
async def export_observations(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Event-level observations: one row per species per event with count."""
    project = _resolve_project(project_id, db)
    headers, rows = export_crud.build_observation_rows(db, project)
    base = _filename_base(project, "counts")
    return _tabular_response(headers, rows, base, "Counts", format)


@router.get("/spreadsheet")
async def export_spreadsheet(
    project_id: str,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Combined workbook: a Detections sheet and a Counts sheet.

    XLSX only — the one format that holds two tables in a single file. For
    CSV / TSV the client downloads the two single-table endpoints instead.
    """
    project = _resolve_project(project_id, db)
    sheets = export_crud.build_spreadsheet_sheets(db, project)
    payload = export_formats.serialize_xlsx_multi(sheets)
    base = _filename_base(project, "spreadsheet")
    return StreamingResponse(
        BytesIO(payload),
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers=_attachment_headers(f"{base}.xlsx"),
    )


@router.get("/spatial")
async def export_spatial(
    project_id: str,
    format: Literal["geojson", "shapefile", "gpkg"] = Query("geojson"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Spatial layers (deployments, observations, species summary)."""
    project = _resolve_project(project_id, db)
    scoped = export_crud.get_scoped_detection_rows(db, project)
    layers, skipped_deployment_ids = export_crud.build_spatial_layers(
        db, project, scoped
    )

    base = _filename_base(project, "spatial")
    if format == "shapefile":
        payload = export_formats.serialize_shapefile_zip(layers)
        media_type = "application/zip"
        filename = f"{base}.zip"
    elif format == "gpkg":
        payload = export_formats.serialize_geopackage(layers)
        media_type = "application/geopackage+sqlite3"
        filename = f"{base}.gpkg"
    else:
        payload = export_formats.serialize_geojson(layers)
        media_type = "application/geo+json"
        filename = f"{base}.geojson"

    response_headers = _attachment_headers(filename)
    if skipped_deployment_ids:
        response_headers["X-Skipped-Deployment-Ids"] = ",".join(
            skipped_deployment_ids
        )
        response_headers["Access-Control-Expose-Headers"] = (
            "X-Skipped-Deployment-Ids"
        )
    return StreamingResponse(
        BytesIO(payload),
        media_type=media_type,
        headers=response_headers,
    )


@router.post("/camtrap-dp/prepare", status_code=202)
async def prepare_camtrap_dp(
    project_id: str,
    include_thumbnails: bool = Query(
        False,
        description=(
            "When true, generate JPEG thumbnails for every media file and "
            "bundle them under `media/` in the ZIP with paths rewritten to "
            "relative names. Resulting package is self-contained."
        ),
    ),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Start a CamTrap DP export job. Returns `job_id` the client uses
    to track progress via WebSocket and then follow the `/download`
    endpoint to fetch the finished ZIP.
    """
    project = _resolve_project(project_id, db)

    has_deployment_with_site = (
        db.query(Deployment.id)
        .filter(Deployment.project_id == project.id)
        .filter(Deployment.site_id.isnot(None))
        .first()
        is not None
    )
    if not has_deployment_with_site:
        raise HTTPException(
            status_code=422,
            detail=(
                "Camtrap DP requires at least one deployment with a camera "
                "site (deployments without a site have no lat/lon and are "
                "excluded from this format)."
            ),
        )

    job = job_crud.create_job(
        db,
        JobCreate(
            type="camtrap_export",
            payload={
                "project_id": project.id,
                "include_thumbnails": include_thumbnails,
            },
        ),
    )

    from app.workers.camtrap_export_worker import process_camtrap_export_job

    ws_manager.register_start(
        job.id, lambda jid=job.id: process_camtrap_export_job(jid)
    )
    return {"job_id": job.id}


@router.get("/camtrap-dp/download")
async def download_camtrap_dp(
    project_id: str,
    job_id: str = Query(..., description="Job id returned by /camtrap-dp/prepare"),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Fetch the finished CamTrap DP ZIP for a completed export job."""
    project = _resolve_project(project_id, db)

    job = job_crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    payload = job.payload or {}
    if payload.get("project_id") != project.id:
        raise HTTPException(status_code=404, detail="Export job not found")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Export not ready (status={job.status})",
        )
    zip_path = payload.get("zip_path")
    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(status_code=410, detail="Export ZIP expired")

    base = _filename_base(project, "camtrap-dp")
    response_headers = _attachment_headers(f"{base}.zip")
    skipped = payload.get("skipped_deployment_ids") or []
    if skipped:
        response_headers["X-Skipped-Deployment-Ids"] = ",".join(skipped)
        response_headers["Access-Control-Expose-Headers"] = (
            "X-Skipped-Deployment-Ids"
        )
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{base}.zip",
        headers=response_headers,
    )
