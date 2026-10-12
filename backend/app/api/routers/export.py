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
- ``/summary``: one row per species / category with counts over the other
  tables (the overview) in CSV / TSV / XLSX.
- ``/spreadsheet``: combined XLSX with Summary, Counts, Detections, Files
  and Deployments sheets for the one format that holds several tables in
  one file.
- ``/spatial``: GIS layers in GeoJSON / Shapefile (ZIP) / GeoPackage.
- ``/camtrap-dp``: Camera Trap Data Package v1.0 (GBIF-compatible) as a ZIP.
- ``/recognition-json/prepare``: not a download. A job that writes the
  Timelapse recognition file into each deployment folder in scope and
  reports per deployment on completion.
- ``/media/preview`` and ``/media/prepare``: the image copies export.
  The same media modules and worker as a folder run's Save step, scoped
  to deployments and file filters, writing into a folder the user picks.

See ``app/api/crud/export.py`` for the data layer and
``app/api/crud/export_formats.py`` for the serializers.
"""

from __future__ import annotations

import asyncio
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
from app.api.schemas.export import (
    MediaExportOptions,
    MediaExportRequest,
    OutputPreviewResponse,
)
from app.api.schemas.job import JobCreate
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.postprocessing_outputs._scope import MediaScope
from app.ml.postprocessing_outputs.output_preview import build_output_preview
from app.models import Deployment, Project
from app.utils.datetime_serialization import set_active_project_timezone

router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["Export"])


def _parse_ids(value: str | None) -> list[str] | None:
    """Split a comma-separated query param into a list, or None if absent.

    None means "no scope" (whole project); a list narrows the export.
    """
    if not value:
        return None
    ids = [v for v in (s.strip() for s in value.split(",")) if v]
    return ids or None


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
        try:
            payload = export_formats.serialize_xlsx(
                headers, rows, sheet_title=sheet_title
            )
        except export_formats.XlsxRowLimitError as e:
            # The message names the table, its size and the CSV way out,
            # and `fetchBlob` in the frontend surfaces a string `detail`
            # verbatim in the Export page's error box.
            raise HTTPException(status_code=422, detail=str(e)) from e
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
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    deployment_ids: str | None = Query(
        None, description="Comma-separated deployment IDs"
    ),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Location / effort table: one row per deployment (site + trap-nights)."""
    project = _resolve_project(project_id, db)
    scope = export_crud.resolve_scope_deployment_ids(
        db, project, _parse_ids(site_ids), _parse_ids(deployment_ids)
    )
    headers, rows = export_crud.build_deployments_rows(db, project, scope)
    base = _filename_base(project, "deployments")
    return _tabular_response(headers, rows, base, "Deployments", format)


@router.get("/files")
async def export_files(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    deployment_ids: str | None = Query(
        None, description="Comma-separated deployment IDs"
    ),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Media / membership table: one row per file, including empties."""
    project = _resolve_project(project_id, db)
    scope = export_crud.resolve_scope_deployment_ids(
        db, project, _parse_ids(site_ids), _parse_ids(deployment_ids)
    )
    headers, rows = export_crud.build_files_rows(db, project, scope)
    base = _filename_base(project, "files")
    return _tabular_response(headers, rows, base, "Files", format)


@router.get("/detections")
async def export_detections(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    deployment_ids: str | None = Query(
        None, description="Comma-separated deployment IDs"
    ),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Flat detections: one row per detection (the labels grain)."""
    project = _resolve_project(project_id, db)
    scope = export_crud.resolve_scope_deployment_ids(
        db, project, _parse_ids(site_ids), _parse_ids(deployment_ids)
    )
    scoped = export_crud.get_scoped_detection_rows(
        db, project, deployment_ids=scope
    )
    headers, rows = export_crud.build_detection_rows(db, project, scoped)
    base = _filename_base(project, "detections")
    return _tabular_response(headers, rows, base, "Detections", format)


@router.get("/observations")
async def export_observations(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    deployment_ids: str | None = Query(
        None, description="Comma-separated deployment IDs"
    ),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Event-level observations: one row per species per event with count."""
    project = _resolve_project(project_id, db)
    scope = export_crud.resolve_scope_deployment_ids(
        db, project, _parse_ids(site_ids), _parse_ids(deployment_ids)
    )
    headers, rows = export_crud.build_observation_rows(db, project, scope)
    base = _filename_base(project, "counts")
    return _tabular_response(headers, rows, base, "Counts", format)


@router.get("/summary")
async def export_summary(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    deployment_ids: str | None = Query(
        None, description="Comma-separated deployment IDs"
    ),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Summary: one row per species / category with counts over the other tables."""
    project = _resolve_project(project_id, db)
    scope = export_crud.resolve_scope_deployment_ids(
        db, project, _parse_ids(site_ids), _parse_ids(deployment_ids)
    )
    scoped = export_crud.get_scoped_detection_rows(
        db, project, deployment_ids=scope
    )
    headers, rows = export_crud.build_summary_rows(db, project, scoped, scope)
    base = _filename_base(project, "summary")
    return _tabular_response(headers, rows, base, "Summary", format)


@router.get("/spreadsheet")
async def export_spreadsheet(
    project_id: str,
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    deployment_ids: str | None = Query(
        None, description="Comma-separated deployment IDs"
    ),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Combined workbook: Summary, Counts, Detections, Files and Deployments
    sheets.

    XLSX only, the one format that holds several tables in a single file.
    For CSV / TSV the client downloads the five single-table endpoints
    instead, in the same order.
    """
    project = _resolve_project(project_id, db)
    scope = export_crud.resolve_scope_deployment_ids(
        db, project, _parse_ids(site_ids), _parse_ids(deployment_ids)
    )
    sheets = export_crud.build_spreadsheet_sheets(db, project, scope)
    try:
        payload = export_formats.serialize_xlsx_multi(sheets)
    except export_formats.XlsxRowLimitError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
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
    """Spatial layers (deployments, species summary)."""
    project = _resolve_project(project_id, db)
    scoped = export_crud.get_scoped_detection_rows(db, project)
    # Deployments with no site are dropped from the layers (GeoJSON needs
    # coordinates). Their ids are not returned to the client: the Export
    # page already warns before the download, via SpatialExportConfirmDialog
    # driven by /deployments-without-site, so a response header saying the
    # same thing had no reader and was dead weight on the API.
    layers, _skipped_deployment_ids = export_crud.build_spatial_layers(
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

    return StreamingResponse(
        BytesIO(payload),
        media_type=media_type,
        headers=_attachment_headers(filename),
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
    # Deployments with no site are absent from the package (Camtrap DP
    # requires lat/lon). The user is told before the download, by
    # CamtrapDPExportConfirmDialog on the Export page, so this response
    # carried a header naming them that nothing ever read.
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{base}.zip",
        headers=_attachment_headers(f"{base}.zip"),
    )


@router.post("/recognition-json/prepare", status_code=202)
async def prepare_recognition_json(
    project_id: str,
    site_ids: str | None = Query(None),
    deployment_ids: str | None = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Start writing ``addaxai-recognitions.json`` into each deployment
    folder in scope, paths relative to that folder: the same file and the
    same writer as a folder run's save step.

    Into the folders rather than a download because that is where
    Timelapse looks for it, and a project's deployments sit on several
    drives, some of them unplugged: a ZIP the user unpacks into the right
    folders by hand is the error-prone version of this. A job, because a
    project can hold hundreds of deployments; the client follows it over
    WebSocket and gets the written/skipped list on completion. An
    existing file is replaced, as a folder-run save replaces it.
    """
    project = _resolve_project(project_id, db)
    scope = export_crud.resolve_scope_deployment_ids(
        db, project, _parse_ids(site_ids), _parse_ids(deployment_ids)
    )
    job = job_crud.create_job(
        db,
        JobCreate(
            type="recognition_json_export",
            payload={"project_id": project.id, "deployment_ids": scope},
        ),
    )

    from app.workers.recognition_json_export_worker import (
        process_recognition_json_export_job,
    )

    ws_manager.register_start(
        job.id, lambda jid=job.id: process_recognition_json_export_job(jid)
    )
    return {"job_id": job.id}


def _media_scope(
    db: Session, project: Project, body: MediaExportOptions
) -> MediaScope:
    """The one scope object both media endpoints hand to the modules.

    An explicit empty id list is refused: `resolve_scope_deployment_ids`
    reads it as "no scope", so it would copy the whole project, and an
    empty export would still replace a previous one in that folder. The
    Export page disables the button in that state; this is the server's
    own guard. Ids are resolved through the same helper the tables use,
    so ids from another project are dropped.
    """
    if body.deployment_ids is not None and len(body.deployment_ids) == 0:
        raise HTTPException(status_code=422, detail="No deployments selected")
    resolved = export_crud.resolve_scope_deployment_ids(
        db, project, None, body.deployment_ids
    )
    scope = MediaScope.from_payload(body.model_dump(mode="json"))
    if resolved is None:
        return scope
    return MediaScope(
        deployment_ids=tuple(resolved),
        favorited_only=scope.favorited_only,
        verified_only=scope.verified_only,
        date_from=scope.date_from,
        date_to=scope.date_to,
    )


@router.post("/media/preview", response_model=OutputPreviewResponse)
def preview_media_export(
    project_id: str,
    body: MediaExportOptions,
    db: Session = Depends(get_db),
) -> OutputPreviewResponse:
    """What the image copies export would write: file counts, bytes and
    the folder tree, for the dialog's summary line.

    Sync on purpose, like the folder-run preview: it loads every detection
    in scope and loops in Python, which must not sit on the event loop.
    Nothing here returns an observational datetime.
    """
    project = _resolve_project(project_id, db)
    scope = _media_scope(db, project, body)
    preview = build_output_preview(
        db,
        project.id,
        media_threshold=body.media_threshold,
        excluded_label_ids=(
            frozenset(body.excluded_label_ids)
            if body.excluded_label_ids
            else None
        ),
        include_empty=body.include_empty,
        name_mode=body.name_mode,
        group_events=body.group_events,
        group_by=body.separate_group_by,
        species_last=body.separate_species_last,
        # With blur on a video is written as its blurred still.
        videos_as_stills=body.anonymise,
        scope=scope,
        deployment_folder_in_path=True,
    )
    return OutputPreviewResponse(**preview.to_dict())


@router.post("/media/prepare", status_code=202)
async def prepare_media_export(
    project_id: str,
    body: MediaExportRequest,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Start the image copies export: the folder-run save-outputs job
    with only the media modules, a scope, and the deployment folder name
    kept in the layout (the worker derives that from `project.mode`).

    Copies go into `output_dir/addaxai-media`. The worker owns the marker
    there and replaces a previous export's subfolder; this endpoint never
    stamps it (the marker is the wipe's proof of ownership).
    """
    project = _resolve_project(project_id, db)
    scope = _media_scope(db, project, body)

    output_root = Path(body.output_dir)
    try:
        # Off the loop: the user picked this folder, and a network drive
        # that is asleep answers a mkdir in seconds, not microseconds.
        await asyncio.to_thread(output_root.mkdir, parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot create the output folder: {e}",
        ) from e

    job = job_crud.create_job(
        db,
        JobCreate(
            type="media_export",
            payload={
                **body.model_dump(mode="json"),
                "project_id": project.id,
                "deployment_ids": (
                    list(scope.deployment_ids)
                    if scope.deployment_ids is not None
                    else None
                ),
                "output_dir": str(output_root),
                # The worker runs whatever is ticked. Copies always; the
                # run readme defaults to on when the key is missing
                # (`payload.get("run_readme", True)` there), and it would
                # write an addaxai-run-info.txt into the user's folder.
                "separate_folders": True,
                "run_readme": False,
            },
        ),
    )

    from app.workers.folder_run_save_outputs_worker import (
        process_save_outputs_job,
    )

    ws_manager.register_start(
        job.id, lambda jid=job.id: process_save_outputs_job(jid)
    )
    return {"job_id": job.id}
