"""
Export endpoints for project data. Every export writes into a folder and
answers with what it wrote; nothing is a browser download.

- ``/tables``: Summary, Counts, Detections, Files and Deployments, as one
  XLSX workbook or five CSV / TSV files.
- ``/spatial``: GIS layers as GeoPackage, GeoJSON or a Shapefile ZIP.
- ``/camtrap-dp/prepare``: a job writing a Camera Trap Data Package ZIP.
- ``/recognition-json/prepare``: a job writing the Timelapse recognition
  file into each deployment folder in scope.
- ``/media/preview`` and ``/media/prepare``: the image copies export, the
  folder-run Save step's media modules, scoped and into a picked folder.
- ``/package/size`` and ``/package/prepare``: the project package, a
  folder another AddaxAI imports (``app/services/project_package.py``).

Every export takes the same ``ExportTarget``: a folder that must exist,
and the deployments to cover. A file never overwrites one already there
(``export_formats.unique_export_path``); names are
``<kind>-<project>-<date>.<ext>`` (``export_formats.export_file_name``).
See ``app/api/crud/export.py`` for the data layer and
``app/api/crud/export_formats.py`` for the serializers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api.crud import export as export_crud
from app.api.crud import export_formats
from app.api.crud import job as job_crud
from app.api.schemas.export import (
    CamtrapExportRequest,
    MediaExportOptions,
    MediaExportRequest,
    OutputPreviewResponse,
    SpatialExportRequest,
    TablesExportRequest,
    WrittenFiles,
)
from app.api.schemas.job import JobCreate
from app.api.schemas.package import (
    PackageExportRequest,
    PackageSizeResponse,
)
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.postprocessing_outputs._scope import MediaScope
from app.ml.postprocessing_outputs.output_preview import build_output_preview
from app.ml.postprocessing_outputs.separate_folders import keeps_deployment_folder
from app.models import Deployment, File, Project
from app.services.project_package import PackageError, package_destination
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


def _scope(
    db: Session, project: Project, deployment_ids: list[str] | None
) -> list[str] | None:
    """The deployments an export covers: the picked ones, or None for the
    whole project.

    An explicit empty list is refused: `resolve_scope_deployment_ids`
    reads it as "no scope", so it would export the whole project. The
    dialogs disable Start in that state; this is the server's own guard.
    Ids from another project are dropped by the same helper.
    """
    if deployment_ids is not None and len(deployment_ids) == 0:
        raise HTTPException(status_code=422, detail="No deployments selected")
    return export_crud.resolve_scope_deployment_ids(db, project, None, deployment_ids)


async def _output_dir(path: str) -> Path:
    """The picked folder, which must exist. Checked off the loop: a
    network drive that is asleep answers a stat in seconds."""
    folder = Path(path)
    if not await asyncio.to_thread(folder.is_dir):
        raise HTTPException(
            status_code=400, detail="Choose a folder to save the export in"
        )
    return folder



def _require_a_site(
    db: Session, project: Project, scope: list[str] | None, export: str
) -> None:
    """422 when no deployment in scope has a site. Both location exports
    leave deployments without coordinates out, so with none left they
    would write an empty file."""
    with_site = db.query(Deployment.id).filter(
        Deployment.project_id == project.id, Deployment.site_id.isnot(None)
    )
    if scope is not None:
        with_site = with_site.filter(Deployment.id.in_(scope))
    if with_site.first() is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{export} needs at least one deployment with a camera site: "
                "deployments without a site have no coordinates and are left out."
            ),
        )


_T = TypeVar("_T")


async def _write(build_and_write: Callable[[], _T]) -> _T:
    """Build an export and write it, off the event loop: a large project
    must not freeze every other request, and the thread inherits the
    project timezone. A folder the user cannot write to is a 400 that
    says why, not a bare 500."""
    try:
        return await asyncio.to_thread(build_and_write)
    except OSError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not write to that folder: {e.strerror or e}",
        ) from e


# The five tables, in the workbook's sheet order, as the kind in their
# CSV / TSV file names.
_TABLE_FILE_NAMES = {
    "Summary": "summary",
    "Counts": "counts",
    "Detections": "detections",
    "Files": "files",
    "Deployments": "deployments",
}


@router.post("/tables", response_model=WrittenFiles)
async def export_tables(
    project_id: str,
    body: TablesExportRequest,
    db: Session = Depends(get_db),
) -> WrittenFiles:
    """Summary, Counts, Detections, Files and Deployments.

    XLSX writes one workbook with five sheets, CSV and TSV write five
    files; either way every table is streamed to disk row by row.
    """
    project = _resolve_project(project_id, db)
    scope = _scope(db, project, body.deployment_ids)
    folder = await _output_dir(body.output_dir)

    def build_and_write() -> list[str]:
        sheets = export_crud.build_spreadsheet_sheets(db, project, scope)
        if body.format == "xlsx":
            path = export_formats.unique_export_path(
                folder, export_formats.export_file_name(project.name, "spreadsheet", "xlsx")
            )
            export_formats.write_xlsx_multi(sheets, path)
            return [path.name]
        delimiter = "," if body.format == "csv" else "\t"
        names = []
        for title, headers, rows in sheets:
            path = export_formats.unique_export_path(
                folder,
                export_formats.export_file_name(
                    project.name, _TABLE_FILE_NAMES[title], body.format
                ),
            )
            export_formats.write_csv(path, headers, rows, delimiter=delimiter)
            names.append(path.name)
        return names

    try:
        files = await _write(build_and_write)
    except export_formats.XlsxRowLimitError as e:
        # The message names the table, its size and the CSV way out.
        raise HTTPException(status_code=422, detail=str(e)) from e
    return WrittenFiles(output_dir=str(folder), files=files)


_SPATIAL_WRITERS = {
    "gpkg": ("gpkg", export_formats.serialize_geopackage),
    "geojson": ("geojson", export_formats.serialize_geojson),
    "shapefile": ("zip", export_formats.serialize_shapefile_zip),
}


@router.post("/spatial", response_model=WrittenFiles)
async def export_spatial(
    project_id: str,
    body: SpatialExportRequest,
    db: Session = Depends(get_db),
) -> WrittenFiles:
    """Spatial layers (deployments, species summary). Deployments without
    a site have no coordinates and are left out; the dialog says so."""
    project = _resolve_project(project_id, db)
    scope = _scope(db, project, body.deployment_ids)
    _require_a_site(db, project, scope, "A spatial export")
    folder = await _output_dir(body.output_dir)
    extension, serialize = _SPATIAL_WRITERS[body.format]

    def build_and_write() -> str:
        layers, _skipped = export_crud.build_spatial_layers(db, project, scope)
        path = export_formats.unique_export_path(
            folder, export_formats.export_file_name(project.name, "spatial", extension)
        )
        path.write_bytes(serialize(layers))
        return path.name

    name = await _write(build_and_write)
    return WrittenFiles(output_dir=str(folder), files=[name])


@router.post("/camtrap-dp/prepare", status_code=202)
async def prepare_camtrap_dp(
    project_id: str,
    body: CamtrapExportRequest,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Start a Camtrap DP job. It writes ``camtrap-dp-<project>-<date>.zip``
    into the picked folder and reports it on completion over the job WebSocket.
    """
    project = _resolve_project(project_id, db)
    scope = _scope(db, project, body.deployment_ids)
    folder = await _output_dir(body.output_dir)

    _require_a_site(db, project, scope, "Camtrap DP")

    job = job_crud.create_job(
        db,
        JobCreate(
            type="camtrap_export",
            payload={
                "project_id": project.id,
                "deployment_ids": scope,
                "output_dir": str(folder),
                "include_thumbnails": body.include_thumbnails,
            },
        ),
    )

    from app.workers.camtrap_export_worker import process_camtrap_export_job

    ws_manager.register_start(
        job.id, lambda jid=job.id: process_camtrap_export_job(jid)
    )
    return {"job_id": job.id}


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

    Deployments through `_scope`, like every export; the file filters
    from the media options.
    """
    resolved = _scope(db, project, body.deployment_ids)
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
        deployment_folder_in_path=keeps_deployment_folder(project.mode),
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

    output_root = await _output_dir(body.output_dir)

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


def _package_scope(db: Session, project: Project, deployment_ids: list[str] | None) -> list[str]:
    """The deployment ids a package covers: the picked ones, or all. A
    package lists its deployments, so None becomes the full list."""
    resolved = _scope(db, project, deployment_ids)
    if resolved is not None:
        return resolved
    return list(
        db.execute(
            select(Deployment.id).where(Deployment.project_id == project.id)
        ).scalars()
    )


@router.get("/package/size", response_model=PackageSizeResponse)
def package_size(
    project_id: str,
    deployment_ids: str | None = Query(None),
    db: Session = Depends(get_db),
) -> PackageSizeResponse:
    """How much the export dialog is about to write: files and media
    bytes of the selected deployments, from the sizes recorded at
    ingest, no filesystem walk.

    The data is an estimate: the database's used pages times the
    selection's share of all files. Rows and embeddings grow with the
    files, so this lands in the right order of magnitude without
    measuring every table; it spreads embeddings evenly over projects
    that have them and projects that do not."""
    project = _resolve_project(project_id, db)
    scope = _package_scope(db, project, _parse_ids(deployment_ids))
    files, media_bytes = db.execute(
        select(func.count(File.id), func.coalesce(func.sum(File.size_bytes), 0)).where(
            File.deployment_id.in_(scope)
        )
    ).one()
    all_files = db.execute(select(func.count(File.id))).scalar_one()
    page_size, page_count, free_pages = (
        db.execute(text(f"PRAGMA {name}")).scalar_one()
        for name in ("page_size", "page_count", "freelist_count")
    )
    used_bytes = page_size * (page_count - free_pages)
    return PackageSizeResponse(
        deployments=len(scope),
        files=int(files),
        media_bytes=int(media_bytes),
        data_bytes=used_bytes * files // all_files if all_files else 0,
    )


@router.post("/package/prepare", status_code=202)
async def prepare_package_export(
    project_id: str,
    body: PackageExportRequest,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Start writing a project package into `output_dir`. The folder the
    package goes in must exist; the package folder itself must not."""
    project = _resolve_project(project_id, db)
    if project.mode != "research":
        raise HTTPException(
            status_code=400, detail="Only projects can be exported as a bundle"
        )
    scope = _package_scope(db, project, body.deployment_ids)
    output_dir = await _output_dir(body.output_dir)
    folders = [
        f for f in db.execute(
            select(Deployment.folder_path).where(Deployment.id.in_(scope))
        ).scalars() if f
    ]
    try:
        await asyncio.to_thread(package_destination, output_dir, project.name, folders)
    except PackageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    job = job_crud.create_job(
        db,
        JobCreate(
            type="package_export",
            payload={
                "project_id": project.id,
                "deployment_ids": scope,
                "output_dir": str(output_dir),
                "include_media": body.include_media,
            },
        ),
    )

    from app.workers.project_package_worker import process_package_export_job

    ws_manager.register_start(
        job.id, lambda jid=job.id: process_package_export_job(jid)
    )
    return {"job_id": job.id}
