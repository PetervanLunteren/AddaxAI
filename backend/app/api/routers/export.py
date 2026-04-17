"""
Export endpoints for project data.

Three synchronous streaming endpoints, one per export type:

- ``/observations``: flat one-row-per-species-per-image in CSV / TSV / XLSX.
- ``/spatial``: GIS layers in GeoJSON / Shapefile (ZIP) / GeoPackage.
- ``/camtrap-dp``: Camera Trap Data Package v1.0 (GBIF-compatible) as a ZIP.

See ``app/api/crud/export.py`` for the data layer and
``app/api/crud/export_formats.py`` for the serializers.
"""

from __future__ import annotations

import json as _json
from datetime import date
from io import BytesIO
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.crud import export as export_crud
from app.api.crud import export_formats
from app.db.base import get_db
from app.models import Deployment, Project, Site
from app.utils.datetime_serialization import set_active_project_timezone

router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["Export"])


def _resolve_project(project_id: str, db: Session) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    set_active_project_timezone(project.timezone)
    return project


def _filename_base(project: Project, kind: str) -> str:
    slug = export_formats.slugify(project.name)
    today = date.today().isoformat()
    return f"{kind}-{slug}-{today}"


def _attachment_headers(filename: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


@router.get("/observations")
async def export_observations(
    project_id: str,
    format: Literal["csv", "tsv", "xlsx"] = Query("csv"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Flat observations: one row per species per image."""
    project = _resolve_project(project_id, db)
    scoped = export_crud.get_scoped_detection_rows(db, project)
    headers, rows = export_crud.build_observation_rows(db, project, scoped)

    base = _filename_base(project, "observations")
    if format == "xlsx":
        payload = export_formats.serialize_xlsx(headers, rows, sheet_title="Observations")
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


@router.get("/spatial")
async def export_spatial(
    project_id: str,
    format: Literal["geojson", "shapefile", "gpkg"] = Query("geojson"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Spatial layers (deployments, observations, species summary)."""
    project = _resolve_project(project_id, db)
    scoped = export_crud.get_scoped_detection_rows(db, project)
    layers = export_crud.build_spatial_layers(db, project, scoped)

    base = _filename_base(project, "spatial")
    if format == "shapefile":
        payload = export_formats.serialize_shapefile_zip(layers)
        return StreamingResponse(
            BytesIO(payload),
            media_type="application/zip",
            headers=_attachment_headers(f"{base}.zip"),
        )
    if format == "gpkg":
        payload = export_formats.serialize_geopackage(layers)
        return StreamingResponse(
            BytesIO(payload),
            media_type="application/geopackage+sqlite3",
            headers=_attachment_headers(f"{base}.gpkg"),
        )
    payload = export_formats.serialize_geojson(layers)
    return StreamingResponse(
        BytesIO(payload),
        media_type="application/geo+json",
        headers=_attachment_headers(f"{base}.geojson"),
    )


@router.get("/camtrap-dp")
async def export_camtrap_dp(
    project_id: str,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """CamTrap DP v1.0 package (ZIP with datapackage.json + three CSVs)."""
    project = _resolve_project(project_id, db)

    has_deployments = (
        db.query(Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .filter(Site.project_id == project.id)
        .first()
        is not None
    )
    if not has_deployments:
        raise HTTPException(
            status_code=422,
            detail=(
                "CamTrap DP requires at least one deployment in the project."
            ),
        )

    scoped = export_crud.get_scoped_detection_rows(db, project)
    deps_rows, media_rows, obs_rows, datapackage = export_crud.build_camtrap_dp_tables(
        db, project, scoped
    )
    deps_h, media_h, obs_h = export_crud.camtrap_dp_headers()

    deps_csv = export_formats.serialize_csv(deps_h, deps_rows)
    media_csv = export_formats.serialize_csv(media_h, media_rows)
    obs_csv = export_formats.serialize_csv(obs_h, obs_rows)
    datapackage_bytes = _json.dumps(datapackage, indent=2, ensure_ascii=False).encode(
        "utf-8"
    )
    payload = export_formats.build_camtrap_dp_zip(
        datapackage_bytes, deps_csv, media_csv, obs_csv
    )

    base = _filename_base(project, "camtrap-dp")
    return StreamingResponse(
        BytesIO(payload),
        media_type="application/zip",
        headers=_attachment_headers(f"{base}.zip"),
    )
