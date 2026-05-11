"""
Files API router.

Provides endpoints for browsing and viewing files (images/videos) with detections.
"""

import io
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse as FastAPIFileResponse
from fastapi.responses import Response
from PIL import Image
from sqlalchemy.orm import Session

from app.api.crud import file as file_crud
from app.api.crud.event import VERIFY_SORT_VALUES
from app.api.schemas.file import (
    AdjacentFilesResponse,
    FileResponse,
    FileSummary,
    FileUpdate,
    FileVerificationStats,
    FileWithDetections,
)
from app.db.base import get_db
from app.models import Deployment, File, Project
from app.utils.datetime_serialization import set_active_project_timezone

router = APIRouter(prefix="/api/files", tags=["files"])


def _set_project_tz_for_file(db: Session, file_id: str) -> None:
    """Activate project timezone for a file based on its deployment chain."""
    tz = (
        db.query(Project.timezone)
        .join(Deployment, Deployment.project_id == Project.id)
        .join(File, File.deployment_id == Deployment.id)
        .filter(File.id == file_id)
        .scalar()
    )
    if tz:
        set_active_project_timezone(tz)


def _set_project_tz(db: Session, project_id: str) -> None:
    """Activate the project's timezone from project_id."""
    tz = db.query(Project.timezone).filter(Project.id == project_id).scalar()
    if tz:
        set_active_project_timezone(tz)


def _parse_verify_filter_params(
    site_ids: str | None,
    date_from: str | None,
    date_to: str | None,
    labels: str | None,
    verification: str | None,
    min_confidence: float | None,
    max_confidence: float | None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    sort: str = "newest",
    seed: int | None = None,
) -> dict:
    """Parse shared filter query params for the Files verify endpoints."""
    if sort not in VERIFY_SORT_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f"sort must be one of: {sorted(VERIFY_SORT_VALUES)}",
        )

    return dict(
        site_ids=site_ids.split(",") if site_ids else None,
        date_from=datetime.fromisoformat(date_from) if date_from else None,
        date_to=datetime.fromisoformat(date_to) if date_to else None,
        labels=labels.split(",") if labels else None,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        sort=sort,
        seed=seed,
    )


def _apply_project_threshold(filters: dict, project_id: str, db: Session) -> dict:
    """Inject the project's detection threshold as `project_floor`.

    Same rule used by the events endpoints: a detection is visible when
    `confidence >= floor OR verified`. The floor and the user's
    `min_confidence` are kept as separate filter knobs so a verified
    low-confidence detection passes the floor's OR clause but cannot
    sneak past a narrow user-set min.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return filters
    filters["project_floor"] = project.detection_threshold
    return filters


@router.get("/list-for-verify", response_model=list[FileSummary])
async def list_files_for_verify(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated taxonomy IDs"),
    verification: str | None = Query(
        None, description="Filter: 'verified', 'unverified', or 'all'"
    ),
    flagged: str | None = Query(
        None, description="Filter: 'flagged', 'not_flagged', or 'all'"
    ),
    favorited: str | None = Query(
        None, description="Filter: 'favorited', 'not_favorited', or 'all'"
    ),
    empty: str | None = Query(
        None, description="Filter: 'show_only', 'hide', or 'all'"
    ),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    sort: str = Query("newest"),
    seed: int | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(48, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List file summaries for the Files verify tab.

    Returns one row per media item (images and videos). Raw frame rows
    are excluded; video tiles surface through best_frame_path via the
    existing /api/files/{id}/image endpoint.

    `async def` so the active project timezone ContextVar set below is
    visible to the datetime serializer (see DEVELOPERS.md).
    """
    _set_project_tz(db, project_id)
    filters = _parse_verify_filter_params(
        site_ids, date_from, date_to, labels, verification,
        min_confidence, max_confidence,
        flagged=flagged, favorited=favorited, empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        sort=sort, seed=seed,
    )
    _apply_project_threshold(filters, project_id, db)
    return file_crud.get_files_for_verify(
        db, project_id, skip=skip, limit=limit, **filters,
    )


@router.get("/count-for-verify")
def count_files_for_verify(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated taxonomy IDs"),
    verification: str | None = Query(None),
    flagged: str | None = Query(None),
    favorited: str | None = Query(None),
    empty: str | None = Query(None),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """Total file count for the Files verify tab with the given filters."""
    filters = _parse_verify_filter_params(
        site_ids, date_from, date_to, labels, verification,
        min_confidence, max_confidence,
        flagged=flagged, favorited=favorited, empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
    )
    _apply_project_threshold(filters, project_id, db)
    # Count is sort-invariant; the count CRUD doesn't accept sort/seed.
    filters.pop("sort", None)
    filters.pop("seed", None)
    count = file_crud.count_files_for_verify(db, project_id, **filters)
    return {"count": count}


@router.get("/verification-stats", response_model=FileVerificationStats)
def get_file_verification_stats_endpoint(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated taxonomy IDs"),
    verification: str | None = Query(None),
    flagged: str | None = Query(None),
    favorited: str | None = Query(None),
    empty: str | None = Query(None),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """Aggregate verified/total file counts for the Files verify tab."""
    filters = _parse_verify_filter_params(
        site_ids, date_from, date_to, labels, verification,
        min_confidence, max_confidence,
        flagged=flagged, favorited=favorited, empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
    )
    _apply_project_threshold(filters, project_id, db)
    # Verification stats are sort-invariant.
    filters.pop("sort", None)
    filters.pop("seed", None)
    return file_crud.get_file_verification_stats(db, project_id, **filters)


@router.get("/stats/observation-types")
def get_observation_type_stats(
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_db),
):
    """
    Get observation type counts for a project.

    Returns:
        Dict mapping observation_type -> count
    """
    return file_crud.get_observation_type_stats(db, project_id)


@router.get("", response_model=list[FileResponse])
async def list_files(
    deployment_id: str | None = Query(None, description="Filter by deployment ID"),
    project_id: str | None = Query(None, description="Filter by project ID"),
    observation_type: str | None = Query(None, description="Filter by observation type"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    db: Session = Depends(get_db),
):
    """
    List files with optional filters.

    Args:
        deployment_id: Optional deployment ID filter
        project_id: Optional project ID filter
        observation_type: Optional observation type filter
        skip: Number of records to skip
        limit: Number of records to return
        db: Database session

    Returns:
        List of files
    """
    if project_id:
        _set_project_tz(db, project_id)
        files = file_crud.get_files_by_project(
            db, project_id, skip=skip, limit=limit, observation_type=observation_type
        )
    elif deployment_id:
        tz = (
            db.query(Project.timezone)
            .join(Deployment, Deployment.project_id == Project.id)
            .filter(Deployment.id == deployment_id)
            .scalar()
        )
        if tz:
            set_active_project_timezone(tz)
        files = file_crud.get_files_by_deployment(
            db, deployment_id, skip=skip, limit=limit, observation_type=observation_type
        )
    else:
        files = file_crud.get_files(db, skip=skip, limit=limit, observation_type=observation_type)

    return files


@router.patch("/{file_id}", response_model=FileResponse)
async def update_file(
    file_id: str,
    update: FileUpdate,
    db: Session = Depends(get_db),
):
    """
    Update file verification status and/or notes.
    """
    file = file_crud.update_file(db, file_id, update)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    _set_project_tz_for_file(db, file_id)
    return file


@router.get("/{file_id}", response_model=FileWithDetections)
async def get_file(
    file_id: str,
    db: Session = Depends(get_db),
):
    """
    Get file by ID with detections.

    Args:
        file_id: File ID
        db: Database session

    Returns:
        File with detections

    Raises:
        HTTPException: If file not found
    """
    file = file_crud.get_file_with_detections(db, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    _set_project_tz_for_file(db, file_id)
    return file


VIDEO_MEDIA_TYPES = {
    "mp4": "video/mp4",
    "m4v": "video/mp4",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
    "wmv": "video/x-ms-wmv",
    "flv": "video/x-flv",
}


@router.get("/{file_id}/video")
def get_file_video(
    file_id: str,
    db: Session = Depends(get_db),
):
    """
    Serve the raw video file.

    Returns:
        Video file with appropriate Content-Type

    Raises:
        HTTPException: If file not found, not a video, or video file missing on disk
    """
    file = file_crud.get_file_with_detections(db, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    if file.file_type != "video":
        raise HTTPException(status_code=400, detail="File is not a video")

    file_path = Path(file.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    media_type = VIDEO_MEDIA_TYPES.get(
        (file.file_format or "").lower(), "video/mp4"
    )

    return FastAPIFileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )


_THUMB_MAX_WIDTH = 512
_THUMB_JPEG_QUALITY = 85
# Browser cache lifetime for image responses. File IDs are stable per
# deployment, so a long max-age is safe and means subsequent verifies
# of the same image never round-trip to the server.
_IMAGE_CACHE_HEADERS = {"Cache-Control": "public, max-age=86400, immutable"}


def _render_thumbnail_bytes(source_path: Path) -> bytes:
    """Resize an image to thumbnail width and return JPEG bytes.

    Done in-memory to avoid filling user_data_dir with cached
    thumbnails (camera-trap projects can have hundreds of thousands of
    files). Browser-level caching via Cache-Control covers repeat views.
    """
    with Image.open(source_path) as img:
        img = img.convert("RGB")
        if img.width > _THUMB_MAX_WIDTH:
            ratio = _THUMB_MAX_WIDTH / img.width
            new_height = int(img.height * ratio)
            img = img.resize((_THUMB_MAX_WIDTH, new_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_THUMB_JPEG_QUALITY)
    return buf.getvalue()


@router.get("/{file_id}/image")
def get_file_image(
    file_id: str,
    size: str | None = Query(
        None,
        description=(
            "Optional size hint. 'thumb' returns a 512px-wide JPEG "
            "(in-memory resize). Omit for the original file."
        ),
    ),
    db: Session = Depends(get_db),
):
    """
    Serve the actual image file, or a resized thumbnail when requested.

    Args:
        file_id: File ID
        size: 'thumb' for a 512px JPEG, otherwise full original
        db: Database session

    Returns:
        Image bytes with Cache-Control headers for browser caching

    Raises:
        HTTPException: If file not found or path invalid
    """
    file = file_crud.get_file_with_detections(db, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    # For videos, serve the best frame JPEG instead of the video file
    if file.file_type == "video" and file.best_frame_path:
        source_path = Path(file.best_frame_path)
        source_media_type = "image/jpeg"
    else:
        source_path = Path(file.file_path)
        source_media_type = f"image/{file.file_format}"

    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    if size == "thumb":
        thumb_bytes = _render_thumbnail_bytes(source_path)
        return Response(
            content=thumb_bytes,
            media_type="image/jpeg",
            headers=_IMAGE_CACHE_HEADERS,
        )

    return FastAPIFileResponse(
        path=str(source_path),
        media_type=source_media_type,
        filename=source_path.name,
        headers=_IMAGE_CACHE_HEADERS,
    )


@router.get("/{file_id}/adjacent", response_model=AdjacentFilesResponse)
def get_adjacent_files(
    file_id: str,
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated taxonomy IDs"),
    verification: str | None = Query(None),
    flagged: str | None = Query(None),
    favorited: str | None = Query(None),
    empty: str | None = Query(None),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    sort: str = Query("newest"),
    seed: int | None = Query(None),
    db: Session = Depends(get_db),
):
    """Adjacent file IDs for file-to-file navigation in the Files verify tab."""
    filters = _parse_verify_filter_params(
        site_ids, date_from, date_to, labels, verification,
        min_confidence, max_confidence,
        flagged=flagged, favorited=favorited, empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        sort=sort, seed=seed,
    )
    _apply_project_threshold(filters, project_id, db)
    return file_crud.get_adjacent_files_for_verify(
        db, file_id, project_id, **filters,
    )
