"""
Files API router.

Provides endpoints for browsing and viewing files (images/videos) with detections.
"""

import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse as FastAPIFileResponse
from fastapi.responses import Response
from PIL import Image
from sqlalchemy.orm import Session

from app.api.crud import file as file_crud
from app.api.schemas.file import (
    FileResponse,
    FileUpdate,
    FileWithDetections,
    FilmstripFrame,
    FilmstripResponse,
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


# Sync `def` on purpose: decoding the filmstrip reads the clip with cv2 and
# can take a second or more, so FastAPI runs it in the threadpool instead of
# blocking the event loop. Results are cached in build_filmstrip's LRU.
@router.get("/{file_id}/filmstrip", response_model=FilmstripResponse)
def get_file_filmstrip(
    file_id: str,
    db: Session = Depends(get_db),
) -> FilmstripResponse:
    """
    Decode a small set of evenly-spaced low-res frames for a video, for the
    counts-modal gallery. Frames are generated on demand and never persisted.

    Raises:
        HTTPException: 404 if the file is missing, 400 if it is not a video.
    """
    from app.services.filmstrip_service import build_filmstrip

    file = file_crud.get_file_with_detections(db, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    if file.file_type != "video":
        raise HTTPException(status_code=400, detail="File is not a video")

    frames = build_filmstrip(file.file_path, file.frame_rate)
    return FilmstripResponse(
        frames=[FilmstripFrame(**frame) for frame in frames]
    )


# Wide enough for the largest tile the Empties grid offers (about 600px
# at a typical window), so a tile is never upscaled and soft. The decode
# of the 2048px source dominates the cost, so a wider target is free in
# CPU terms (measured: 58ms at 512px, 45ms at 768px on a real file) and
# roughly doubles the bytes, 72 KB to 148 KB, which is nothing over
# localhost. Raise this before offering a bigger tile than that.
_THUMB_MAX_WIDTH = 768
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
