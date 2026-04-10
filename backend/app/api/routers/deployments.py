"""
Deployment API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

import io
import subprocess
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import deployment as crud_deployment
from app.api.schemas.deployment import (
    DeploymentCreate,
    DeploymentResponse,
    DeploymentUpdate,
    DeploymentWithStats,
    FolderPreviewResponse,
    GPSCoordinates,
    SampleFile,
)
from app.core.logging_config import get_logger
from app.core.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.db.base import get_db
from app.services.folder_scanner import scan_folder

logger = get_logger(__name__)
router = APIRouter(prefix="/api/deployments", tags=["Deployments"])


@router.get("", response_model=list[DeploymentResponse])
def list_deployments(
    site_id: str | None = Query(None, description="Filter by site ID"),
    db: Session = Depends(get_db),
) -> list[DeploymentResponse]:
    """
    List all deployments, optionally filtered by site_id.

    Returns empty list if no deployments exist.
    """
    deployments = crud_deployment.get_deployments(db, site_id=site_id)
    return [DeploymentResponse.model_validate(d) for d in deployments]


@router.post("", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
def create_deployment(
    deployment: DeploymentCreate, db: Session = Depends(get_db)
) -> DeploymentResponse:
    """
    Create a new deployment.

    Returns 400 if site_id is invalid (foreign key constraint).
    """
    try:
        db_deployment = crud_deployment.create_deployment(db, deployment)
        logger.info(f"Created deployment for site {deployment.site_id} (ID: {db_deployment.id})")
        return DeploymentResponse.model_validate(db_deployment)
    except IntegrityError as e:
        # Foreign key constraint violation (invalid site_id)
        logger.warning(f"Failed to create deployment: site {deployment.site_id} not found")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid site_id: {deployment.site_id}",
        ) from e


@router.get("/preview-folder", response_model=FolderPreviewResponse)
def preview_folder_path(
    path: str = Query(..., description="Absolute path to folder to preview"),
) -> FolderPreviewResponse:
    """
    Preview a folder before creating a deployment.

    Scans the folder to count images/videos and check for GPS coordinates.
    Used by the frontend to validate folder selection before adding to queue.

    Returns 400 if folder doesn't exist or isn't accessible.
    """
    if not path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder path is required",
        )

    # Scan folder
    try:
        logger.info(f"Scanning folder: {path}")
        preview = scan_folder(path)
        img_count = preview['image_count']
        vid_count = preview['video_count']
        logger.info(
            f"Folder scan complete: {img_count} images, "
            f"{vid_count} videos"
        )
    except FileNotFoundError as e:
        logger.error(f"Folder not found: {path}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Folder not found: {str(e)}",
        ) from e
    except PermissionError as e:
        logger.error(f"Permission denied: {path}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(f"Error scanning folder: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scanning folder: {str(e)}",
        ) from e

    # Convert to response schema
    return FolderPreviewResponse(
        image_count=preview["image_count"],
        video_count=preview["video_count"],
        total_count=preview["total_count"],
        gps_location=GPSCoordinates(**preview["gps_location"]) if preview["gps_location"] else None,
        suggested_site_id=None,
        sample_files=[SampleFile(**sf) for sf in preview["sample_files"]],
        start_date=preview["start_date"],
        end_date=preview["end_date"],
        missing_datetime=preview["missing_datetime"],
        datetime_validation_log=preview["datetime_validation_log"],
    )


# Maximum thumbnail width for preview images (avoids sending 20 MB RAW files)
_PREVIEW_MAX_WIDTH = 800


@router.get("/preview-image")
def preview_image(
    folder: str = Query(..., description="Absolute path to deployment folder"),
    file: str = Query(..., description="Relative file path from sample_files"),
):
    """Serve a resized image from a deployment folder for datetime preview.

    Used by the DatetimeOffsetModal to show sample images so users can
    compare the burned-in pixel date with the extracted EXIF datetime.

    Security: rejects paths containing '..' and validates the resolved
    path is inside the folder.
    """
    # Block path traversal
    if ".." in file or PurePosixPath(file).is_absolute():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )

    folder_path = Path(folder)
    file_path = (folder_path / file).resolve()

    # Ensure resolved path is inside the folder
    if not str(file_path).startswith(str(folder_path.resolve())):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File path is outside the deployment folder",
        )

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file}",
        )

    ext = file_path.suffix.lower()
    is_video = ext in VIDEO_EXTENSIONS

    try:
        if is_video:
            # Extract first frame from video via ffmpeg → JPEG pipe
            result = subprocess.run(
                [
                    "ffmpeg", "-i", str(file_path),
                    "-vframes", "1",
                    "-vf", f"scale={_PREVIEW_MAX_WIDTH}:-1",
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout:
                raise RuntimeError(
                    f"ffmpeg failed (exit {result.returncode}): "
                    f"{result.stderr[:200].decode(errors='replace')}"
                )
            buf = io.BytesIO(result.stdout)
            return StreamingResponse(buf, media_type="image/jpeg")

        # Image: open with Pillow, resize, serve as JPEG
        img = Image.open(file_path)
        if img.width > _PREVIEW_MAX_WIDTH:
            ratio = _PREVIEW_MAX_WIDTH / img.width
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Failed to serve preview for {file_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read file: {str(e)}",
        ) from e


@router.get("/file-datetime")
def get_file_datetime(
    folder: str = Query(..., description="Absolute path to deployment folder"),
    file: str = Query(..., description="Relative file path"),
):
    """Extract the EXIF/metadata datetime from a single file on demand.

    Called lazily by the DatetimeOffsetModal as the user navigates through
    images, so we don't pay the cost of extracting all 10k+ datetimes
    during the initial folder scan.
    """
    if ".." in file or PurePosixPath(file).is_absolute():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )

    folder_path = Path(folder)
    file_path = (folder_path / file).resolve()

    if not str(file_path).startswith(str(folder_path.resolve())):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File path is outside the deployment folder",
        )

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file}",
        )

    dt = None
    ext = file_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        from app.services.folder_scanner import _extract_exif_date_single

        dt = _extract_exif_date_single(file_path)
    elif ext in VIDEO_EXTENSIONS:
        from app.utils.media_dates import extract_video_date

        dt = extract_video_date(file_path)

    return {"file_datetime": dt.isoformat() if dt else None}


@router.get("/{deployment_id}", response_model=DeploymentResponse)
def get_deployment(deployment_id: str, db: Session = Depends(get_db)) -> DeploymentResponse:
    """
    Get deployment by ID.

    Returns 404 if deployment doesn't exist.
    """
    db_deployment = crud_deployment.get_deployment(db, deployment_id)
    if db_deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )
    return DeploymentResponse.model_validate(db_deployment)


@router.patch("/{deployment_id}", response_model=DeploymentResponse)
def update_deployment(
    deployment_id: str, deployment: DeploymentUpdate, db: Session = Depends(get_db)
) -> DeploymentResponse:
    """
    Update an existing deployment.

    Returns 404 if deployment doesn't exist.
    Use this endpoint to re-link folder paths if files have moved.
    """
    try:
        db_deployment = crud_deployment.update_deployment(db, deployment_id, deployment)
        if db_deployment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment with id '{deployment_id}' not found",
            )
        return DeploymentResponse.model_validate(db_deployment)
    except IntegrityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database constraint violation",
        ) from e


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_deployment(deployment_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a deployment.

    Returns 404 if deployment doesn't exist.
    Cascades deletion to all files and events.
    """
    deleted = crud_deployment.delete_deployment(db, deployment_id)
    if not deleted:
        logger.warning(f"Cannot delete deployment: {deployment_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )
    logger.info(f"Deleted deployment: {deployment_id} (cascaded to files and events)")


@router.get("/{deployment_id}/stats", response_model=DeploymentWithStats)
def get_deployment_stats(deployment_id: str, db: Session = Depends(get_db)) -> DeploymentWithStats:
    """
    Get deployment with statistics.

    Returns deployment info plus counts of files, events, and detections.
    Returns 404 if deployment doesn't exist.
    """
    db_deployment = crud_deployment.get_deployment(db, deployment_id)
    if db_deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )

    stats = crud_deployment.get_deployment_stats(db, deployment_id)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )

    # Combine deployment data with stats
    deployment_dict = DeploymentResponse.model_validate(db_deployment).model_dump()
    deployment_dict.update(stats)

    return DeploymentWithStats(**deployment_dict)


@router.post("/{deployment_id}/preview-folder", response_model=FolderPreviewResponse)
def preview_deployment_folder(
    deployment_id: str, db: Session = Depends(get_db)
) -> FolderPreviewResponse:
    """
    Preview a deployment folder before running analysis.

    Scans the folder to count images/videos and check for GPS coordinates.
    Does NOT create File records - that happens after MegaDetector runs.

    Returns 404 if deployment doesn't exist.
    Returns 400 if deployment has no folder_path set.
    Returns 400 if folder doesn't exist or isn't accessible.
    """
    # Get deployment
    db_deployment = crud_deployment.get_deployment(db, deployment_id)
    if db_deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )

    # Check folder_path is set
    if not db_deployment.folder_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deployment has no folder_path set. Please set folder_path first.",
        )

    # Scan folder
    try:
        logger.info(f"Scanning folder for deployment {deployment_id}: {db_deployment.folder_path}")
        preview = scan_folder(db_deployment.folder_path)
        logger.info(
            f"Folder scan complete for {deployment_id}: "
            f"{preview['image_count']} images, {preview['video_count']} videos"
        )
    except FileNotFoundError as e:
        logger.error(
            f"Folder not found for deployment {deployment_id}: {db_deployment.folder_path}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Folder not found: {str(e)}",
        ) from e
    except PermissionError as e:
        logger.error(
            f"Permission denied for deployment {deployment_id}: {db_deployment.folder_path}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(
            f"Error scanning folder for deployment {deployment_id}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scanning folder: {str(e)}",
        ) from e

    # TODO: Site matching based on GPS (for later)
    suggested_site_id = None

    # Convert to response schema
    return FolderPreviewResponse(
        image_count=preview["image_count"],
        video_count=preview["video_count"],
        total_count=preview["total_count"],
        gps_location=GPSCoordinates(**preview["gps_location"]) if preview["gps_location"] else None,
        suggested_site_id=suggested_site_id,
        sample_files=preview["sample_files"],
        start_date=preview["start_date"],
        end_date=preview["end_date"],
        missing_datetime=preview["missing_datetime"],
        datetime_validation_log=preview["datetime_validation_log"],
    )
