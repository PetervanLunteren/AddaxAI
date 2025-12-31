"""
Deployment API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
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
)
from app.core.logging_config import get_logger
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


@router.post(
    "", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED
)
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
        logger.info(
            f"Folder scan complete: {preview['image_count']} images, {preview['video_count']} videos"
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
        logger.error(
            f"Error scanning folder: {type(e).__name__}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scanning folder: {str(e)}",
        ) from e

    # Convert to response schema
    return FolderPreviewResponse(
        image_count=preview["image_count"],
        video_count=preview["video_count"],
        total_count=preview["total_count"],
        gps_location=GPSCoordinates(**preview["gps_location"])
        if preview["gps_location"]
        else None,
        suggested_site_id=None,
        sample_files=preview["sample_files"],
        start_date=preview["start_date"],
        end_date=preview["end_date"],
        missing_datetime=preview["missing_datetime"],
        datetime_validation_log=preview["datetime_validation_log"],
    )


@router.get("/{deployment_id}", response_model=DeploymentResponse)
def get_deployment(
    deployment_id: str, db: Session = Depends(get_db)
) -> DeploymentResponse:
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
def get_deployment_stats(
    deployment_id: str, db: Session = Depends(get_db)
) -> DeploymentWithStats:
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
        logger.error(f"Folder not found for deployment {deployment_id}: {db_deployment.folder_path}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Folder not found: {str(e)}",
        ) from e
    except PermissionError as e:
        logger.error(f"Permission denied for deployment {deployment_id}: {db_deployment.folder_path}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(
            f"Error scanning folder for deployment {deployment_id}: {type(e).__name__}: {e}",
            exc_info=True
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
        gps_location=GPSCoordinates(**preview["gps_location"])
        if preview["gps_location"]
        else None,
        suggested_site_id=suggested_site_id,
        sample_files=preview["sample_files"],
        start_date=preview["start_date"],
        end_date=preview["end_date"],
        missing_datetime=preview["missing_datetime"],
        datetime_validation_log=preview["datetime_validation_log"],
    )
