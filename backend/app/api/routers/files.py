"""
Files API router.

Provides endpoints for browsing and viewing files (images/videos) with detections.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse as FastAPIFileResponse
from sqlalchemy.orm import Session

from app.api.crud import file as file_crud
from app.api.schemas.file import FileResponse, FileWithDetections
from app.db.base import get_db

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("", response_model=list[FileResponse])
def list_files(
    deployment_id: str | None = Query(None, description="Filter by deployment ID"),
    project_id: str | None = Query(None, description="Filter by project ID"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    db: Session = Depends(get_db),
):
    """
    List files with optional filters.

    Args:
        deployment_id: Optional deployment ID filter
        project_id: Optional project ID filter
        skip: Number of records to skip
        limit: Number of records to return
        db: Database session

    Returns:
        List of files
    """
    if project_id:
        files = file_crud.get_files_by_project(db, project_id, skip=skip, limit=limit)
    elif deployment_id:
        files = file_crud.get_files_by_deployment(db, deployment_id, skip=skip, limit=limit)
    else:
        files = file_crud.get_files(db, skip=skip, limit=limit)

    return files


@router.get("/{file_id}", response_model=FileWithDetections)
def get_file(
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

    return file


@router.get("/{file_id}/image")
def get_file_image(
    file_id: str,
    db: Session = Depends(get_db),
):
    """
    Serve the actual image file.

    Args:
        file_id: File ID
        db: Database session

    Returns:
        Image file

    Raises:
        HTTPException: If file not found or path invalid
    """
    file = file_crud.get_file_with_detections(db, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = Path(file.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FastAPIFileResponse(
        path=str(file_path),
        media_type=f"image/{file.file_format}",
        filename=file_path.name,
    )
