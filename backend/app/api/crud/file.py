"""
CRUD operations for files.
"""

from sqlalchemy.orm import Session, joinedload

from app.models import Deployment, File


def get_files(db: Session, skip: int = 0, limit: int = 100) -> list[File]:
    """
    Get all files with pagination.

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Number of records to return

    Returns:
        List of files
    """
    return db.query(File).order_by(File.timestamp.desc()).offset(skip).limit(limit).all()


def get_files_by_deployment(
    db: Session, deployment_id: str, skip: int = 0, limit: int = 100
) -> list[File]:
    """
    Get files by deployment ID.

    Args:
        db: Database session
        deployment_id: Deployment ID
        skip: Number of records to skip
        limit: Number of records to return

    Returns:
        List of files
    """
    return (
        db.query(File)
        .filter(File.deployment_id == deployment_id)
        .order_by(File.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_files_by_project(
    db: Session, project_id: str, skip: int = 0, limit: int = 100
) -> list[File]:
    """
    Get files by project ID.

    Args:
        db: Database session
        project_id: Project ID
        skip: Number of records to skip
        limit: Number of records to return

    Returns:
        List of files
    """
    return (
        db.query(File)
        .join(Deployment)
        .join(Deployment.site)
        .filter(Deployment.site.has(project_id=project_id))
        .order_by(File.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_file_with_detections(db: Session, file_id: str) -> File | None:
    """
    Get file by ID with detections loaded.

    Args:
        db: Database session
        file_id: File ID

    Returns:
        File with detections or None if not found
    """
    return (
        db.query(File)
        .options(joinedload(File.detections))
        .filter(File.id == file_id)
        .first()
    )
