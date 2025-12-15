"""
Site API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import site as crud_site
from app.api.schemas.site import SiteCreate, SiteResponse, SiteUpdate
from app.core.logging_config import get_logger
from app.db.base import get_db

logger = get_logger(__name__)
router = APIRouter(prefix="/api/sites", tags=["Sites"])


@router.get("", response_model=list[SiteResponse])
def list_sites(
    project_id: str | None = Query(None, description="Filter by project ID"),
    db: Session = Depends(get_db),
) -> list[SiteResponse]:
    """
    List all sites, optionally filtered by project.

    Returns empty list if no sites exist.
    """
    sites = crud_site.get_sites(db, project_id=project_id)
    return [SiteResponse.model_validate(s) for s in sites]


@router.post("", response_model=SiteResponse, status_code=status.HTTP_201_CREATED)
def create_site(site: SiteCreate, db: Session = Depends(get_db)) -> SiteResponse:
    """
    Create a new site.

    Returns 400 if project doesn't exist.
    Returns 409 if site name already exists in the project.
    """
    try:
        db_site = crud_site.create_site(db, site)
        logger.info(f"Created site: {site.name} in project {site.project_id} (ID: {db_site.id})")
        return SiteResponse.model_validate(db_site)
    except IntegrityError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)

        # Check if it's a foreign key error (project doesn't exist)
        if "FOREIGN KEY" in error_msg or "foreign key" in error_msg.lower():
            logger.warning(f"Failed to create site: project {site.project_id} not found")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Project with id '{site.project_id}' does not exist",
            ) from e

        # Otherwise it's likely a unique constraint violation (duplicate name)
        logger.warning(f"Failed to create site '{site.name}': duplicate name in project")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Site with name '{site.name}' already exists in this project",
        ) from e


@router.get("/{site_id}", response_model=SiteResponse)
def get_site(site_id: str, db: Session = Depends(get_db)) -> SiteResponse:
    """
    Get site by ID.

    Returns 404 if site doesn't exist.
    """
    db_site = crud_site.get_site(db, site_id)
    if db_site is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Site with id '{site_id}' not found",
        )
    return SiteResponse.model_validate(db_site)


@router.patch("/{site_id}", response_model=SiteResponse)
def update_site(
    site_id: str, site: SiteUpdate, db: Session = Depends(get_db)
) -> SiteResponse:
    """
    Update an existing site.

    Returns 404 if site doesn't exist.
    Returns 409 if new name conflicts with existing site in same project.
    """
    try:
        db_site = crud_site.update_site(db, site_id, site)
        if db_site is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Site with id '{site_id}' not found",
            )
        return SiteResponse.model_validate(db_site)
    except IntegrityError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Site name already exists in this project",
        ) from e


@router.delete("/{site_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_site(site_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a site.

    Returns 404 if site doesn't exist.
    Cascades deletion to all deployments, files, etc.
    """
    deleted = crud_site.delete_site(db, site_id)
    if not deleted:
        logger.warning(f"Cannot delete site: {site_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Site with id '{site_id}' not found",
        )
    logger.info(f"Deleted site: {site_id} (cascaded to deployments and files)")
