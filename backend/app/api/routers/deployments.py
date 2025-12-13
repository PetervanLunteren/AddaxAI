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
)
from app.db.base import get_db

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
        return DeploymentResponse.model_validate(db_deployment)
    except IntegrityError as e:
        # Foreign key constraint violation (invalid site_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid site_id: {deployment.site_id}",
        ) from e


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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )


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
