"""
Project API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import project as crud_project
from app.api.schemas.project import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ProjectWithStats,
)
from app.db.base import get_db

router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("", response_model=list[ProjectResponse])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectResponse]:
    """
    List all projects.

    Returns empty list if no projects exist.
    """
    projects = crud_project.get_projects(db)
    return [ProjectResponse.model_validate(p) for p in projects]


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    project: ProjectCreate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """
    Create a new project.

    Returns 409 if project name already exists.
    """
    try:
        db_project = crud_project.create_project(db, project)
        return ProjectResponse.model_validate(db_project)
    except IntegrityError as e:
        # Duplicate name constraint violation
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project with name '{project.name}' already exists",
        ) from e


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    """
    Get project by ID.

    Returns 404 if project doesn't exist.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )
    return ProjectResponse.model_validate(db_project)


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str, project: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """
    Update an existing project.

    Returns 404 if project doesn't exist.
    Returns 409 if new name conflicts with existing project.
    """
    try:
        db_project = crud_project.update_project(db, project_id, project)
        if db_project is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with id '{project_id}' not found",
            )
        return ProjectResponse.model_validate(db_project)
    except IntegrityError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project name already exists",
        ) from e


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a project.

    Returns 404 if project doesn't exist.
    Cascades deletion to all sites, deployments, files, etc.
    """
    deleted = crud_project.delete_project(db, project_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )


@router.get("/{project_id}/stats", response_model=ProjectWithStats)
def get_project_stats(
    project_id: str, db: Session = Depends(get_db)
) -> ProjectWithStats:
    """
    Get project with statistics.

    Returns project info plus counts of sites, deployments, files, and detections.
    Returns 404 if project doesn't exist.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    stats = crud_project.get_project_stats(db, project_id)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    # Combine project data with stats
    project_dict = ProjectResponse.model_validate(db_project).model_dump()
    project_dict.update(stats)

    return ProjectWithStats(**project_dict)
