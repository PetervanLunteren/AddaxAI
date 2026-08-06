"""
Site API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import site as crud_site
from app.api.schemas.csv_import import (
    CsvImportProblem,
    CsvImportResult,
    SiteImportPreview,
    SiteImportRow,
)
from app.api.schemas.site import (
    SiteCreate,
    SiteInfoResponse,
    SiteResponse,
    SiteUpdate,
    SiteWithStats,
)
from app.core.logging_config import get_logger
from app.db.base import get_db
from app.models import Project
from app.services.csv_import import MAX_CSV_BYTES, drop_problem_rows
from app.services.csv_import_sites import parse_site_csv, validate_site_rows

logger = get_logger(__name__)
router = APIRouter(prefix="/api/sites", tags=["Sites"])


def _require_project(db: Session, project_id: str) -> None:
    """404 when the project is gone, with the wording the other routers use."""
    if db.get(Project, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )


def _read_csv_upload(file: UploadFile) -> bytes:
    """The uploaded bytes, refusing anything too big to hold in memory."""
    contents = file.file.read()
    if len(contents) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file is larger than 2 MB. Import your sites in smaller files.",
        )
    return contents


def _check_site_csv(
    db: Session, project_id: str, contents: bytes
) -> tuple[list[SiteImportRow], list[CsvImportProblem]]:
    """Parse and validate in one go, the way both import routes need it.

    Rows that turned out to have a problem are dropped, so what comes back is
    exactly what would be created.
    """
    rows, problems = parse_site_csv(contents)
    problems += validate_site_rows(db, project_id, rows)
    # File-level problems (no row number) first, then in file order.
    problems.sort(key=lambda p: (p.row is not None, p.row or 0))
    return drop_problem_rows(rows, problems), problems


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


@router.post("/import/preview", response_model=SiteImportPreview)
def preview_site_import(
    file: UploadFile,
    project_id: str = Query(..., description="Project the sites would be added to"),
    db: Session = Depends(get_db),
) -> SiteImportPreview:
    """
    Check a site CSV without writing anything.

    Always 200: per-row problems are the expected case and are reported in
    the body, not raised. An empty `problems` list means the same file can be
    posted to /import.
    """
    _require_project(db, project_id)
    rows, problems = _check_site_csv(db, project_id, _read_csv_upload(file))
    return SiteImportPreview(rows=rows, problems=problems)


@router.post("/import", response_model=CsvImportResult)
def import_sites(
    file: UploadFile,
    project_id: str = Query(..., description="Project the sites are added to"),
    db: Session = Depends(get_db),
) -> CsvImportResult:
    """
    Import a site CSV, all or nothing.

    The file is checked again rather than trusting the preview: the project
    may have gained a clashing site name in between. Any problem means
    nothing is written and `imported` is 0.
    """
    _require_project(db, project_id)
    rows, problems = _check_site_csv(db, project_id, _read_csv_upload(file))
    if problems:
        return CsvImportResult(imported=0, problems=problems)

    creates = [
        SiteCreate(
            project_id=project_id,
            name=row.name,
            latitude=row.latitude,
            longitude=row.longitude,
            elevation_m=row.elevation_m,
            habitat_type=row.habitat_type,
            notes=row.notes,
        )
        for row in rows
    ]

    try:
        created = crud_site.create_sites_bulk(db, creates)
    except IntegrityError as e:
        # Unreachable unless a site was created between the check above and
        # the insert. Roll back or the session stays unusable.
        db.rollback()
        logger.warning(f"Site CSV import failed for project {project_id}: {e}")
        return CsvImportResult(
            imported=0,
            problems=[
                CsvImportProblem(
                    message=(
                        "The sites could not be saved because the project changed "
                        "during the import. Import the file again."
                    )
                )
            ],
        )

    logger.info(f"Imported {len(created)} sites into project {project_id}")
    return CsvImportResult(imported=len(created), problems=[])


@router.get("/with-stats", response_model=list[SiteWithStats])
def list_sites_with_stats(
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_db),
) -> list[SiteWithStats]:
    """
    List all sites for a project with deployment counts.

    Used by the metadata management page.
    """
    rows = crud_site.get_sites_with_stats(db, project_id=project_id)
    return [SiteWithStats(**row) for row in rows]


@router.get("/{site_id}/info", response_model=SiteInfoResponse)
async def site_info(
    site_id: str, db: Session = Depends(get_db)
) -> SiteInfoResponse:
    """
    Investigation-level payload for the Sites → Info sheet.

    Aggregates across every deployment at this site: file and size
    totals, verification progress, event / observation counts, the
    detection-category breakdown, top species, trap nights, rate, and
    the first and last capture timestamps. Returns 404 if the site
    does not exist.

    `async def` so the project-timezone ContextVar reaches the
    response serializer.
    """
    info = crud_site.get_site_info(db, site_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Site with id '{site_id}' not found",
        )
    return info


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
    Returns 409 if the site has any deployments. Deployments must be
    reassigned or deleted first to avoid accidentally orphaning data.
    """
    db_site = crud_site.get_site(db, site_id)
    if db_site is None:
        logger.warning(f"Cannot delete site: {site_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Site with id '{site_id}' not found",
        )

    deployment_count = len(db_site.deployments)
    if deployment_count > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot delete site with {deployment_count} deployment(s). "
                "Reassign or delete the deployments first."
            ),
        )

    crud_site.delete_site(db, site_id)
    logger.info(f"Deleted site: {site_id}")
