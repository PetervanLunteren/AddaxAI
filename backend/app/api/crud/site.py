"""
CRUD operations for Site model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.site import SiteCreate, SiteUpdate
from app.models import Site


def get_sites(db: Session, project_id: str | None = None) -> list[Site]:
    """
    Get all sites, optionally filtered by project.

    Returns empty list if no sites exist.
    """
    query = select(Site).order_by(Site.created_at.desc())
    if project_id is not None:
        query = query.where(Site.project_id == project_id)

    result = db.execute(query)
    return list(result.scalars().all())


def get_site(db: Session, site_id: str) -> Site | None:
    """
    Get site by ID.

    Returns None if site doesn't exist.
    """
    result = db.execute(select(Site).where(Site.id == site_id))
    return result.scalar_one_or_none()


def create_site(db: Session, site: SiteCreate) -> Site:
    """
    Create a new site.

    Crashes if:
    - Project doesn't exist (foreign key constraint)
    - Duplicate site name in same project (unique constraint)
    This is intentional - we want to surface errors immediately.
    """
    db_site = Site(
        project_id=site.project_id,
        name=site.name,
        latitude=site.latitude,
        longitude=site.longitude,
        elevation_m=site.elevation_m,
        habitat_type=site.habitat_type,
        notes=site.notes,
    )
    db.add(db_site)
    db.commit()
    db.refresh(db_site)
    return db_site


def update_site(db: Session, site_id: str, site: SiteUpdate) -> Site | None:
    """
    Update an existing site.

    Returns None if site doesn't exist.
    Only updates fields that are provided (not None).
    Crashes if database constraint violated (e.g., duplicate name).
    """
    db_site = get_site(db, site_id)
    if db_site is None:
        return None

    # Only update provided fields
    update_data = site.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_site, field, value)

    db.commit()
    db.refresh(db_site)
    return db_site


def delete_site(db: Session, site_id: str) -> bool:
    """
    Delete a site.

    Returns True if deleted, False if site doesn't exist.
    Cascades to all related deployments, files, etc.
    """
    db_site = get_site(db, site_id)
    if db_site is None:
        return False

    db.delete(db_site)
    db.commit()
    return True


def get_or_create_unknown_site(db: Session, project_id: str) -> Site:
    """
    Get or create the "Unknown Site" for a project.

    This is used when deployments are created without explicit site selection.
    Always returns a valid site, creating it if it doesn't exist.

    Args:
        db: Database session
        project_id: Project ID to get/create unknown site for

    Returns:
        The "Unknown Site" for this project

    Raises:
        IntegrityError if project_id is invalid
    """
    # Try to find existing "Unknown Site" for this project
    query = select(Site).where(
        Site.project_id == project_id, Site.name == "Unknown Site"
    )
    result = db.execute(query)
    existing_site = result.scalar_one_or_none()

    if existing_site:
        return existing_site

    # Create new "Unknown Site"
    site_create = SiteCreate(
        project_id=project_id,
        name="Unknown Site",
        latitude=None,
        longitude=None,
        elevation_m=None,
        habitat_type=None,
        notes="Auto-created site for deployments without explicit site selection",
    )

    return create_site(db, site_create)
