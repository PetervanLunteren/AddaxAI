"""
CRUD operations for Site model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas.site import SiteCreate, SiteUpdate
from app.models import Deployment, Site


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


def get_sites_with_stats(
    db: Session, project_id: str
) -> list[dict]:
    """
    Get all sites for a project with deployment counts.

    Returns list of dicts with site fields + deployment_count.
    """
    dep_count = func.count(Deployment.id).label("deployment_count")
    rows = db.execute(
        select(Site, dep_count)
        .outerjoin(Deployment, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
        .group_by(Site.id)
        .order_by(Site.created_at.desc())
    ).all()

    results = []
    for site, count in rows:
        site_dict = {
            "id": site.id,
            "project_id": site.project_id,
            "name": site.name,
            "latitude": site.latitude,
            "longitude": site.longitude,
            "elevation_m": site.elevation_m,
            "habitat_type": site.habitat_type,
            "notes": site.notes,
            "created_at": site.created_at,
            "deployment_count": count,
        }
        results.append(site_dict)
    return results
