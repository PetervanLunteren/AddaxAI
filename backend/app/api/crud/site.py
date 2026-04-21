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
    query = select(Site).order_by(Site.created_at_utc.desc())
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
        tags=site.tags,
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


def get_site_info(db: Session, site_id: str):
    """
    Build the investigation-level payload for the Sites → Info sheet.

    Aggregates across every deployment at the site. Returns `None` when
    the site does not exist so the router can map to a 404. Trap nights
    is the sum of each deployment's folder-aware count (see
    `app.api.crud.trap_nights`); `None` when the total is 0 so the UI
    can render "n/a" instead of a misleading zero rate.
    """
    from sqlalchemy import case

    from app.api.schemas.site import (
        SiteDetectionCategories,
        SiteFileCounts,
        SiteInfoResponse,
        SiteTopSpecies,
        SiteVerification,
    )
    from app.models import Event, EventObservation, File, LabelTaxonomy

    site = get_site(db, site_id)
    if site is None:
        return None

    # Deployment ids scoped to this site. Most aggregate queries filter
    # files / events by `deployment_id IN (...)`. Using a list keeps the
    # downstream SQL simple.
    deployment_ids = [
        row[0]
        for row in db.execute(
            select(Deployment.id).where(Deployment.site_id == site_id)
        ).all()
    ]
    deployment_count = len(deployment_ids)

    # File counts split by file_type + verification + total size. One
    # grouped query. Returns zeros when the site has no files.
    if deployment_ids:
        file_row = db.execute(
            select(
                func.count(File.id),
                func.coalesce(
                    func.sum(case((File.file_type == "image", 1), else_=0)), 0
                ),
                func.coalesce(
                    func.sum(case((File.file_type == "video", 1), else_=0)), 0
                ),
                func.coalesce(
                    func.sum(case((File.verified.is_(True), 1), else_=0)), 0
                ),
                func.coalesce(func.sum(File.size_bytes), 0),
            )
            .select_from(File)
            .where(File.deployment_id.in_(deployment_ids))
        ).one()
        total_files, images, videos, verified_files, total_size_bytes = file_row
    else:
        total_files = images = videos = verified_files = total_size_bytes = 0

    event_count = 0
    observation_count = 0
    animal_count = person_count = vehicle_count = 0
    empty_count = 0
    top_species: list[SiteTopSpecies] = []
    first_captured_at_local = None
    last_captured_at_local = None

    if deployment_ids:
        event_count = (
            db.scalar(
                select(func.count(Event.id)).where(
                    Event.deployment_id.in_(deployment_ids)
                )
            )
            or 0
        )

        observation_count = (
            db.scalar(
                select(func.coalesce(func.sum(EventObservation.max_n), 0))
                .select_from(EventObservation)
                .join(Event, Event.id == EventObservation.event_id)
                .where(Event.deployment_id.in_(deployment_ids))
            )
            or 0
        )

        cat_row = db.execute(
            select(
                func.coalesce(
                    func.sum(
                        case(
                            (
                                EventObservation.category == "animal",
                                EventObservation.max_n,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                EventObservation.category == "person",
                                EventObservation.max_n,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                EventObservation.category == "vehicle",
                                EventObservation.max_n,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
            )
            .select_from(EventObservation)
            .join(Event, Event.id == EventObservation.event_id)
            .where(Event.deployment_id.in_(deployment_ids))
        ).one()
        animal_count = int(cat_row[0])
        person_count = int(cat_row[1])
        vehicle_count = int(cat_row[2])

        empty_count = (
            db.scalar(
                select(func.count(File.id))
                .where(File.deployment_id.in_(deployment_ids))
                .where(File.observation_type == "blank")
            )
            or 0
        )

        top_species_rows = db.execute(
            select(
                EventObservation.label,
                LabelTaxonomy.display_name,
                func.sum(EventObservation.max_n),
            )
            .select_from(EventObservation)
            .join(Event, Event.id == EventObservation.event_id)
            .outerjoin(
                LabelTaxonomy, LabelTaxonomy.name == EventObservation.label
            )
            .where(Event.deployment_id.in_(deployment_ids))
            .where(EventObservation.category == "animal")
            .where(EventObservation.label.isnot(None))
            .group_by(EventObservation.label, LabelTaxonomy.display_name)
            .order_by(func.sum(EventObservation.max_n).desc())
            .limit(5)
        ).all()
        top_species = [
            SiteTopSpecies(
                label=row[0], display_name=row[1], count=int(row[2])
            )
            for row in top_species_rows
        ]

        timestamps_row = db.execute(
            select(
                func.min(File.captured_at_local),
                func.max(File.captured_at_local),
            ).where(File.deployment_id.in_(deployment_ids))
        ).one()
        first_captured_at_local, last_captured_at_local = timestamps_row

    # Trap nights: sum of each deployment's folder-aware trap-nights count.
    # For a clean single-folder deployment this equals (end - start + 1);
    # for a mixed backlog it sums each folder's own span so the offline
    # gaps between SD cards don't inflate the denominator. `None` when
    # the whole sum ends up 0 (e.g. empty site).
    from app.api.crud.trap_nights import compute_trap_nights_for_deployments

    trap_nights: int | None = None
    if deployment_ids:
        per_dep = compute_trap_nights_for_deployments(db, deployment_ids)
        total_nights = sum(per_dep.values())
        trap_nights = total_nights if total_nights > 0 else None

    rate: float | None = None
    if trap_nights is not None and trap_nights > 0:
        rate = float(observation_count) / trap_nights * 100.0

    return SiteInfoResponse(
        site_id=site.id,
        name=site.name,
        latitude=site.latitude,
        longitude=site.longitude,
        elevation_m=site.elevation_m,
        habitat_type=site.habitat_type,
        notes=site.notes,
        tags=site.tags or {},
        deployment_count=deployment_count,
        files=SiteFileCounts(
            total=int(total_files), images=int(images), videos=int(videos)
        ),
        total_size_bytes=int(total_size_bytes),
        verification=SiteVerification(
            verified=int(verified_files), total=int(total_files)
        ),
        event_count=int(event_count),
        observation_count=int(observation_count),
        detection_categories=SiteDetectionCategories(
            animal=animal_count,
            person=person_count,
            vehicle=vehicle_count,
            empty=int(empty_count),
        ),
        top_species=top_species,
        trap_nights=trap_nights,
        observation_rate_per_100_trap_nights=rate,
        first_captured_at_local=first_captured_at_local,
        last_captured_at_local=last_captured_at_local,
    )


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
        .order_by(Site.created_at_utc.desc())
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
            "tags": site.tags,
            "created_at_utc": site.created_at_utc,
            "deployment_count": count,
        }
        results.append(site_dict)
    return results
