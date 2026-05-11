"""
CRUD operations for Deployment model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas.deployment import DeploymentCreate, DeploymentUpdate
from app.core.logging_config import get_logger
from app.models import Deployment, Event, File, Site

logger = get_logger(__name__)

# Number of files to sample when verifying a deployment folder's identity.
_VERIFY_SAMPLE_SIZE = 10

# Reserved token used inside site_ids URL filters to mean "deployments
# with no site". Frontend emits the literal string "null" alongside
# real site UUIDs; backend translates it to `site_id IS NULL`.
NO_SITE_SENTINEL: str = "null"


def site_ids_filter(site_ids: list[str] | None):
    """
    Turn a list of site_ids (optionally containing NO_SITE_SENTINEL)
    into a SQLAlchemy boolean clause on Deployment.site_id. Returns
    None when there is no filter to apply.
    """
    from sqlalchemy import or_

    if not site_ids:
        return None
    real_ids = [s for s in site_ids if s != NO_SITE_SENTINEL]
    include_null = NO_SITE_SENTINEL in site_ids
    clauses = []
    if real_ids:
        clauses.append(Deployment.site_id.in_(real_ids))
    if include_null:
        clauses.append(Deployment.site_id.is_(None))
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return or_(*clauses)


def get_deployments(
    db: Session,
    site_id: str | None = None,
    project_id: str | None = None,
) -> list[Deployment]:
    """
    Get all deployments, optionally filtered by site_id or project_id.

    Returns empty list if no deployments exist.
    """
    query = select(Deployment).order_by(Deployment.created_at_utc.desc())
    if site_id:
        query = query.where(Deployment.site_id == site_id)
    if project_id:
        query = query.where(Deployment.project_id == project_id)
    result = db.execute(query)
    return list(result.scalars().all())


def get_deployment(db: Session, deployment_id: str) -> Deployment | None:
    """
    Get deployment by ID.

    Returns None if deployment doesn't exist.
    """
    result = db.execute(select(Deployment).where(Deployment.id == deployment_id))
    return result.scalar_one_or_none()


def create_deployment(db: Session, deployment: DeploymentCreate) -> Deployment:
    """
    Create a new deployment.

    Crashes if database constraint violated (e.g., invalid project_id
    or site_id). site_id may be None for deployment-agnostic batches.
    """
    db_deployment = Deployment(
        project_id=deployment.project_id,
        site_id=deployment.site_id,
        folder_path=deployment.folder_path,
        start_date_local=deployment.start_date_local,
        end_date_local=deployment.end_date_local,
        camera_model=deployment.camera_model,
        camera_serial=deployment.camera_serial,
        notes=deployment.notes,
        tags=deployment.tags,
    )
    db.add(db_deployment)
    db.commit()
    db.refresh(db_deployment)
    return db_deployment


def update_deployment(
    db: Session, deployment_id: str, deployment: DeploymentUpdate
) -> Deployment | None:
    """
    Update an existing deployment.

    Returns None if deployment doesn't exist.
    Only updates fields that are provided (not None).
    Crashes if database constraint violated.

    When folder_path is updated (re-linking), also updates last_validated_at.

    When datetime_offset_seconds changes, the delta is applied in-place
    to every File.captured_at_local and every Event.event_start_local /
    event_end_local in the deployment, and the deployment's
    start_date_local / end_date_local are recomputed from the new file
    range. Without this cascade, the slideout's first / last dates,
    dashboard charts, and event listings would stay frozen at the
    pre-edit offset because those views read off the baked
    File.captured_at_local rather than recomputing on every render.
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return None

    # Only update provided fields
    update_data = deployment.model_dump(exclude_unset=True)

    # If folder_path is being updated, update validation timestamp
    if "folder_path" in update_data and update_data["folder_path"] is not None:
        db_deployment.folder_status = "valid"
        db_deployment.last_validated_at_utc = datetime.now(UTC)

    # Capture the offset change (if any) before mutating the row.
    # None and 0 are semantically equivalent ("no offset"); coerce to 0
    # so the delta arithmetic doesn't trip on the nullable column.
    offset_delta_seconds = 0
    if "datetime_offset_seconds" in update_data:
        old_offset = db_deployment.datetime_offset_seconds or 0
        new_offset = update_data["datetime_offset_seconds"] or 0
        offset_delta_seconds = new_offset - old_offset

    for field_name, value in update_data.items():
        setattr(db_deployment, field_name, value)

    if offset_delta_seconds != 0:
        _apply_offset_shift(db, db_deployment, offset_delta_seconds)

    db.commit()
    db.refresh(db_deployment)
    return db_deployment


def _apply_offset_shift(
    db: Session, deployment: Deployment, delta_seconds: int
) -> None:
    """
    Shift every observational datetime in a deployment by
    `delta_seconds`, then recompute the deployment's date range from
    the shifted files.

    Touches `File.captured_at_local`, `Event.event_start_local`,
    `Event.event_end_local`, and `Deployment.start_date_local /
    end_date_local`. Audit datetimes (`*_utc`) are left alone; they
    record server actions, not camera observations.

    Uses SQLite's `datetime(col, '+N seconds')` modifier so the shift
    runs as a single bulk UPDATE per table without round-tripping rows
    through Python. Per CONVENTIONS.md datetime rules, observational
    timestamps stay naive in the camera's local clock; the offset is
    just an integer second translation.
    """
    from sqlalchemy import func

    from app.models import Event, File

    # SQLite's datetime() takes signed numeric strings: '+30 seconds',
    # '-3600 seconds'. Build the modifier that way.
    sign = "+" if delta_seconds > 0 else ""
    modifier = f"{sign}{delta_seconds} seconds"

    db.query(File).filter(File.deployment_id == deployment.id).update(
        {"captured_at_local": func.datetime(File.captured_at_local, modifier)},
        synchronize_session=False,
    )

    db.query(Event).filter(Event.deployment_id == deployment.id).update(
        {
            "event_start_local": func.datetime(
                Event.event_start_local, modifier
            ),
            "event_end_local": func.datetime(Event.event_end_local, modifier),
        },
        synchronize_session=False,
    )

    # Refresh the deployment's date window from the post-shift file
    # range. start_date_local / end_date_local are calendar dates, not
    # datetimes, so we extract just the date portion.
    new_min, new_max = db.execute(
        select(
            func.min(File.captured_at_local), func.max(File.captured_at_local)
        ).where(File.deployment_id == deployment.id)
    ).one()

    if new_min is not None:
        deployment.start_date_local = (
            new_min.date() if isinstance(new_min, datetime) else new_min
        )
        deployment.end_date_local = (
            new_max.date() if isinstance(new_max, datetime) else new_max
        )


def delete_deployment(db: Session, deployment_id: str) -> bool:
    """
    Delete a deployment.

    Returns True if deleted, False if deployment doesn't exist.

    Cascades to:
    - related files, events, detections (DB, via SQLAlchemy ondelete=CASCADE)
    - on-disk ML artifacts in `<folder_path>/.addaxai/projects/<project_id>/`
      (via _delete_deployment_artifacts; best-effort, never blocks DB delete)
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return False

    # Capture the path info BEFORE the row is deleted; we still need it
    # to clean up the on-disk artifacts after the DB cascade fires.
    folder_path = db_deployment.folder_path
    project_id = db_deployment.project_id

    db.delete(db_deployment)
    db.commit()

    if folder_path:
        _delete_deployment_artifacts(folder_path, project_id)
    return True


def _delete_deployment_artifacts(folder_path: str, project_id: str) -> None:
    """
    Remove the project-scoped .addaxai folder for a deleted deployment.

    Best-effort: missing paths and OS errors are logged and swallowed so
    that DB deletes never roll back because of a stale filesystem state
    (e.g. an unmounted external drive). Cleans up empty parent dirs
    (`.addaxai/projects/`, `.addaxai/`) so the folder is left as the
    user originally placed it on disk.
    """
    project_dir = Path(folder_path) / ".addaxai" / "projects" / project_id
    projects_dir = project_dir.parent
    addaxai_dir = projects_dir.parent

    if project_dir.exists():
        try:
            shutil.rmtree(project_dir)
            logger.info(f"Removed deployment artifacts: {project_dir}")
        except OSError as e:
            logger.warning(
                f"Failed to remove deployment artifacts at {project_dir}: {e}"
            )
            return

    # Roll up empty parents so the .addaxai marker disappears entirely
    # when the last project is gone. Only remove if empty; never recurse.
    for empty_candidate in (projects_dir, addaxai_dir):
        try:
            if empty_candidate.exists() and not any(empty_candidate.iterdir()):
                empty_candidate.rmdir()
        except OSError as e:
            logger.warning(f"Failed to clean up empty {empty_candidate}: {e}")
            return


def get_deployment_stats(db: Session, deployment_id: str) -> dict[str, int] | None:
    """
    Get statistics for a deployment.

    Returns dict with counts, or None if deployment doesn't exist.
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return None

    # Count files
    file_count = (
        db.scalar(
            select(func.count(File.id)).where(File.deployment_id == deployment_id)
        )
        or 0
    )

    # Count events
    event_count = (
        db.scalar(
            select(func.count(Event.id)).where(Event.deployment_id == deployment_id)
        )
        or 0
    )

    # TODO: Count detections (model not fully implemented yet)
    detection_count = 0

    return {
        "file_count": file_count,
        "event_count": event_count,
        "detection_count": detection_count,
    }


def get_bulk_deployment_stats(
    db: Session, project_id: str
) -> dict[str, dict[str, int]]:
    """
    Get file/event/detection counts for all deployments in a project.

    Single round-trip per count type. Returns
    `{deployment_id: {file_count, event_count, detection_count}}`.
    """
    from app.models import Detection

    # All deployment IDs in project
    dep_ids_subq = (
        select(Deployment.id)
        .where(Deployment.project_id == project_id)
        .subquery()
    )

    file_counts = dict(
        db.execute(
            select(File.deployment_id, func.count(File.id))
            .where(File.deployment_id.in_(select(dep_ids_subq)))
            .group_by(File.deployment_id)
        ).all()
    )

    event_counts = dict(
        db.execute(
            select(Event.deployment_id, func.count(Event.id))
            .where(Event.deployment_id.in_(select(dep_ids_subq)))
            .group_by(Event.deployment_id)
        ).all()
    )

    detection_counts = dict(
        db.execute(
            select(File.deployment_id, func.count(Detection.id))
            .join(File, Detection.file_id == File.id)
            .where(File.deployment_id.in_(select(dep_ids_subq)))
            .group_by(File.deployment_id)
        ).all()
    )

    all_ids = set(file_counts) | set(event_counts) | set(detection_counts)
    return {
        dep_id: {
            "file_count": file_counts.get(dep_id, 0),
            "event_count": event_counts.get(dep_id, 0),
            "detection_count": detection_counts.get(dep_id, 0),
        }
        for dep_id in all_ids
    }


def get_deployment_info(db: Session, deployment_id: str):
    """
    Build the investigation-level payload for the Deployments → Info sheet.

    Returns `None` when the deployment does not exist so the router can
    map it to a 404. Applies the project's detection threshold with the
    verified override when averaging confidences.
    """
    from sqlalchemy import case

    from app.api.crud.statistics import _apply_threshold, _get_detection_threshold
    from app.api.schemas.deployment import (
        DeploymentDetectionCategories,
        DeploymentFileCounts,
        DeploymentInfoResponse,
        DeploymentTopSpecies,
        DeploymentVerification,
    )
    from app.models import Detection, EventObservation, LabelTaxonomy

    deployment = get_deployment(db, deployment_id)
    if deployment is None:
        return None

    # Site is optional. project_id comes directly from the deployment.
    site_id = deployment.site_id
    project_id = deployment.project_id
    site = db.get(Site, site_id) if site_id is not None else None
    site_name = site.name if site is not None else None

    # File counts split by file_type + verification + total size in one
    # grouped query.
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
        .where(File.deployment_id == deployment_id)
    ).one()
    total_files, images, videos, verified_files, total_size_bytes = file_row

    event_count = (
        db.scalar(
            select(func.count(Event.id)).where(
                Event.deployment_id == deployment_id
            )
        )
        or 0
    )

    # Sum of MaxN across event observations belonging to this deployment.
    observation_count = (
        db.scalar(
            select(func.coalesce(func.sum(EventObservation.max_n), 0))
            .select_from(EventObservation)
            .join(Event, Event.id == EventObservation.event_id)
            .where(Event.deployment_id == deployment_id)
        )
        or 0
    )

    # Detection categories (animal / person / vehicle) via MaxN sums
    # grouped by EventObservation.category. Matches the dashboard's
    # `get_detection_categories` so the numbers line up.
    cat_row = db.execute(
        select(
            func.coalesce(
                func.sum(
                    case(
                        (EventObservation.category == "animal", EventObservation.max_n),
                        else_=0,
                    )
                ),
                0,
            ),
            func.coalesce(
                func.sum(
                    case(
                        (EventObservation.category == "person", EventObservation.max_n),
                        else_=0,
                    )
                ),
                0,
            ),
            func.coalesce(
                func.sum(
                    case(
                        (EventObservation.category == "vehicle", EventObservation.max_n),
                        else_=0,
                    )
                ),
                0,
            ),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .where(Event.deployment_id == deployment_id)
    ).one()
    animal_count, person_count, vehicle_count = (
        int(cat_row[0]),
        int(cat_row[1]),
        int(cat_row[2]),
    )

    # Empty = files whose all detections were skipped (observation_type
    # == "blank"). Scoped to this deployment.
    empty_count = (
        db.scalar(
            select(func.count(File.id))
            .where(File.deployment_id == deployment_id)
            .where(File.observation_type == "blank")
        )
        or 0
    )

    # Top 5 species by MaxN sum. Only counts animal observations; people
    # / vehicles have their own row in the categories block already. Uses
    # LabelTaxonomy.display_name when available (matches the label
    # coalesce pattern in the activity-overlap CRUD).
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
        .where(Event.deployment_id == deployment_id)
        .where(EventObservation.category == "animal")
        .where(EventObservation.label.isnot(None))
        .group_by(EventObservation.label, LabelTaxonomy.display_name)
        .order_by(func.sum(EventObservation.max_n).desc())
        .limit(5)
    ).all()

    top_species = [
        DeploymentTopSpecies(
            label=row[0],
            display_name=row[1],
            count=int(row[2]),
        )
        for row in top_species_rows
    ]

    # First / last capture timestamps across files in this deployment.
    timestamps_row = db.execute(
        select(
            func.min(File.captured_at_local),
            func.max(File.captured_at_local),
        ).where(File.deployment_id == deployment_id)
    ).one()
    first_captured_at_local, last_captured_at_local = timestamps_row

    # Trap nights is folder-aware: each SD-card-folder contributes its own
    # (max - min) + 1 day span, summed. For a clean single-folder
    # deployment this equals the old `(end - start) + 1` formula; for a
    # mixed backlog it correctly excludes the offline gaps between cards.
    # See `app.api.crud.trap_nights` for detail.
    from app.api.crud.trap_nights import compute_trap_nights_for_deployment

    trap_nights = compute_trap_nights_for_deployment(db, deployment_id)
    rate: float | None = None
    if trap_nights is not None and trap_nights > 0:
        rate = float(observation_count) / trap_nights * 100.0

    # Mean confidences. Detection mean applies the threshold-with-verified
    # filter (per CONVENTIONS.md). Classification mean uses the same
    # filter plus `label_confidence IS NOT NULL` so we only average over
    # detections that were actually classified.
    threshold = _get_detection_threshold(db, project_id)
    detection_q = _apply_threshold(
        select(func.avg(Detection.confidence))
        .join(File, Detection.file_id == File.id)
        .where(File.deployment_id == deployment_id),
        threshold,
    )
    mean_det = db.scalar(detection_q)
    mean_detection_confidence = (
        float(mean_det) if mean_det is not None else None
    )

    classification_q = _apply_threshold(
        select(func.avg(Detection.label_confidence))
        .join(File, Detection.file_id == File.id)
        .where(File.deployment_id == deployment_id)
        .where(Detection.label_confidence.isnot(None)),
        threshold,
    )
    mean_cls = db.scalar(classification_q)
    mean_classification_confidence = (
        float(mean_cls) if mean_cls is not None else None
    )

    return DeploymentInfoResponse(
        deployment_id=deployment.id,
        folder_path=deployment.folder_path,
        site_id=site_id,
        site_name=site_name,
        start_date_local=deployment.start_date_local,
        end_date_local=deployment.end_date_local,
        files=DeploymentFileCounts(
            total=int(total_files), images=int(images), videos=int(videos)
        ),
        total_size_bytes=int(total_size_bytes),
        verification=DeploymentVerification(
            verified=int(verified_files), total=int(total_files)
        ),
        event_count=int(event_count),
        observation_count=int(observation_count),
        detection_categories=DeploymentDetectionCategories(
            animal=animal_count,
            person=person_count,
            vehicle=vehicle_count,
            empty=int(empty_count),
        ),
        top_species=top_species,
        trap_nights=trap_nights,
        observation_rate_per_100_trap_nights=rate,
        mean_detection_confidence=mean_detection_confidence,
        mean_classification_confidence=mean_classification_confidence,
        first_captured_at_local=first_captured_at_local,
        last_captured_at_local=last_captured_at_local,
        warnings=deployment.warnings,
    )


@dataclass
class VerifyResult:
    """Outcome of verifying a deployment folder against its file records."""

    status: str  # "valid" or "needs_relink"
    checked_count: int
    mismatches: list[str] = field(default_factory=list)


@dataclass
class RelinkResult:
    """Outcome of an attempted relink."""

    success: bool
    files_rewritten: int = 0
    verify_result: VerifyResult | None = None


def _sample_files_for_verification(
    deployment: Deployment, folder_to_verify: Path
) -> list[tuple[File, Path]]:
    """
    Pick up to _VERIFY_SAMPLE_SIZE files from a deployment and compute the
    path where each should live if folder_to_verify were the deployment's
    root. Prefers files with non-null size_bytes (the identity check needs
    a size to compare against).

    Returns a list of (file_record, expected_path) tuples. Files whose
    file_path is not under the current deployment folder are skipped
    (defensive).
    """
    if not deployment.folder_path:
        return []

    old_folder = Path(deployment.folder_path)
    candidates = [f for f in deployment.files if f.size_bytes is not None]
    if len(candidates) < _VERIFY_SAMPLE_SIZE:
        # Fall back to any files if we don't have enough with sizes
        seen_ids = {f.id for f in candidates}
        candidates.extend(f for f in deployment.files if f.id not in seen_ids)

    sample = candidates[:_VERIFY_SAMPLE_SIZE]
    pairs: list[tuple[File, Path]] = []
    for f in sample:
        try:
            relative = Path(f.file_path).relative_to(old_folder)
        except ValueError:
            # file_path is not under the current deployment folder,
            # can't verify or relink it — skip.
            logger.warning(
                f"File {f.id} path {f.file_path} is not under "
                f"deployment folder {old_folder}; skipping"
            )
            continue
        pairs.append((f, folder_to_verify / relative))
    return pairs


def verify_deployment_folder(
    deployment: Deployment, folder_path: str | None
) -> VerifyResult:
    """
    Verify that a folder_path holds the expected content for a deployment.

    Checks, in order:
    1. folder_path is a non-empty string
    2. folder_path points to an existing directory
    3. A sample of files from the deployment exist at their expected
       locations under folder_path AND have matching sizes (the identity
       check that distinguishes the real folder from a lookalike).

    Returns a VerifyResult. status is "valid" only if all sampled files
    pass both the existence and size checks; otherwise "needs_relink".
    """
    if not folder_path:
        return VerifyResult(status="needs_relink", checked_count=0)

    if not os.path.isdir(folder_path):
        return VerifyResult(
            status="needs_relink",
            checked_count=0,
            mismatches=[f"Folder not found: {folder_path}"],
        )

    folder = Path(folder_path)
    samples = _sample_files_for_verification(deployment, folder)

    # No File records at all — nothing to verify against. Trust the
    # directory-exists check and mark valid.
    if not samples:
        return VerifyResult(status="valid", checked_count=0)

    mismatches: list[str] = []
    for file_record, expected_path in samples:
        if not expected_path.exists():
            mismatches.append(f"Missing: {expected_path}")
            continue
        if file_record.size_bytes is not None:
            try:
                actual_size = expected_path.stat().st_size
            except OSError as e:
                mismatches.append(f"Cannot stat {expected_path}: {e}")
                continue
            if actual_size != file_record.size_bytes:
                mismatches.append(
                    f"Size mismatch at {expected_path}: "
                    f"expected {file_record.size_bytes}, got {actual_size}"
                )

    status = "valid" if not mismatches else "needs_relink"
    return VerifyResult(
        status=status, checked_count=len(samples), mismatches=mismatches
    )


def check_deployment_folder(db: Session, deployment_id: str) -> Deployment | None:
    """
    Re-verify a single deployment's folder_path and update its status.

    Runs the full identity check (existence + size match) against the
    currently stored folder_path. Updates last_validated_at regardless.
    Deployments with folder_path=None are returned unchanged.
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return None

    if db_deployment.folder_path:
        result = verify_deployment_folder(db_deployment, db_deployment.folder_path)
        db_deployment.folder_status = result.status
    db_deployment.last_validated_at_utc = datetime.now(UTC)
    db.commit()
    db.refresh(db_deployment)
    return db_deployment


def check_all_deployment_folders(db: Session) -> dict[str, int]:
    """
    Re-verify every deployment's folder and update statuses in bulk.

    Used at app startup so the folder_status column reflects the current
    filesystem state. Skips deployments with folder_path=None. Also
    migrates any legacy "missing" status values to "needs_relink".

    Returns counts of checked/changed/skipped for logging.
    """
    deployments = db.execute(select(Deployment)).scalars().all()
    checked = 0
    changed = 0
    skipped = 0
    now = datetime.now(UTC)

    for dep in deployments:
        # Legacy data migration: collapse the old "missing" status into
        # "needs_relink" before re-verifying.
        if dep.folder_status == "missing":
            dep.folder_status = "needs_relink"
            changed += 1

        if not dep.folder_path:
            skipped += 1
            continue

        result = verify_deployment_folder(dep, dep.folder_path)
        if dep.folder_status != result.status:
            dep.folder_status = result.status
            changed += 1
        dep.last_validated_at_utc = now
        checked += 1

    if checked > 0 or changed > 0:
        db.commit()

    return {"checked": checked, "changed": changed, "skipped": skipped}


def relink_deployment(
    db: Session, deployment_id: str, new_folder_path: str
) -> RelinkResult:
    """
    Point a deployment at a new folder on disk.

    Verifies the new folder holds the expected files (sample exists +
    size match) and, if so, rewrites every File.file_path and
    File.best_frame_path for the deployment to point at the new
    location. Atomic: either all files are rewritten or none are.

    Returns:
        RelinkResult with success=False and verify_result if verification
        fails (no DB writes in this case). Otherwise success=True with
        the number of file records rewritten.
    """
    deployment = get_deployment(db, deployment_id)
    if deployment is None:
        return RelinkResult(success=False)

    if not deployment.folder_path:
        return RelinkResult(
            success=False,
            verify_result=VerifyResult(
                status="needs_relink",
                checked_count=0,
                mismatches=["Deployment has no folder_path to relink from"],
            ),
        )

    # Preflight: make sure the new location actually holds this deployment's
    # files. Uses the same sampling logic as the startup/modal check.
    result = verify_deployment_folder(deployment, new_folder_path)
    if result.status != "valid":
        return RelinkResult(success=False, verify_result=result)

    old_folder = Path(deployment.folder_path)
    new_folder = Path(new_folder_path)
    files_rewritten = 0

    def _rewrite(old: str | None) -> str | None:
        """Strip old_folder prefix and prepend new_folder. Returns None on miss."""
        if not old:
            return None
        try:
            relative = Path(old).relative_to(old_folder)
        except ValueError:
            logger.warning(
                f"Path {old} is not under deployment folder {old_folder}; "
                f"leaving unchanged during relink"
            )
            return None
        return str(new_folder / relative)

    for f in deployment.files:
        new_file_path = _rewrite(f.file_path)
        if new_file_path is not None:
            f.file_path = new_file_path
            files_rewritten += 1
        new_frame_path = _rewrite(f.best_frame_path)
        if new_frame_path is not None:
            f.best_frame_path = new_frame_path

    deployment.folder_path = str(new_folder)
    deployment.folder_status = "valid"
    deployment.last_validated_at_utc = datetime.now(UTC)

    db.commit()
    db.refresh(deployment)

    return RelinkResult(
        success=True,
        files_rewritten=files_rewritten,
        verify_result=result,
    )
