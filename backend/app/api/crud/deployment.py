"""
CRUD operations for Deployment model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import Session, object_session

from app.api.schemas.deployment import DeploymentCreate, DeploymentUpdate
from app.core.logging_config import get_logger
from app.models import (
    Deployment,
    Detection,
    DetectionEmbedding,
    Event,
    EventObservation,
    File,
    Site,
    event_files,
)

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
        paired_cameras=deployment.paired_cameras,
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

    When paired_cameras changes, the deployment's events are regenerated
    at once (the grouping rule changed, see `event_clustering`), carrying
    confirmed counts onto events whose file set stayed the same. The
    project's postprocessing hash is cleared so the "needs reprocessing"
    banner asks for a reprocess, which re-runs smoothing on the new
    grouping. Events of other deployments are untouched.

    When camera_offsets changes (paired cameras), each camera's delta is
    applied to the files of that subfolder only, without touching the
    event bounds: the order of files across cameras changed, so the
    events are regenerated like for a paired_cameras change, with the
    same carry rule and the same hash clearing. A key that disappears is
    a delta back to zero, so unpairing (which sends {}) un-shifts.
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

    paired_changed = (
        "paired_cameras" in update_data
        and update_data["paired_cameras"] is not None
        and update_data["paired_cameras"] != db_deployment.paired_cameras
    )

    # Per-camera deltas over the union of old and new keys, so a removed
    # camera shifts back to zero.
    camera_deltas: dict[str, int] = {}
    if "camera_offsets" in update_data and update_data["camera_offsets"] is not None:
        old_cams = db_deployment.camera_offsets or {}
        new_cams = update_data["camera_offsets"]
        for cam in set(old_cams) | set(new_cams):
            delta = new_cams.get(cam, 0) - old_cams.get(cam, 0)
            if delta:
                camera_deltas[cam] = delta
    if camera_deltas and not db_deployment.folder_path:
        raise ValueError("Camera offsets need a deployment folder")

    for field_name, value in update_data.items():
        setattr(db_deployment, field_name, value)

    if offset_delta_seconds != 0:
        _apply_offset_shift(db, db_deployment, offset_delta_seconds)

    for cam, delta in camera_deltas.items():
        prefix = str(Path(db_deployment.folder_path) / cam) + os.sep
        _apply_offset_shift(
            db, db_deployment, delta, path_prefix=prefix, shift_events=False
        )

    if paired_changed or camera_deltas:
        from app.api.crud.event import generate_events_for_deployment

        generate_events_for_deployment(db, db_deployment)
        db_deployment.project.postprocessing_settings_hash = None

    db.commit()
    db.refresh(db_deployment)
    return db_deployment


def _apply_offset_shift(
    db: Session,
    deployment: Deployment,
    delta_seconds: int,
    *,
    path_prefix: str | None = None,
    shift_events: bool = True,
) -> None:
    """
    Shift every observational datetime in a deployment by
    `delta_seconds`, then recompute the deployment's date range from
    the shifted files.

    `path_prefix` limits the file shift to one camera subfolder of a
    paired deployment (an exact prefix match on `File.file_path`, not
    LIKE, because `_` and `%` in folder names are LIKE wildcards). The
    caller passes `shift_events=False` with it: a partial shift changes
    the order of files across cameras, so the events are regenerated
    afterwards instead of translated.

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

    files_query = db.query(File).filter(File.deployment_id == deployment.id)
    if path_prefix is not None:
        files_query = files_query.filter(
            func.substr(File.file_path, 1, len(path_prefix)) == path_prefix
        )
    files_query.update(
        {"captured_at_local": func.datetime(File.captured_at_local, modifier)},
        synchronize_session=False,
    )

    if shift_events:
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


def purge_deployment_data(db: Session, deployment_ids: Select) -> list[tuple[str, int]]:
    """
    Empty every table under these deployments, leaves first.

    `deployment_ids` is a SELECT of deployment ids, not a list, so the
    whole teardown stays inside the database and nothing is loaded into
    the session.

    **This runs ahead of the cascade, it does not replace it.** Every
    foreign key still declares `ON DELETE CASCADE` and still enforces it,
    so a child table missing from the list below is still removed when
    its parent goes. What the list buys is speed: SQLite runs a foreign
    key action program for every row it deletes, at every level, and
    emptying the leaves first means those programs find nothing left to
    do. Measured on a project of 400,000 files, 800,000 detections and
    400,000 embeddings, against the SQLite the packaged build carries:
    124 s for the plain cascade, 39 s this way, same end state, foreign
    key check clean.

    So forgetting to add a new child table here costs a slower delete,
    never a wrong one. Do not "simplify" it into trusting the cascade
    alone, and do not turn it into per-row ORM deletes either: bulk
    statements are what keep memory flat (see "Deleting analysis data"
    in DEVELOPERS.md).

    Returns `(table, rows)` per stage, in the order they ran, which is
    what the callers log.
    """
    file_ids = select(File.id).where(File.deployment_id.in_(deployment_ids))
    detection_ids = select(Detection.id).where(Detection.file_id.in_(file_ids))
    event_ids = select(Event.id).where(Event.deployment_id.in_(deployment_ids))

    stages = (
        (
            "detection_embeddings",
            delete(DetectionEmbedding).where(
                DetectionEmbedding.detection_id.in_(detection_ids)
            ),
        ),
        ("detections", delete(Detection).where(Detection.file_id.in_(file_ids))),
        (
            "event_observations",
            delete(EventObservation).where(EventObservation.event_id.in_(event_ids)),
        ),
        (
            "event_files",
            delete(event_files).where(event_files.c.event_id.in_(event_ids)),
        ),
        ("events", delete(Event).where(Event.deployment_id.in_(deployment_ids))),
        ("files", delete(File).where(File.deployment_id.in_(deployment_ids))),
    )

    removed: list[tuple[str, int]] = []
    for table, statement in stages:
        result = db.execute(
            statement, execution_options={"synchronize_session": False}
        )
        removed.append((table, result.rowcount))
    return removed


def delete_deployment(db: Session, deployment_id: str) -> bool:
    """
    Delete a deployment.

    Returns True if deleted, False if deployment doesn't exist.

    Cascades to:
    - related files, events, detections (DB, via SQLAlchemy ondelete=CASCADE,
      with `purge_deployment_data` emptying the leaves first for speed)
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

    started = time.time()
    removed = purge_deployment_data(
        db, select(Deployment.id).where(Deployment.id == deployment_id)
    )
    db.delete(db_deployment)
    db.commit()
    logger.info(
        f"Deleted deployment {deployment_id} in {time.time() - started:.1f}s: "
        + ", ".join(f"{rows} {table}" for table, rows in removed if rows)
    )

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
            # Timed: this walks the whole cache tree and can be slow on an
            # external or network drive, which is indistinguishable from a
            # slow DB delete in the log otherwise.
            started = time.perf_counter()
            shutil.rmtree(project_dir)
            logger.info(
                f"Removed deployment artifacts: {project_dir} "
                f"({time.perf_counter() - started:.1f}s)"
            )
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

    from app.api.crud.statistics import _apply_threshold, _get_counting_threshold
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
            select(func.coalesce(func.sum(EventObservation.effective_count), 0))
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
                        (EventObservation.category == "animal", EventObservation.effective_count),
                        else_=0,
                    )
                ),
                0,
            ),
            func.coalesce(
                func.sum(
                    case(
                        (EventObservation.category == "person", EventObservation.effective_count),
                        else_=0,
                    )
                ),
                0,
            ),
            func.coalesce(
                func.sum(
                    case(
                        (EventObservation.category == "vehicle", EventObservation.effective_count),
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
    # LabelTaxonomy.scientific_name when available (matches the label
    # coalesce pattern in the activity-overlap CRUD).
    top_species_rows = db.execute(
        select(
            EventObservation.label,
            LabelTaxonomy.scientific_name,
            LabelTaxonomy.common_name,
            func.sum(EventObservation.effective_count),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .outerjoin(
            LabelTaxonomy, LabelTaxonomy.name == EventObservation.label
        )
        .where(Event.deployment_id == deployment_id)
        .where(EventObservation.category == "animal")
        .where(EventObservation.label.isnot(None))
        .group_by(
            EventObservation.label,
            LabelTaxonomy.scientific_name,
            LabelTaxonomy.common_name,
        )
        .order_by(func.sum(EventObservation.effective_count).desc())
        .limit(5)
    ).all()

    top_species = [
        DeploymentTopSpecies(
            label=row[0],
            scientific_name=row[1],
            common_name=row[2],
            count=int(row[3]),
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
    # A paired-cameras deployment counts as one camera. See
    # `app.api.crud.trap_nights` for detail.
    from app.api.crud.trap_nights import compute_trap_nights_for_deployment

    trap_nights = compute_trap_nights_for_deployment(db, deployment_id)
    rate: float | None = None
    if trap_nights is not None and trap_nights > 0:
        rate = float(observation_count) / trap_nights * 100.0

    # Mean confidences. Detection mean applies the threshold-with-verified
    # filter (per CONVENTIONS.md). Classification mean uses the same
    # filter plus `label_confidence IS NOT NULL` so we only average over
    # detections that were actually classified.
    threshold = _get_counting_threshold(db, project_id)
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
        paired_cameras=deployment.paired_cameras,
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

    The sample is taken in SQL, never off `deployment.files`. That
    relationship is a plain lazy select, so reading it pulls every File
    row of the deployment into memory to hand back ten of them. The
    startup check walks every deployment, which made it load the whole
    files table before the first stat() call: seconds to minutes on a
    large project, during which the API still serves the previous
    session's folder_status. Do not "simplify" this back into a list
    comprehension over the relationship.

    Deliberately not fixed by making the relationship `lazy="dynamic"`:
    it carries `passive_deletes=True` and the cascade rules described in
    DEVELOPERS.md under "Deleting analysis data". Taking the session off
    the instance leaves the delete path untouched.
    """
    if not deployment.folder_path:
        return []

    db = object_session(deployment)
    if db is None:
        raise RuntimeError(
            f"Deployment {deployment.id} is detached from its session; "
            f"cannot sample files for verification"
        )

    old_folder = Path(deployment.folder_path)
    # Files with a size come first: the identity check needs one to tell
    # the real folder from a lookalike. Rows without a size still make
    # the sample when there are not enough with one, which is what the
    # old two-pass list comprehension did. `id` keeps the sample stable
    # between the verify and the relink of the same folder.
    sample = (
        db.execute(
            select(File)
            .where(File.deployment_id == deployment.id)
            .order_by(File.size_bytes.is_(None), File.id)
            .limit(_VERIFY_SAMPLE_SIZE)
        )
        .scalars()
        .all()
    )

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
        # `Path.exists()` swallows only ENOENT / ENOTDIR / EBADF / ELOOP.
        # EACCES and EIO propagate, so on an unreadable folder or a failing
        # drive this used to throw all the way out of the request: a 500 on
        # /check-folder, and at startup one bad folder aborted the whole
        # loop so NO deployment's status was refreshed. Report it the same
        # way the size check below reports a failed stat.
        try:
            present = expected_path.exists()
        except OSError as e:
            mismatches.append(f"Cannot check {expected_path}: {e}")
            continue
        if not present:
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


def check_all_deployment_folders(
    db: Session, project_id: str | None = None
) -> dict[str, int]:
    """
    Re-verify every deployment's folder and update statuses in bulk.

    Used at app startup (no `project_id`) so the folder_status column
    reflects the current filesystem state, and per project when the
    Deployments page opens. Skips deployments with folder_path=None. Also
    migrates any legacy "missing" status values to "needs_relink".

    The per-project call is what keeps the recovery page honest in both
    directions. Nothing else re-checks a deployment already marked
    needs_relink, so a folder the user reconnected outside the app (an
    external drive plugged back in) kept being reported as missing while
    its pictures were plainly on screen. It also corrects the count:
    detection is driven by images failing to load, and a browser-cached
    thumbnail never fails, so deployments the user browsed recently could
    be broken without anything noticing.

    Returns counts of checked/changed/skipped for logging.
    """
    query = select(Deployment)
    if project_id is not None:
        query = query.where(Deployment.project_id == project_id)
    deployments = db.execute(query).scalars().all()
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

        # One unreadable folder must never cost the other deployments
        # their check. `verify_deployment_folder` handles the errors it can
        # attribute to a file; anything else (an unreadable folder_path
        # itself, a drive that vanished mid-loop) lands here, marks that
        # one deployment as needing a relink, and the loop carries on.
        try:
            result = verify_deployment_folder(dep, dep.folder_path)
            new_status = result.status
        except OSError as e:
            logger.warning(
                f"Could not check folder for deployment {dep.id} "
                f"({dep.folder_path}): {e}"
            )
            new_status = "needs_relink"
        if dep.folder_status != new_status:
            dep.folder_status = new_status
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
        # Log every reason. The refusal is otherwise unactionable: the
        # endpoint answers 200 with the mismatch list, so a rejected relink
        # left no trace at all on the server and the user was told only how
        # many samples failed, never which ones or why.
        logger.warning(
            f"Relink refused for deployment {deployment_id}: "
            f"{len(result.mismatches)} of {result.checked_count} sampled "
            f"file(s) did not match at {new_folder_path}"
        )
        for mismatch in result.mismatches:
            logger.warning(f"  relink mismatch: {mismatch}")
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
