"""
CRUD operations for DeploymentQueue model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.deployment_queue import DeploymentQueueCreate
from app.models import DeploymentQueue


def get_queue_entries(
    db: Session, project_id: str, status: str | None = None
) -> list[DeploymentQueue]:
    """
    Get all queue entries for a project.

    Optionally filter by status.
    Returns empty list if no entries exist.
    """
    query = select(DeploymentQueue).where(DeploymentQueue.project_id == project_id)

    if status:
        query = query.where(DeploymentQueue.status == status)

    query = query.order_by(DeploymentQueue.created_at_utc.asc())
    result = db.execute(query)
    return list(result.scalars().all())


def get_queue_entry(db: Session, entry_id: str) -> DeploymentQueue | None:
    """
    Get queue entry by ID.

    Returns None if entry doesn't exist.
    """
    result = db.execute(select(DeploymentQueue).where(DeploymentQueue.id == entry_id))
    return result.scalar_one_or_none()


def _new_queue_entry(entry: DeploymentQueueCreate) -> DeploymentQueue:
    """Build the row. Shared so the single and bulk paths cannot drift apart
    when a column is added."""
    return DeploymentQueue(
        project_id=entry.project_id,
        folder_path=entry.folder_path,
        site_id=entry.site_id,
        video_count=entry.video_count,
        image_count=entry.image_count,
        datetime_offset_seconds=entry.datetime_offset_seconds,
        use_file_mtime_fallback=entry.use_file_mtime_fallback,
        paired_cameras=entry.paired_cameras,
        notes=entry.notes,
        tags=entry.tags,
        status="pending",
    )


def create_queue_entry(db: Session, entry: DeploymentQueueCreate) -> DeploymentQueue:
    """
    Create a new queue entry.

    Crashes if database constraint violated.
    This is intentional - we want to surface errors immediately.

    Note: Model configuration is now project-scoped (not per-deployment).
    """
    db_entry = _new_queue_entry(entry)
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry


def create_queue_entries_bulk(
    db: Session, entries: list[DeploymentQueueCreate]
) -> list[DeploymentQueue]:
    """Create many queue entries in one transaction.

    Used by the CSV import, which is all or nothing: one commit means either
    every row lands or none does.
    """
    db_entries = [_new_queue_entry(entry) for entry in entries]
    db.add_all(db_entries)
    db.commit()
    return db_entries


def update_queue_counts(
    db: Session, entry_id: str, video_count: int, image_count: int
) -> DeploymentQueue | None:
    """
    Update file counts for a queue entry.

    Used after folder scanning to set video_count and image_count.
    Returns None if entry doesn't exist.
    """
    db_entry = get_queue_entry(db, entry_id)
    if db_entry is None:
        return None

    db_entry.video_count = video_count
    db_entry.image_count = image_count

    db.commit()
    db.refresh(db_entry)
    return db_entry


def update_queue_warnings(
    db: Session, entry_id: str, warnings: str | None
) -> DeploymentQueue | None:
    """
    Record non-fatal warnings on an in-flight queue entry without
    changing its status. Used mid-run, e.g. after the JSON loader
    reports files skipped due to missing capture timestamps.
    """
    db_entry = get_queue_entry(db, entry_id)
    if db_entry is None:
        return None

    db_entry.warnings = warnings
    db.commit()
    db.refresh(db_entry)
    return db_entry


def update_queue_status(
    db: Session,
    entry_id: str,
    status: str,
    error: str | None = None,
    deployment_id: str | None = None,
    warnings: str | None = None,
) -> DeploymentQueue | None:
    """
    Update queue entry status.

    Returns None if entry doesn't exist. `warnings` carries non-fatal
    ingest messages (e.g. files skipped because they had no capture
    timestamp). Newline-joined paths so the frontend can split them.
    """
    db_entry = get_queue_entry(db, entry_id)
    if db_entry is None:
        return None

    db_entry.status = status

    if error:
        db_entry.error = error

    if warnings:
        db_entry.warnings = warnings

    if deployment_id:
        db_entry.deployment_id = deployment_id

    if status in ["completed", "failed"]:
        db_entry.processed_at_utc = datetime.now(UTC)

    db.commit()
    db.refresh(db_entry)
    return db_entry


def delete_queue_entry(db: Session, entry_id: str) -> bool:
    """
    Delete a queue entry.

    Returns True if deleted, False if entry doesn't exist.
    """
    db_entry = get_queue_entry(db, entry_id)
    if db_entry is None:
        return False

    db.delete(db_entry)
    db.commit()
    return True
