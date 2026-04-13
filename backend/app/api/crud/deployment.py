"""
CRUD operations for Deployment model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas.deployment import DeploymentCreate, DeploymentUpdate
from app.core.logging_config import get_logger
from app.models import Deployment, Event, File, Site

logger = get_logger(__name__)

# Number of files to sample when verifying a deployment folder's identity.
_VERIFY_SAMPLE_SIZE = 10


def get_deployments(
    db: Session,
    site_id: str | None = None,
    project_id: str | None = None,
) -> list[Deployment]:
    """
    Get all deployments, optionally filtered by site_id or project_id.

    Returns empty list if no deployments exist.
    """
    query = select(Deployment).order_by(Deployment.created_at.desc())
    if site_id:
        query = query.where(Deployment.site_id == site_id)
    if project_id:
        query = query.join(Site).where(Site.project_id == project_id)
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

    Crashes if database constraint violated (e.g., invalid site_id).
    This is intentional - we want to surface errors immediately.
    """
    db_deployment = Deployment(
        site_id=deployment.site_id,
        folder_path=deployment.folder_path,
        start_date=deployment.start_date,
        end_date=deployment.end_date,
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
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return None

    # Only update provided fields
    update_data = deployment.model_dump(exclude_unset=True)

    # If folder_path is being updated, update validation timestamp
    if "folder_path" in update_data and update_data["folder_path"] is not None:
        db_deployment.folder_status = "valid"
        db_deployment.last_validated_at = datetime.utcnow()

    for field, value in update_data.items():
        setattr(db_deployment, field, value)

    db.commit()
    db.refresh(db_deployment)
    return db_deployment


def delete_deployment(db: Session, deployment_id: str) -> bool:
    """
    Delete a deployment.

    Returns True if deleted, False if deployment doesn't exist.
    Cascades to all related files and events.
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return False

    db.delete(db_deployment)
    db.commit()
    return True


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

    Single round-trip per count type. Returns {deployment_id: {file_count, event_count, detection_count}}.
    """
    from app.models import Detection

    # All deployment IDs in project
    dep_ids_subq = (
        select(Deployment.id)
        .join(Site)
        .where(Site.project_id == project_id)
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
    db_deployment.last_validated_at = datetime.utcnow()
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
    now = datetime.utcnow()

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
        dep.last_validated_at = now
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
    deployment.last_validated_at = datetime.utcnow()

    db.commit()
    db.refresh(deployment)

    return RelinkResult(
        success=True,
        files_rewritten=files_rewritten,
        verify_result=result,
    )
