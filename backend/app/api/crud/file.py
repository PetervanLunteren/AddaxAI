"""
CRUD operations for files.
"""

from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.api.schemas.file import FileUpdate
from app.ml.observation_type import derive_observation_type
from app.models import Deployment, Detection, File, Project


def _get_counting_threshold(db: Session, file: File) -> float:
    """Get the project's detection threshold for a file."""
    row = (
        db.query(Project.counting_threshold)
        .join(Deployment, Deployment.project_id == Project.id)
        .filter(Deployment.id == file.deployment_id)
        .first()
    )
    return row[0] if row else 0.0


def get_files(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    observation_type: str | None = None,
) -> list[File]:
    """
    Get all files with pagination.

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Number of records to return
        observation_type: Optional filter by observation type

    Returns:
        List of files
    """
    query = db.query(File)
    if observation_type:
        query = query.filter(File.observation_type == observation_type)
    # file_path tiebreak (here and in the sibling queries below): burst
    # shots share one second-resolution timestamp, and offset pagination
    # over a non-unique sort key can skip or repeat rows across pages.
    return (
        query.order_by(File.captured_at_local.desc(), File.file_path.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_files_by_deployment(
    db: Session,
    deployment_id: str,
    skip: int = 0,
    limit: int = 100,
    observation_type: str | None = None,
) -> list[File]:
    """
    Get files by deployment ID.

    Args:
        db: Database session
        deployment_id: Deployment ID
        skip: Number of records to skip
        limit: Number of records to return
        observation_type: Optional filter by observation type

    Returns:
        List of files
    """
    query = db.query(File).filter(File.deployment_id == deployment_id)
    if observation_type:
        query = query.filter(File.observation_type == observation_type)
    return (
        query.order_by(File.captured_at_local.desc(), File.file_path.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_files_by_project(
    db: Session,
    project_id: str,
    skip: int = 0,
    limit: int = 100,
    observation_type: str | None = None,
) -> list[File]:
    """
    Get files by project ID.

    Args:
        db: Database session
        project_id: Project ID
        skip: Number of records to skip
        limit: Number of records to return
        observation_type: Optional filter by observation type

    Returns:
        List of files
    """
    query = (
        db.query(File)
        .join(Deployment)
        .join(Deployment.site)
        .filter(Deployment.site.has(project_id=project_id))
    )
    if observation_type:
        query = query.filter(File.observation_type == observation_type)
    return (
        query.order_by(File.captured_at_local.desc(), File.file_path.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_file_with_detections(db: Session, file_id: str) -> File | None:
    """
    Get file by ID with detections loaded.

    Args:
        db: Database session
        file_id: File ID

    Returns:
        File with detections or None if not found
    """
    return (
        db.query(File)
        .options(joinedload(File.detections))
        .filter(File.id == file_id)
        .first()
    )


def recompute_file_verified(db: Session, file_ids: Iterable[str]) -> None:
    """Maintain `File.verified` as the observation-level rollup.

    A file is verified when every reviewable detection on it is verified.
    "Reviewable" = confidence >= the project's detection threshold OR
    already verified (the standard threshold-or-verified rule).

    Files with no reviewable detections (empty / blank) are left
    untouched: there are no observations to roll up, so `File.verified`
    is owned directly by the file-verify action (a human reviewing the
    empty frame).

    Call this after any change to a detection's verified status. Every
    user-facing surface (file badge, file filter, "next unverified"
    navigation, the event MaxN rollup) reads `File.verified`, so keeping
    it in sync here is the single point where the upward cascade lives.
    No commit — the caller owns the transaction.
    """
    ids = [fid for fid in dict.fromkeys(file_ids)]
    if not ids:
        return
    files = db.query(File).filter(File.id.in_(ids)).all()
    if not files:
        return

    # One threshold lookup per deployment, shared across its files.
    threshold_cache: dict[str, float] = {}
    now = datetime.now(UTC)
    for f in files:
        dep_id = f.deployment_id
        if dep_id not in threshold_cache:
            threshold_cache[dep_id] = _get_counting_threshold(db, f)
        floor = threshold_cache[dep_id]

        rows = (
            db.query(Detection.verified)
            .filter(Detection.file_id == f.id)
            .filter(
                or_(Detection.confidence >= floor, Detection.verified == True)  # noqa: E712
            )
            .all()
        )
        if not rows:
            # No reviewable detections: leave File.verified as-is (owned
            # by the file-verify action for empty frames).
            continue

        new_verified = all(r[0] for r in rows)
        if new_verified != f.verified:
            f.verified = new_verified
            f.verified_at_utc = now if new_verified else None


def recompute_file_verified_for_detections(
    db: Session, detection_ids: Iterable[str]
) -> None:
    """Recompute `File.verified` for the files owning the given detections."""
    ids = list(dict.fromkeys(detection_ids))
    if not ids:
        return
    file_ids = [
        fid
        for (fid,) in db.query(Detection.file_id)
        .filter(Detection.id.in_(ids))
        .distinct()
        .all()
    ]
    recompute_file_verified(db, file_ids)


def update_file(db: Session, file_id: str, update: FileUpdate) -> File | None:
    """
    Update a file's verification status and/or notes.

    Sets verified_at_utc to current time when verified changes to True,
    clears it when verified changes to False.
    """
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return None

    if update.verified is not None:
        # The file-verify action (modal Enter). Sets File.verified and
        # cascades down to the detections. recompute keeps File.verified
        # consistent for files that have detections; for empty frames it
        # is a no-op, so the flag we set here stands.
        if update.verified and not file.verified:
            now = datetime.now(UTC)
            file.verified = True
            file.verified_at_utc = now
            # Only verify detections above the project's detection threshold
            # (below-threshold detections are not visible to the user)
            threshold = _get_counting_threshold(db, file)
            det_filter = [
                Detection.file_id == file_id,
                Detection.verified == False,  # noqa: E712
                Detection.confidence >= threshold,
            ]
            db.query(Detection).filter(*det_filter).update(
                {"verified": True, "verified_at_utc": now}
            )
            recompute_file_verified(db, [file_id])
        elif not update.verified and file.verified:
            file.verified = False
            file.verified_at_utc = None
            db.query(Detection).filter(
                Detection.file_id == file_id,
            ).update({"verified": False, "verified_at_utc": None})
            recompute_file_verified(db, [file_id])

    if update.notes is not None:
        file.notes = update.notes

    if update.favorited is not None:
        file.favorited = update.favorited

    if update.flagged is not None:
        if update.flagged and not file.flagged:
            file.flagged = True
            file.flagged_at_utc = datetime.now(UTC)
        elif not update.flagged and file.flagged:
            file.flagged = False
            file.flagged_at_utc = None

    db.commit()
    db.refresh(file)
    return file


def _project_threshold_for_file(db: Session, file: File) -> float:
    """The detection threshold of the project a file belongs to.

    observation_type only counts detections at or above this (or verified),
    so the threshold has to come from the owning project. Defaults to 0.0
    if the chain is somehow broken (every detection then passes, matching
    the pre-threshold behaviour).
    """
    row = (
        db.query(Project.counting_threshold)
        .join(Deployment, Deployment.project_id == Project.id)
        .filter(Deployment.id == file.deployment_id)
        .first()
    )
    return row[0] if row else 0.0


def recalculate_observation_type(db: Session, file_id: str) -> None:
    """
    Re-derive observation_type from the file's *passing* detections.

    Passing = over the project threshold OR verified (see
    ``derive_observation_type``). Called after any detection create /
    update / delete / verify so the summary stays consistent with what
    the verify grid shows.
    """
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return

    detections = (
        db.query(Detection)
        .filter(Detection.file_id == file_id)
        .all()
    )
    threshold = _project_threshold_for_file(db, file)
    file.observation_type = derive_observation_type(detections, threshold)
    db.commit()


def recalculate_observation_types_for_project(
    db: Session, project_id: str
) -> int:
    """Re-derive observation_type for every file in a project.

    Run when the project detection threshold changes: the threshold feeds
    the passing rule, so a file whose only detection now falls below it
    flips to ``"blank"`` (and vice-versa). Returns the number of files
    whose observation_type actually changed. One pass, one commit.
    """
    project = db.get(Project, project_id)
    if project is None:
        return 0
    threshold = project.counting_threshold

    files = (
        db.query(File)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .all()
    )
    detections = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .all()
    )
    by_file: dict[str, list[Detection]] = {}
    for det in detections:
        by_file.setdefault(det.file_id, []).append(det)

    changed = 0
    for file in files:
        new_type = derive_observation_type(
            by_file.get(file.id, []), threshold
        )
        if file.observation_type != new_type:
            file.observation_type = new_type
            changed += 1
    db.commit()
    return changed
