"""
CRUD operations for files.
"""

from datetime import UTC, datetime, time

from sqlalchemy import Integer, exists, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.api.crud.event import (
    _sort_adjacency_predicates,
    _sort_order_by_clauses,
)
from app.api.schemas.file import FileUpdate
from app.models import Deployment, Detection, File, Project


def _file_min_cls_subquery():
    """Correlated scalar subquery: MIN(label_confidence) over a file's own
    detections. The Images grid lists images and frame rows directly, so
    detections always live on the file row itself.
    """
    return (
        select(func.min(Detection.label_confidence))
        .where(Detection.file_id == File.id)
        .correlate(File)
        .scalar_subquery()
    )


def _file_sort_spec(sort: str, seed: int | None):
    """Map (sort, seed) to (sort_key_expression, descending, nulls_last)."""
    if sort == "newest":
        return File.captured_at_local, True, False
    if sort == "oldest":
        return File.captured_at_local, False, False
    if sort == "random":
        if seed is None:
            raise ValueError("random sort requires a seed")
        return func.seeded_hash(File.id, seed), False, False
    if sort == "cls_low":
        return _file_min_cls_subquery(), False, True
    raise ValueError(f"unknown sort: {sort}")


def _get_detection_threshold(db: Session, file: File) -> float:
    """Get the project's detection threshold for a file."""
    row = (
        db.query(Project.detection_threshold)
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
    return query.order_by(File.captured_at_local.desc()).offset(skip).limit(limit).all()


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
    return query.order_by(File.captured_at_local.desc()).offset(skip).limit(limit).all()


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
    return query.order_by(File.captured_at_local.desc()).offset(skip).limit(limit).all()


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
        if update.verified and not file.verified:
            now = datetime.now(UTC)
            file.verified = True
            file.verified_at_utc = now
            # Only verify detections above the project's detection threshold
            # (below-threshold detections are not visible to the user)
            threshold = _get_detection_threshold(db, file)
            det_filter = [
                Detection.file_id == file_id,
                Detection.verified == False,  # noqa: E712
                Detection.confidence >= threshold,
            ]
            db.query(Detection).filter(*det_filter).update(
                {"verified": True, "verified_at_utc": now}
            )
        elif not update.verified and file.verified:
            file.verified = False
            file.verified_at_utc = None
            db.query(Detection).filter(
                Detection.file_id == file_id,
            ).update({"verified": False, "verified_at_utc": None})

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


def get_observation_type_stats(
    db: Session, project_id: str
) -> dict[str, int]:
    """
    Get observation type counts for a project.

    Args:
        db: Database session
        project_id: Project ID

    Returns:
        Dict mapping observation_type -> count
    """
    rows = (
        db.query(File.observation_type, func.count(File.id))
        .join(Deployment)
        .join(Deployment.site)
        .filter(Deployment.site.has(project_id=project_id))
        .filter(File.file_type.in_(["image", "video"]))
        .group_by(File.observation_type)
        .all()
    )
    return {obs_type: count for obs_type, count in rows}


def _apply_file_verify_filters(
    query,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
):
    """Apply shared filters to a File query. Expects File already joined to Deployment.

    Files are filtered to file_type IN ("image", "video"): images render
    directly, videos render their `best_frame_path` JPEG. Legacy
    `file_type="frame"` rows (pre-2026-05 migration) are dropped from
    the grid by the same filter and surface via their parent video row.

    `min_label_confidence` / `max_label_confidence` filter on
    `Detection.label_confidence` (the classifier score). NULL values are
    excluded automatically by SQL `WHERE` semantics — a NULL classification
    cannot satisfy any range, which is the correct behaviour: at default
    [0, 1] no params are sent, the WHERE clause never runs, and unclassified
    detections pass.

    `project_floor` is the project's `detection_threshold`. It is applied
    as `(Detection.confidence >= floor OR Detection.verified == True)` —
    the global "threshold + verified override" rule documented in
    DEVELOPERS.md. Distinct from `min_confidence` (the user's slider value)
    which is applied LITERALLY: a verified low-confidence detection
    satisfies the floor's OR clause, but cannot satisfy a user's narrow
    `min_confidence` window.

    When `empty == "show_only"`, the labels and confidence-floor gates
    are skipped: empty files have no detections by definition, so the
    "must have at least one visible detection" precondition would always
    exclude them. The user explicitly asked for empties; trust the ask.
    """
    from app.api.crud.deployment import site_ids_filter

    if empty == "show_only":
        labels = None
        min_confidence = None
        max_confidence = None
        min_label_confidence = None
        max_label_confidence = None
        project_floor = None

    query = query.filter(File.file_type.in_(("image", "video")))

    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.filter(site_clause)

    if date_from is not None:
        query = query.filter(File.captured_at_local >= date_from)

    if date_to is not None:
        end_of_day = (
            datetime.combine(date_to.date(), time.max)
            if isinstance(date_to, datetime)
            else date_to
        )
        query = query.filter(File.captured_at_local <= end_of_day)

    if labels:
        label_subq = (
            select(Detection.id)
            .where(Detection.file_id == File.id)
            .where(Detection.label_taxonomy_id.in_(labels))
        )
        if project_floor is not None:
            label_subq = label_subq.where(
                or_(
                    Detection.confidence >= project_floor,
                    Detection.verified == True,  # noqa: E712
                )
            )
        if min_confidence is not None:
            label_subq = label_subq.where(Detection.confidence >= min_confidence)
        if max_confidence is not None:
            label_subq = label_subq.where(Detection.confidence <= max_confidence)
        if min_label_confidence is not None:
            label_subq = label_subq.where(
                Detection.label_confidence >= min_label_confidence
            )
        if max_label_confidence is not None:
            label_subq = label_subq.where(
                Detection.label_confidence <= max_label_confidence
            )
        query = query.filter(exists(label_subq))
    elif (
        project_floor is not None
        or min_confidence is not None
        or max_confidence is not None
        or min_label_confidence is not None
        or max_label_confidence is not None
    ):
        conf_subq = (
            select(Detection.id)
            .where(Detection.file_id == File.id)
        )
        if project_floor is not None:
            conf_subq = conf_subq.where(
                or_(
                    Detection.confidence >= project_floor,
                    Detection.verified == True,  # noqa: E712
                )
            )
        if min_confidence is not None:
            conf_subq = conf_subq.where(Detection.confidence >= min_confidence)
        if max_confidence is not None:
            conf_subq = conf_subq.where(Detection.confidence <= max_confidence)
        if min_label_confidence is not None:
            conf_subq = conf_subq.where(
                Detection.label_confidence >= min_label_confidence
            )
        if max_label_confidence is not None:
            conf_subq = conf_subq.where(
                Detection.label_confidence <= max_label_confidence
            )
        # Blank files have no detections and so always fail an `EXISTS`
        # gate. When the only active gate is the implicit project floor
        # (no user-set conf range, no labels) and the user has not asked
        # to hide empties, let blanks pass — otherwise "Empty: All" would
        # silently drop every blank tile because of an invisible filter.
        # Explicit user filters (slider, label) keep their strict
        # semantics: a confidence range or label name has no meaning for
        # a file with zero detections.
        has_user_conf_gate = (
            min_confidence is not None
            or max_confidence is not None
            or min_label_confidence is not None
            or max_label_confidence is not None
        )
        if not has_user_conf_gate and empty != "hide":
            query = query.filter(
                or_(exists(conf_subq), File.observation_type == "blank")
            )
        else:
            query = query.filter(exists(conf_subq))

    if verification == "verified":
        query = query.filter(File.verified == True)  # noqa: E712
    elif verification == "unverified":
        query = query.filter(File.verified == False)  # noqa: E712

    if flagged == "flagged":
        query = query.filter(File.flagged == True)  # noqa: E712
    elif flagged == "not_flagged":
        query = query.filter(File.flagged == False)  # noqa: E712

    if favorited == "favorited":
        query = query.filter(File.favorited == True)  # noqa: E712
    elif favorited == "not_favorited":
        query = query.filter(File.favorited == False)  # noqa: E712

    if empty == "show_only":
        query = query.filter(File.observation_type == "blank")
    elif empty == "hide":
        query = query.filter(File.observation_type != "blank")

    return query


def get_files_for_verify(
    db: Session,
    project_id: str,
    skip: int = 0,
    limit: int = 48,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
    sort: str = "newest",
    seed: int | None = None,
) -> list[dict]:
    """List file summaries for the Images verify tab.

    Returns a list of dicts shaped like FileSummary. One row per file:
    file_type IN ("image", "video"). Video rows render via their
    `best_frame_path` thumbnail; the underlying mp4 stays out of the
    grid.
    """
    query = (
        db.query(File)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
    )
    query = _apply_file_verify_filters(
        query,
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        project_floor=project_floor,
    )
    sort_key, descending, nulls_last = _file_sort_spec(sort, seed)
    order_clauses = _sort_order_by_clauses(
        sort_key, File.id, descending=descending, nulls_last=nulls_last,
    )
    files = (
        query.options(
            joinedload(File.deployment).joinedload(Deployment.site),
            joinedload(File.detections),
        )
        .order_by(*order_clauses)
        .offset(skip)
        .limit(limit)
        .all()
    )

    summaries: list[dict] = []
    for f in files:
        dets = list(f.detections)

        # Collect unique labels for this file's visible detections.
        label_set: set[str] = set()
        label_to_display: dict[str, str] = {}
        for d in dets:
            meets_floor = (
                project_floor is None
                or d.confidence >= project_floor
                or d.verified
            )
            meets_min = (
                min_confidence is None or d.confidence >= min_confidence
            )
            meets_max = max_confidence is None or d.confidence <= max_confidence
            if not (meets_floor and meets_min and meets_max):
                continue
            tid = d.label_taxonomy_id
            if tid:
                label_set.add(tid)
                display = d.display_name or d.label or d.category
                if display and tid not in label_to_display:
                    label_to_display[tid] = display

        site = f.deployment.site if f.deployment else None
        observation_types = [f.observation_type] if f.observation_type else []

        summaries.append(
            {
                "id": f.id,
                "deployment_id": f.deployment_id,
                "file_type": f.file_type,
                "file_format": f.file_format,
                "width_px": f.width_px,
                "height_px": f.height_px,
                "captured_at_local": f.captured_at_local,
                "site_id": site.id if site else None,
                "site_name": site.name if site else None,
                "observation_type": f.observation_type,
                "observation_types": observation_types,
                "labels": sorted(label_set),
                "display_labels": label_to_display,
                "verified": f.verified,
                "favorited": f.favorited,
                "flagged": f.flagged,
                "source_video_id": f.source_video_id,
                "best_frame_number": f.best_frame_number,
                "detections": [
                    {
                        "id": d.id,
                        "category": d.category,
                        "confidence": d.confidence,
                        "bbox_x": d.bbox_x,
                        "bbox_y": d.bbox_y,
                        "bbox_width": d.bbox_width,
                        "bbox_height": d.bbox_height,
                        "label": d.label,
                        "label_taxonomy_id": d.label_taxonomy_id,
                        "frame_number": d.frame_number,
                    }
                    for d in dets
                ],
            }
        )
    return summaries


def count_files_for_verify(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
) -> int:
    """Total file count for the Files verify tab with the given filters."""
    query = (
        db.query(func.count(File.id))
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
    )
    query = _apply_file_verify_filters(
        query,
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        project_floor=project_floor,
    )
    return query.scalar() or 0


def get_file_verification_stats(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
) -> dict[str, int]:
    """Aggregate verified/total file counts for the Files verify tab."""
    query = (
        db.query(
            func.count(File.id),
            func.sum(func.cast(File.verified, Integer)),
        )
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
    )
    query = _apply_file_verify_filters(
        query,
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        project_floor=project_floor,
    )
    total, verified = query.one()
    return {
        "total_files": total or 0,
        "verified_files": int(verified or 0),
    }


def get_adjacent_files_for_verify(
    db: Session,
    file_id: str,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
    sort: str = "newest",
    seed: int | None = None,
) -> dict:
    """Adjacent file IDs in the Files verify tab's filtered list.

    Order matches `get_files_for_verify` for the same `(sort, seed)`. The
    next/prev predicates are derived from the active sort key so modal
    Next/Prev tracks the displayed grid order.
    """
    filter_kwargs = dict(
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        project_floor=project_floor,
    )

    sort_key, descending, nulls_last = _file_sort_spec(sort, seed)

    current_value = db.query(sort_key).filter(File.id == file_id).scalar()
    current = db.query(File.id).filter(File.id == file_id).first()
    if not current:
        return {
            "previous_id": None,
            "next_id": None,
            "next_unverified_id": None,
            "current_index": 0,
            "total_count": 0,
        }

    cid = current.id

    next_pred, prev_pred, next_order, prev_order = _sort_adjacency_predicates(
        sort_key, File.id, current_value, cid,
        descending=descending, nulls_last=nulls_last,
    )

    def base():
        q = (
            db.query(File.id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .filter(Deployment.project_id == project_id)
        )
        return _apply_file_verify_filters(q, **filter_kwargs)

    prev = base().filter(prev_pred).order_by(*prev_order).first()
    nxt = base().filter(next_pred).order_by(*next_order).first()
    nxt_unv = (
        base()
        .filter(next_pred)
        .filter(File.verified == False)  # noqa: E712
        .order_by(*next_order)
        .first()
    )

    total_q = (
        db.query(func.count(File.id))
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
    )
    total_q = _apply_file_verify_filters(total_q, **filter_kwargs)
    total = total_q.scalar() or 0

    idx_q = (
        db.query(func.count(File.id))
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(prev_pred)
    )
    idx_q = _apply_file_verify_filters(idx_q, **filter_kwargs)
    idx = idx_q.scalar() or 0

    return {
        "previous_id": prev[0] if prev else None,
        "next_id": nxt[0] if nxt else None,
        "next_unverified_id": nxt_unv[0] if nxt_unv else None,
        "current_index": idx,
        "total_count": total,
    }


def recalculate_observation_type(db: Session, file_id: str) -> None:
    """
    Re-derive observation_type from current detections.

    Priority: animal > human > vehicle > blank.
    Called after detection create/update/delete.
    """
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return

    detections = (
        db.query(Detection)
        .filter(Detection.file_id == file_id)
        .all()
    )

    if not detections:
        file.observation_type = "blank"
    else:
        # Map detection categories to observation types
        category_map = {"animal": "animal", "person": "human", "vehicle": "vehicle"}
        priority = {"animal": 4, "human": 3, "vehicle": 2}

        best_type = "blank"
        best_priority = 0
        for d in detections:
            obs = category_map.get(d.category, "unknown")
            p = priority.get(obs, 0)
            if p > best_priority:
                best_priority = p
                best_type = obs

        file.observation_type = best_type

    db.commit()
