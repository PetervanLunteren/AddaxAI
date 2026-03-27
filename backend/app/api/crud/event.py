"""
CRUD operations for events.

Events are time-clustered groups of files within a deployment.
"""

import uuid
from datetime import datetime, time

from sqlalchemy import Integer, and_, delete, exists, func, insert, or_, select
from sqlalchemy.orm import Session, aliased, joinedload

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, Event, File, Project, Site
from app.models.event import event_files
from app.models.event_observation import EventObservation

logger = get_logger(__name__)


def _apply_event_filters(
    query,
    db: Session,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
):
    """Apply shared filters to an event query. Expects Event already joined to Deployment→Site."""
    if site_ids:
        query = query.filter(Site.id.in_(site_ids))

    if date_from is not None:
        query = query.filter(Event.start_time >= date_from)

    if date_to is not None:
        # Include the entire end-of-day
        end_of_day = (
            datetime.combine(date_to.date(), time.max) if isinstance(date_to, datetime) else date_to
        )
        query = query.filter(Event.start_time <= end_of_day)

    if labels:
        # EXISTS subquery: event has at least one file with a detection matching labels
        # Use COALESCE to fall back to category when label is null (detection-only projects)
        effective_label = func.coalesce(Detection.label, Detection.category)
        label_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .join(Detection, Detection.file_id == File.id)
            .where(event_files.c.event_id == Event.id)
            .where(effective_label.in_(labels))
        )
        if min_confidence is not None:
            label_subq = label_subq.where(
                or_(Detection.confidence >= min_confidence, Detection.verified == True)  # noqa: E712
            )
        if max_confidence is not None:
            label_subq = label_subq.where(Detection.confidence <= max_confidence)
        query = query.filter(exists(label_subq))
    elif min_confidence is not None or max_confidence is not None:
        # Standalone confidence filter: event has at least one detection in range
        conf_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .join(Detection, Detection.file_id == File.id)
            .where(event_files.c.event_id == Event.id)
        )
        if min_confidence is not None:
            conf_subq = conf_subq.where(
                or_(Detection.confidence >= min_confidence, Detection.verified == True)  # noqa: E712
            )
        if max_confidence is not None:
            conf_subq = conf_subq.where(Detection.confidence <= max_confidence)
        query = query.filter(exists(conf_subq))

    if verification and verification != "all":
        if verification == "fully_verified":
            # All files in event are verified: NOT EXISTS any unverified file
            unverified_subq = (
                select(event_files.c.event_id)
                .join(File, File.id == event_files.c.file_id)
                .where(event_files.c.event_id == Event.id)
                .where(File.verified == False)  # noqa: E712
            )
            query = query.filter(~exists(unverified_subq))
        elif verification == "not_fully_verified":
            # At least one file is unverified
            unverified_subq = (
                select(event_files.c.event_id)
                .join(File, File.id == event_files.c.file_id)
                .where(event_files.c.event_id == Event.id)
                .where(File.verified == False)  # noqa: E712
            )
            query = query.filter(exists(unverified_subq))
        elif verification == "unverified_maxn":
            # No MaxN frames verified: all MaxN frames are unverified
            MaxNFile = aliased(File)
            has_verified_maxn = exists(
                select(EventObservation.id)
                .join(MaxNFile, MaxNFile.id == EventObservation.max_n_file_id)
                .where(
                    and_(
                        EventObservation.event_id == Event.id,
                        MaxNFile.verified == True,  # noqa: E712
                    )
                )
            )
            query = query.filter(~has_verified_maxn)
        elif verification == "all_maxn_verified":
            # Every MaxN frame is verified: NOT EXISTS any unverified MaxN
            MaxNFile = aliased(File)
            has_unverified_maxn = exists(
                select(EventObservation.id)
                .join(MaxNFile, MaxNFile.id == EventObservation.max_n_file_id)
                .where(
                    and_(
                        EventObservation.event_id == Event.id,
                        MaxNFile.verified == False,  # noqa: E712
                    )
                )
            )
            # Must have at least one MaxN frame
            has_any_maxn = exists(
                select(EventObservation.id).where(
                    and_(
                        EventObservation.event_id == Event.id,
                        EventObservation.max_n_file_id.isnot(None),
                    )
                )
            )
            query = query.filter(has_any_maxn, ~has_unverified_maxn)
        elif verification == "some_maxn_verified":
            # At least one MaxN verified AND at least one not verified
            VerifiedMaxNFile = aliased(File)
            UnverifiedMaxNFile = aliased(File)
            has_verified = exists(
                select(EventObservation.id)
                .join(
                    VerifiedMaxNFile,
                    VerifiedMaxNFile.id == EventObservation.max_n_file_id,
                )
                .where(
                    and_(
                        EventObservation.event_id == Event.id,
                        VerifiedMaxNFile.verified == True,  # noqa: E712
                    )
                )
            )
            has_unverified = exists(
                select(EventObservation.id)
                .join(
                    UnverifiedMaxNFile,
                    UnverifiedMaxNFile.id == EventObservation.max_n_file_id,
                )
                .where(
                    and_(
                        EventObservation.event_id == Event.id,
                        UnverifiedMaxNFile.verified == False,  # noqa: E712
                    )
                )
            )
            query = query.filter(has_verified, has_unverified)
        elif verification == "none_verified":
            # Zero files verified: NOT EXISTS any verified file
            verified_subq = (
                select(event_files.c.event_id)
                .join(File, File.id == event_files.c.file_id)
                .where(event_files.c.event_id == Event.id)
                .where(File.verified == True)  # noqa: E712
            )
            query = query.filter(~exists(verified_subq))

    return query


def generate_events_for_project(db: Session, project_id: str) -> int:
    """
    Generate events for all deployments in a project.

    Idempotent: deletes existing events before regenerating.

    1. Fetch project's independence_interval
    2. Delete all existing events for every deployment in the project
    3. For each deployment, query files ordered by timestamp ASC
    4. Walk files: start new event when gap > independence_interval seconds
    5. Create Event records with start_time, end_time, file_count
    6. Insert event_files junction rows with sequence_number

    Returns total event count created.
    """
    from app.models import Project

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    independence_interval = project.independence_interval  # seconds

    # Get all deployments for this project
    deployments = db.query(Deployment).join(Site).filter(Site.project_id == project_id).all()

    # Delete existing events for all deployments in this project
    deployment_ids = [d.id for d in deployments]
    if deployment_ids:
        db.execute(delete(Event).where(Event.deployment_id.in_(deployment_ids)))

    total_events = 0

    for deployment in deployments:
        files = (
            db.query(File)
            .filter(File.deployment_id == deployment.id)
            .filter(File.file_type.in_(["image", "frame"]))
            .order_by(File.timestamp.asc())
            .all()
        )

        if not files:
            continue

        # Walk files and cluster into events
        current_event_files: list[File] = [files[0]]

        for i in range(1, len(files)):
            gap = (files[i].timestamp - files[i - 1].timestamp).total_seconds()

            if gap > independence_interval:
                # Save current event and start new one
                _create_event(db, deployment.id, current_event_files)
                total_events += 1
                current_event_files = [files[i]]
            else:
                current_event_files.append(files[i])

        # Save last event
        if current_event_files:
            _create_event(db, deployment.id, current_event_files)
            total_events += 1

    db.flush()

    # Calculate MaxN observations for all events
    from app.api.crud.event_observation import recalculate_max_n_for_project

    recalculate_max_n_for_project(db, project_id)

    db.commit()
    return total_events


def _create_event(db: Session, deployment_id: str, files: list[File]) -> Event:
    """Create an event with its junction table entries."""
    event = Event(
        id=str(uuid.uuid4()),
        deployment_id=deployment_id,
        start_time=files[0].timestamp,
        end_time=files[-1].timestamp,
        file_count=len(files),
    )
    db.add(event)
    db.flush()  # Get event.id assigned

    # Insert junction table rows
    for seq, file in enumerate(files):
        db.execute(
            insert(event_files).values(
                event_id=event.id,
                file_id=file.id,
                sequence_number=seq,
            )
        )

    return event


def get_events_by_project(
    db: Session,
    project_id: str,
    skip: int = 0,
    limit: int = 50,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> list[dict]:
    """
    Get event summaries for a project.

    Returns event data with representative file, label list,
    observation type, and verification progress.
    """
    query = db.query(Event).join(Deployment).join(Site).filter(Site.project_id == project_id)
    query = _apply_event_filters(
        query,
        db,
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    events = (
        query.options(joinedload(Event.files).joinedload(File.detections))
        .order_by(Event.start_time.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Deduplicate events (joinedload can produce duplicates)
    seen_ids: set[str] = set()
    unique_events: list[Event] = []
    for event in events:
        if event.id not in seen_ids:
            seen_ids.add(event.id)
            unique_events.append(event)

    # Batch-load MaxN frames for all events in this page
    from app.api.crud.event_observation import get_max_n_frames

    event_ids = [e.id for e in unique_events]
    max_n_by_event: dict[str, list[dict]] = {}
    for eid in event_ids:
        max_n_by_event[eid] = get_max_n_frames(db, eid)

    summaries = []
    for event in unique_events:
        # Sort files by sequence within event
        sorted_files = sorted(
            event.files,
            key=lambda f: f.timestamp,
        )

        # Collect unique labels across all files (fall back to category for detection-only)
        label_set: set[str] = set()
        label_to_display: dict[str, str] = {}
        for f in sorted_files:
            for d in f.detections:
                meets_confidence = (
                    min_confidence is None
                    or d.confidence >= min_confidence
                    or d.verified
                )
                if meets_confidence and (
                    max_confidence is None or d.confidence <= max_confidence
                ):
                    raw = d.label if d.label is not None else d.category
                    label_set.add(raw)
                    if d.display_name and raw not in label_to_display:
                        label_to_display[raw] = d.display_name

        # Determine dominant observation type (animal > human > vehicle > blank)
        obs_priority = {"animal": 4, "human": 3, "vehicle": 2, "blank": 1}
        dominant_type = "blank"
        dominant_priority = 0
        observation_types_set: set[str] = set()
        for f in sorted_files:
            if f.observation_type:
                observation_types_set.add(f.observation_type)
            p = obs_priority.get(f.observation_type, 0)
            if p > dominant_priority:
                dominant_priority = p
                dominant_type = f.observation_type

        # Count files by type and verification
        image_count = sum(1 for f in sorted_files if f.file_type == "image")
        frame_count = sum(1 for f in sorted_files if f.file_type == "frame")
        video_count = len(
            {
                f.source_video_id
                for f in sorted_files
                if f.file_type == "frame" and f.source_video_id
            }
        )
        verified_count = sum(1 for f in sorted_files if f.verified)

        # MaxN verification counts
        max_n_frames = max_n_by_event.get(event.id, [])
        file_verified_map = {f.id: f.verified for f in sorted_files}
        total_maxn_count = len(max_n_frames)
        verified_maxn_count = sum(
            1
            for mf in max_n_frames
            if file_verified_map.get(mf["file_id"], False)
        )

        # MaxN-derived thumbnail: dominant species' MaxN frame, fallback to first file
        thumbnail_file_id = max_n_frames[0]["file_id"] if max_n_frames else (
            sorted_files[0].id if sorted_files else None
        )

        summaries.append(
            {
                "id": event.id,
                "deployment_id": event.deployment_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "file_count": event.file_count,
                "thumbnail_file_id": thumbnail_file_id,
                "max_n_frames": max_n_frames,
                "site_name": event.deployment.site.name
                if event.deployment and event.deployment.site
                else None,
                "labels": sorted(label_set),
                "display_labels": {
                    k: v for k, v in label_to_display.items()
                },
                "observation_type": dominant_type,
                "observation_types": sorted(observation_types_set),
                "image_count": image_count,
                "frame_count": frame_count,
                "video_count": video_count,
                "verified_count": verified_count,
                "total_count": len(sorted_files),
                "verified_maxn_count": verified_maxn_count,
                "total_maxn_count": total_maxn_count,
            }
        )

    return summaries


def get_event_with_files(db: Session, event_id: str) -> Event | None:
    """
    Get event with all files and their detections, ordered by sequence_number.
    """
    event = (
        db.query(Event)
        .options(joinedload(Event.files).joinedload(File.detections))
        .options(joinedload(Event.deployment).joinedload(Deployment.site))
        .filter(Event.id == event_id)
        .first()
    )
    return event


def get_event_count_by_project(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> int:
    """Get total event count for a project."""
    query = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
    )
    query = _apply_event_filters(
        query,
        db,
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    count = query.scalar()
    return count or 0


def get_adjacent_events(
    db: Session,
    event_id: str,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> dict:
    """
    Get adjacent event IDs for navigation.

    Returns previous_id, next_id, next_unverified_id, current_index, total_count.
    Events ordered by start_time DESC (newest first).

    Uses targeted SQL queries instead of loading all events into memory.
    When filters are provided, navigation is scoped to the filtered set.
    """
    filter_kwargs = dict(
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    # 1. Get current event's start_time
    current = db.query(Event.id, Event.start_time).filter(Event.id == event_id).first()
    if not current:
        return {
            "previous_id": None,
            "next_id": None,
            "next_unverified_id": None,
            "current_index": 0,
            "total_count": 0,
        }

    ct = current.start_time
    cid = current.id

    def base():
        q = db.query(Event.id).join(Deployment).join(Site).filter(Site.project_id == project_id)
        return _apply_event_filters(q, db, **filter_kwargs)

    # 2. Previous (newer in DESC order): start_time > current, or same time + higher id
    prev = (
        base()
        .filter((Event.start_time > ct) | ((Event.start_time == ct) & (Event.id > cid)))
        .order_by(Event.start_time.asc(), Event.id.asc())
        .first()
    )

    # 3. Next (older in DESC order): start_time < current, or same time + lower id
    nxt = (
        base()
        .filter((Event.start_time < ct) | ((Event.start_time == ct) & (Event.id < cid)))
        .order_by(Event.start_time.desc(), Event.id.desc())
        .first()
    )

    # 4. Next unverified (older, with at least one unverified file).
    # Uses base() so all active filters (labels, sites, dates) are respected.
    unv_file = aliased(File)
    unv_subq = (
        select(event_files.c.event_id)
        .join(unv_file, unv_file.id == event_files.c.file_id)
        .where(event_files.c.event_id == Event.id)
        .where(unv_file.verified == False)  # noqa: E712
        .correlate(Event)
    )
    nxt_unv = (
        base()
        .filter((Event.start_time < ct) | ((Event.start_time == ct) & (Event.id < cid)))
        .filter(exists(unv_subq))
        .order_by(Event.start_time.desc(), Event.id.desc())
        .first()
    )

    # 5a. Total count
    total_q = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
    )
    total_q = _apply_event_filters(total_q, db, **filter_kwargs)
    total = total_q.scalar() or 0

    # 5b. Current index (number of events newer than current = position in DESC list)
    idx_q = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .filter((Event.start_time > ct) | ((Event.start_time == ct) & (Event.id > cid)))
    )
    idx_q = _apply_event_filters(idx_q, db, **filter_kwargs)
    idx = idx_q.scalar() or 0

    return {
        "previous_id": prev[0] if prev else None,
        "next_id": nxt[0] if nxt else None,
        "next_unverified_id": nxt_unv[0] if nxt_unv else None,
        "current_index": idx,
        "total_count": total,
    }


def get_event_verification_stats(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    labels: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> dict[str, int]:
    """Get aggregate file verification stats across filtered events."""
    filter_kwargs = dict(
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    # Base: filtered event IDs
    event_ids_q = (
        db.query(Event.id).join(Deployment).join(Site).filter(Site.project_id == project_id)
    )
    event_ids_q = _apply_event_filters(event_ids_q, db, **filter_kwargs)
    event_ids_subq = event_ids_q.subquery()

    # Query 1: file-level counts
    file_stats = (
        db.query(
            func.count(func.distinct(File.id)),
            func.sum(func.cast(File.verified, Integer)),
        )
        .join(event_files, event_files.c.file_id == File.id)
        .filter(event_files.c.event_id.in_(select(event_ids_subq.c.id)))
        .one()
    )

    # Query 2: MaxN frame counts (distinct max_n_file_ids and their verification)
    MaxNFile = aliased(File)
    maxn_stats = (
        db.query(
            func.count(func.distinct(EventObservation.max_n_file_id)),
            func.sum(
                func.cast(MaxNFile.verified, Integer)
            ),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(MaxNFile, MaxNFile.id == EventObservation.max_n_file_id)
        .filter(Event.id.in_(select(event_ids_subq.c.id)))
        .filter(EventObservation.max_n_file_id.isnot(None))
        .one()
    )

    # Query 3: observation counts (MaxN-based from event_observations)
    obs_q = (
        db.query(
            func.coalesce(func.sum(EventObservation.max_n), 0),
        )
        .join(Event, Event.id == EventObservation.event_id)
        .filter(Event.id.in_(select(event_ids_subq.c.id)))
    )
    total_observations = obs_q.scalar() or 0

    # Verified detection count (still useful for verification progress)
    det_verified_q = (
        db.query(
            func.sum(func.cast(Detection.verified, Integer)),
        )
        .join(File, File.id == Detection.file_id)
        .join(event_files, event_files.c.file_id == File.id)
        .filter(event_files.c.event_id.in_(select(event_ids_subq.c.id)))
    )
    if min_confidence is not None:
        det_verified_q = det_verified_q.filter(
            or_(Detection.confidence >= min_confidence, Detection.verified == True)  # noqa: E712
        )
    verified_detections = int(det_verified_q.scalar() or 0)

    return {
        "total_files": file_stats[0] or 0,
        "verified_files": int(file_stats[1] or 0),
        "total_max_n_frames": maxn_stats[0] or 0,
        "verified_max_n_frames": int(maxn_stats[1] or 0),
        "total_observations": total_observations,
        "verified_detections": verified_detections,
    }


def get_filter_options(db: Session, project_id: str) -> dict:
    """Get available filter options for a project (distinct labels, date range).

    Respects the project's detection threshold so only labels with
    at least one visible detection appear as options.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    threshold = project.detection_threshold if project else 0.0

    # Threshold clause: confidence >= threshold OR verified
    threshold_clause = or_(
        Detection.confidence >= threshold,
        Detection.verified == True,  # noqa: E712
    )

    # Distinct labels across threshold-passing detections
    effective_label = func.coalesce(Detection.label, Detection.category)
    label_rows = (
        db.query(effective_label)
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .join(Site, Site.id == Deployment.site_id)
        .filter(Site.project_id == project_id)
        .filter(threshold_clause)
        .distinct()
        .all()
    )
    label_list = sorted([row[0] for row in label_rows if row[0]])

    # Count distinct events per label (threshold-filtered)
    label_count_rows = (
        db.query(effective_label, func.count(func.distinct(Event.id)))
        .join(File, File.id == Detection.file_id)
        .join(event_files, event_files.c.file_id == File.id)
        .join(Event, Event.id == event_files.c.event_id)
        .join(Deployment, Deployment.id == Event.deployment_id)
        .join(Site, Site.id == Deployment.site_id)
        .filter(Site.project_id == project_id)
        .filter(threshold_clause)
        .group_by(effective_label)
        .all()
    )
    label_event_counts = {name: count for name, count in label_count_rows if name}

    # Date range from events
    date_row = (
        db.query(
            func.min(Event.start_time),
            func.max(Event.start_time),
        )
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .first()
    )

    date_range = None
    if date_row and date_row[0] and date_row[1]:
        date_range = {"min": date_row[0], "max": date_row[1]}

    return {
        "labels": label_list,
        "date_range": date_range,
        "label_event_counts": label_event_counts,
    }
