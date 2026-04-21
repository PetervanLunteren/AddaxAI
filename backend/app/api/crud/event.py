"""
CRUD operations for events.

Events are time-clustered groups of files within a deployment.
"""

import uuid
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path

from sqlalchemy import Integer, and_, delete, exists, func, insert, or_, select
from sqlalchemy.orm import Session, aliased, joinedload

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, Event, File, Project
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
    """Apply shared filters to an event query. Expects Event already joined to Deployment."""
    from app.api.crud.deployment import site_ids_filter

    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.filter(site_clause)

    if date_from is not None:
        query = query.filter(Event.event_start_local >= date_from)

    if date_to is not None:
        # Include the entire end-of-day
        end_of_day = (
            datetime.combine(date_to.date(), time.max) if isinstance(date_to, datetime) else date_to
        )
        query = query.filter(Event.event_start_local <= end_of_day)

    if labels:
        # EXISTS subquery: event has at least one file with a detection
        # matching the given labels. Labels are taxonomy UUIDs when
        # available, falling back to string matching.
        label_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .join(Detection, Detection.file_id == File.id)
            .where(event_files.c.event_id == Event.id)
            .where(Detection.label_taxonomy_id.in_(labels))
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


def _folder_key(f: File) -> str:
    """
    Return the clustering folder for a file.

    For images, it's the file's own parent directory. For extracted video
    frames (`file_type='frame'`), the frame itself lives inside
    `.addaxai/video_frames/...`, which is a pipeline artifact path — not
    where the camera actually was. Fall back to the source video's parent
    so frames of one video cluster with images shot at the same camera.

    A frame row without a source_video (shouldn't happen in healthy data)
    falls back to its own file_path, which at worst over-splits by one.
    """
    if f.file_type == "frame" and f.source_video is not None:
        return str(Path(f.source_video.file_path).parent)
    return str(Path(f.file_path).parent)


def generate_events_for_project(db: Session, project_id: str) -> int:
    """
    Generate events for all deployments in a project.

    Idempotent: deletes existing events before regenerating.

    1. Fetch project's independence_interval
    2. Delete all existing events for every deployment in the project
    3. For each deployment, query files ordered by timestamp ASC
    4. Walk files: start a new event when *either* the gap exceeds
       `independence_interval` *or* the next file lives in a different
       folder than the previous one. The folder check means that when a
       user runs a backlog of multiple SD cards as one deployment, events
       never bridge across SD-card folders, even if timestamps happen to
       overlap. Frame rows use their source video's parent as the folder.
    5. Create Event records with event_start_local, event_end_local, file_count
    6. Insert event_files junction rows with sequence_number

    Returns total event count created.
    """
    from app.models import Project

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    independence_interval = project.independence_interval  # seconds

    # Get all deployments for this project
    deployments = (
        db.query(Deployment).filter(Deployment.project_id == project_id).all()
    )

    # Delete existing events for all deployments in this project
    deployment_ids = [d.id for d in deployments]
    if deployment_ids:
        db.execute(delete(Event).where(Event.deployment_id.in_(deployment_ids)))

    total_events = 0

    for deployment in deployments:
        files = (
            db.query(File)
            .options(joinedload(File.source_video))
            .filter(File.deployment_id == deployment.id)
            .filter(File.file_type.in_(["image", "frame"]))
            .all()
        )

        if not files:
            continue

        # Bucket by folder first, then cluster each bucket by time. A
        # single linear walk over (time-sorted) files would create one
        # event per file when timestamps interleave across folders
        # (two cameras firing in parallel). Bucketing first means each
        # folder's clustering is independent of the others.
        by_folder: dict[str, list[File]] = defaultdict(list)
        for f in files:
            by_folder[_folder_key(f)].append(f)

        # Iterate folders in a deterministic order so tests and logs
        # are stable; event ordering within the deployment is
        # downstream-sorted by event_start_local anyway.
        for folder_key in sorted(by_folder):
            folder_files = sorted(
                by_folder[folder_key], key=lambda f: f.captured_at_local
            )
            current_event_files: list[File] = [folder_files[0]]
            for i in range(1, len(folder_files)):
                gap = (
                    folder_files[i].captured_at_local
                    - folder_files[i - 1].captured_at_local
                ).total_seconds()
                if gap > independence_interval:
                    _create_event(db, deployment.id, current_event_files)
                    total_events += 1
                    current_event_files = [folder_files[i]]
                else:
                    current_event_files.append(folder_files[i])
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
        event_start_local=files[0].captured_at_local,
        event_end_local=files[-1].captured_at_local,
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
    query = (
        db.query(Event)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
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
    events = (
        query.options(joinedload(Event.files).joinedload(File.detections))
        .order_by(Event.event_start_local.desc())
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
            key=lambda f: f.captured_at_local,
        )

        # Collect unique taxonomy IDs across all files
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
                    tid = d.label_taxonomy_id
                    if tid:
                        label_set.add(tid)
                        display = d.display_name or d.label or d.category
                        if display and tid not in label_to_display:
                            label_to_display[tid] = display
                    else:
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
                "event_start_local": event.event_start_local,
                "event_end_local": event.event_end_local,
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
        .filter(Deployment.project_id == project_id)
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
    Events ordered by event_start_local DESC (newest first).

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

    # 1. Get current event's local start time
    current = db.query(Event.id, Event.event_start_local).filter(Event.id == event_id).first()
    if not current:
        return {
            "previous_id": None,
            "next_id": None,
            "next_unverified_id": None,
            "current_index": 0,
            "total_count": 0,
        }

    ct = current.event_start_local
    cid = current.id

    def base():
        q = (
            db.query(Event.id)
            .join(Deployment)
            .filter(Deployment.project_id == project_id)
        )
        return _apply_event_filters(q, db, **filter_kwargs)

    newer_than_current = (Event.event_start_local > ct) | (
        (Event.event_start_local == ct) & (Event.id > cid)
    )
    older_than_current = (Event.event_start_local < ct) | (
        (Event.event_start_local == ct) & (Event.id < cid)
    )

    # 2. Previous (newer in DESC order): event_start_local > current, or same time + higher id
    prev = (
        base()
        .filter(newer_than_current)
        .order_by(Event.event_start_local.asc(), Event.id.asc())
        .first()
    )

    # 3. Next (older in DESC order): event_start_local < current, or same time + lower id
    nxt = (
        base()
        .filter(older_than_current)
        .order_by(Event.event_start_local.desc(), Event.id.desc())
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
        .filter(older_than_current)
        .filter(exists(unv_subq))
        .order_by(Event.event_start_local.desc(), Event.id.desc())
        .first()
    )

    # 5a. Total count
    total_q = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
    )
    total_q = _apply_event_filters(total_q, db, **filter_kwargs)
    total = total_q.scalar() or 0

    # 5b. Current index (number of events newer than current = position in DESC list)
    idx_q = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .filter(newer_than_current)
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
        db.query(Event.id)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
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

    # Distinct taxonomy IDs across threshold-passing detections
    label_rows = (
        db.query(
            Detection.label_taxonomy_id,
            func.coalesce(
                Detection.display_name, Detection.label, Detection.category
            ),
        )
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(Detection.label_taxonomy_id.isnot(None))
        .filter(threshold_clause)
        .distinct()
        .all()
    )
    label_list = sorted([row[0] for row in label_rows if row[0]])

    # Build display_labels mapping (taxonomy_id -> display name)
    display_labels: dict[str, str] = {}
    for tid, display in label_rows:
        if tid and display and tid not in display_labels:
            display_labels[tid] = display

    # Count distinct events per taxonomy ID (threshold-filtered)
    label_count_rows = (
        db.query(
            Detection.label_taxonomy_id,
            func.count(func.distinct(Event.id)),
        )
        .join(File, File.id == Detection.file_id)
        .join(event_files, event_files.c.file_id == File.id)
        .join(Event, Event.id == event_files.c.event_id)
        .join(Deployment, Deployment.id == Event.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(Detection.label_taxonomy_id.isnot(None))
        .filter(threshold_clause)
        .group_by(Detection.label_taxonomy_id)
        .all()
    )
    label_event_counts = {
        tid: count for tid, count in label_count_rows if tid
    }

    # Date range from events
    date_row = (
        db.query(
            func.min(Event.event_start_local),
            func.max(Event.event_start_local),
        )
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .first()
    )

    date_range = None
    if date_row and date_row[0] and date_row[1]:
        date_range = {"min": date_row[0], "max": date_row[1]}

    return {
        "labels": label_list,
        "date_range": date_range,
        "label_event_counts": label_event_counts,
        "display_labels": display_labels,
    }
