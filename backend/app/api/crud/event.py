"""
CRUD operations for events.

Events are time-clustered groups of files within a deployment.
"""

import uuid
from datetime import datetime, time, timedelta

import cv2
from sqlalchemy import and_, delete, exists, func, insert, select
from sqlalchemy.orm import Session, aliased, joinedload, subqueryload

from app.core.logging_config import get_logger
from app.ml.scoring import compute_sharpness, pick_best_candidate, score_detections
from app.models import Deployment, Detection, Event, File, Site
from app.models.event import event_files

logger = get_logger(__name__)


def _apply_event_filters(
    query,
    db: Session,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    species: list[str] | None = None,
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
        end_of_day = datetime.combine(date_to.date(), time.max) if isinstance(date_to, datetime) else date_to
        query = query.filter(Event.start_time <= end_of_day)

    if species:
        # EXISTS subquery: event has at least one file with a detection matching species
        species_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .join(Detection, Detection.file_id == File.id)
            .where(event_files.c.event_id == Event.id)
            .where(Detection.species.in_(species))
        )
        if min_confidence is not None:
            species_subq = species_subq.where(Detection.confidence >= min_confidence)
        if max_confidence is not None:
            species_subq = species_subq.where(Detection.confidence <= max_confidence)
        query = query.filter(exists(species_subq))
    elif min_confidence is not None or max_confidence is not None:
        # Standalone confidence filter: event has at least one detection in range
        conf_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .join(Detection, Detection.file_id == File.id)
            .where(event_files.c.event_id == Event.id)
        )
        if min_confidence is not None:
            conf_subq = conf_subq.where(Detection.confidence >= min_confidence)
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
        elif verification == "unverified_representative":
            # Representative file is unverified
            RepFile = aliased(File)
            query = query.filter(exists(
                select(RepFile.id).where(
                    and_(
                        RepFile.id == Event.representative_file_id,
                        RepFile.verified == False,  # noqa: E712
                    )
                )
            ))
        elif verification == "verified_representative":
            # Representative file is verified
            RepFile = aliased(File)
            query = query.filter(exists(
                select(RepFile.id).where(
                    and_(
                        RepFile.id == Event.representative_file_id,
                        RepFile.verified == True,  # noqa: E712
                    )
                )
            ))
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


def _select_representative_file(files: list[File]) -> str | None:
    """
    Pick the best representative file from an event's files using shared scoring.

    Uses detection confidence as primary signal and image sharpness as tiebreaker.
    """
    if not files:
        return None

    # Build detection tuples: (file_id, confidence, bbox)
    det_tuples = [
        (file.id, det.confidence, (det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height))
        for file in files
        for det in file.detections
    ]

    scores = score_detections(det_tuples)

    total_dets = sum(len(f.detections) for f in files)
    logger.debug(
        f"Representative selection: {len(files)} files, "
        f"{total_dets} total detections, {len(det_tuples)} with bbox, "
        f"{len(scores)} scored above threshold"
    )

    # Build sharpness callback that reads images from disk
    def get_sharpest(keys: list[str]) -> str:
        file_map = {f.id: f for f in files}
        best_key = keys[0]
        best_sharpness = -1.0

        for key in keys:
            f = file_map.get(key)
            if not f:
                continue

            # For video files, use the extracted best frame; for images, use the file itself
            image_path = f.best_frame_path if f.file_type == "video" else f.file_path
            if not image_path:
                continue

            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                sharpness = compute_sharpness(img_rgb)
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_key = key
            except Exception:
                continue

        return best_key

    fallback_keys = [f.id for f in files]

    result = pick_best_candidate(
        scores,
        get_sharpest=get_sharpest,
        fallback_keys=fallback_keys,
    )

    # Ultimate fallback
    return result if result is not None else files[0].id


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
    deployments = (
        db.query(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .all()
    )

    # Delete existing events for all deployments in this project
    deployment_ids = [d.id for d in deployments]
    if deployment_ids:
        db.execute(
            delete(Event).where(Event.deployment_id.in_(deployment_ids))
        )

    total_events = 0

    for deployment in deployments:
        files = (
            db.query(File)
            .options(subqueryload(File.detections))
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

    db.commit()
    return total_events


def _create_event(db: Session, deployment_id: str, files: list[File]) -> Event:
    """Create an event with its junction table entries."""
    representative_file_id = _select_representative_file(files)

    event = Event(
        id=str(uuid.uuid4()),
        deployment_id=deployment_id,
        start_time=files[0].timestamp,
        end_time=files[-1].timestamp,
        file_count=len(files),
        representative_file_id=representative_file_id,
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
    species: list[str] | None = None,
    verification: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> list[dict]:
    """
    Get event summaries for a project.

    Returns event data with representative file, species list,
    observation type, and verification progress.
    """
    query = (
        db.query(Event)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
    )
    query = _apply_event_filters(
        query, db,
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        species=species,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    events = (
        query
        .options(joinedload(Event.files).joinedload(File.detections))
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

    summaries = []
    for event in unique_events:
        # Sort files by sequence within event
        sorted_files = sorted(
            event.files,
            key=lambda f: f.timestamp,
        )

        # Collect unique species across all files
        species_set: set[str] = set()
        for f in sorted_files:
            for d in f.detections:
                if d.species and (min_confidence is None or d.confidence >= min_confidence) and (max_confidence is None or d.confidence <= max_confidence):
                    species_set.add(d.species)

        # Determine dominant observation type (animal > human > vehicle > blank)
        obs_priority = {"animal": 4, "human": 3, "vehicle": 2, "blank": 1}
        dominant_type = "blank"
        dominant_priority = 0
        for f in sorted_files:
            p = obs_priority.get(f.observation_type, 0)
            if p > dominant_priority:
                dominant_priority = p
                dominant_type = f.observation_type

        # Count files by type and verification
        image_count = sum(1 for f in sorted_files if f.file_type == "image")
        frame_count = sum(1 for f in sorted_files if f.file_type == "frame")
        video_count = len({f.source_video_id for f in sorted_files if f.file_type == "frame" and f.source_video_id})
        verified_count = sum(1 for f in sorted_files if f.verified)

        summaries.append({
            "id": event.id,
            "deployment_id": event.deployment_id,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "file_count": event.file_count,
            "representative_file_id": event.representative_file_id,
            "site_name": event.deployment.site.name if event.deployment and event.deployment.site else None,
            "species": sorted(species_set),
            "observation_type": dominant_type,
            "image_count": image_count,
            "frame_count": frame_count,
            "video_count": video_count,
            "verified_count": verified_count,
            "total_count": len(sorted_files),
        })

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
    species: list[str] | None = None,
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
        query, db,
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        species=species,
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
    species: list[str] | None = None,
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
        species=species,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    # 1. Get current event's start_time
    current = (
        db.query(Event.id, Event.start_time)
        .filter(Event.id == event_id)
        .first()
    )
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
        q = (
            db.query(Event.id)
            .join(Deployment)
            .join(Site)
            .filter(Site.project_id == project_id)
        )
        return _apply_event_filters(q, db, **filter_kwargs)

    # 2. Previous (newer in DESC order): start_time > current, or same time + higher id
    prev = (
        base()
        .filter(
            (Event.start_time > ct)
            | ((Event.start_time == ct) & (Event.id > cid))
        )
        .order_by(Event.start_time.asc(), Event.id.asc())
        .first()
    )

    # 3. Next (older in DESC order): start_time < current, or same time + lower id
    nxt = (
        base()
        .filter(
            (Event.start_time < ct)
            | ((Event.start_time == ct) & (Event.id < cid))
        )
        .order_by(Event.start_time.desc(), Event.id.desc())
        .first()
    )

    # 4. Next unverified (older, with at least one unverified file)
    nxt_unv = (
        base()
        .join(event_files, Event.id == event_files.c.event_id)
        .join(File, File.id == event_files.c.file_id)
        .filter(
            (Event.start_time < ct)
            | ((Event.start_time == ct) & (Event.id < cid))
        )
        .filter(File.verified == False)  # noqa: E712
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
        .filter(
            (Event.start_time > ct)
            | ((Event.start_time == ct) & (Event.id > cid))
        )
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


def get_filter_options(db: Session, project_id: str) -> dict:
    """Get available filter options for a project (distinct species, date range)."""
    # Distinct species across all detections in project
    species_rows = (
        db.query(Detection.species)
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .join(Site, Site.id == Deployment.site_id)
        .filter(Site.project_id == project_id)
        .filter(Detection.species.isnot(None))
        .distinct()
        .all()
    )
    species_list = sorted([row[0] for row in species_rows])

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

    return {"species": species_list, "date_range": date_range}
