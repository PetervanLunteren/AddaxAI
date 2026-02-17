"""
CRUD operations for events.

Events are time-clustered groups of files within a deployment.
"""

import uuid
from datetime import datetime

from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session, joinedload

from app.models import Deployment, Event, File, Site
from app.models.event import event_files


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
            .filter(File.deployment_id == deployment.id)
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
) -> list[dict]:
    """
    Get event summaries for a project.

    Returns event data with representative file, species list,
    observation type, and verification progress.
    """
    events = (
        db.query(Event)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
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

        # Find representative file: file with highest sum of animal detection confidences
        representative_file = sorted_files[0] if sorted_files else None
        best_score = -1.0
        for f in sorted_files:
            score = sum(
                d.confidence for d in f.detections if d.category == "animal"
            )
            if score > best_score:
                best_score = score
                representative_file = f

        # Collect unique species across all files
        species_set: set[str] = set()
        for f in sorted_files:
            for d in f.detections:
                if d.species:
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

        # Count verified files
        verified_count = sum(1 for f in sorted_files if f.verified)

        summaries.append({
            "id": event.id,
            "deployment_id": event.deployment_id,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "file_count": event.file_count,
            "representative_file_id": representative_file.id if representative_file else None,
            "species": sorted(species_set),
            "observation_type": dominant_type,
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
        .filter(Event.id == event_id)
        .first()
    )
    return event


def get_event_count_by_project(db: Session, project_id: str) -> int:
    """Get total event count for a project."""
    count = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .scalar()
    )
    return count or 0


def get_adjacent_events(
    db: Session, event_id: str, project_id: str
) -> dict:
    """
    Get adjacent event IDs for navigation.

    Returns previous_id, next_id, next_unverified_id, current_index, total_count.
    Events ordered by start_time DESC (newest first).

    Uses targeted SQL queries instead of loading all events into memory.
    """
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
        return (
            db.query(Event.id)
            .join(Deployment)
            .join(Site)
            .filter(Site.project_id == project_id)
        )

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
    total = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .scalar()
    ) or 0

    # 5b. Current index (number of events newer than current = position in DESC list)
    idx = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .filter(
            (Event.start_time > ct)
            | ((Event.start_time == ct) & (Event.id > cid))
        )
        .scalar()
    ) or 0

    return {
        "previous_id": prev[0] if prev else None,
        "next_id": nxt[0] if nxt else None,
        "next_unverified_id": nxt_unv[0] if nxt_unv else None,
        "current_index": idx,
        "total_count": total,
    }
