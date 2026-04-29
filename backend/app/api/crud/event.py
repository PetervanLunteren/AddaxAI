"""
CRUD operations for events.

Events are time-clustered groups of files within a deployment.
"""

import uuid
from datetime import datetime, time

from sqlalchemy import Integer, and_, delete, exists, func, insert, or_, select
from sqlalchemy.orm import Session, aliased, joinedload

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, Event, File, Project
from app.models.event import event_files
from app.models.event_observation import EventObservation

logger = get_logger(__name__)

VERIFY_SORT_VALUES = {"newest", "oldest", "random", "cls_low"}

COLLAGE_TILE_LIMIT = 4


def _build_collage_file_ids(
    max_n_frames: list[dict],
    sorted_files: list[File],
) -> list[str]:
    """Pick up to four representative file IDs for an event card collage.

    The first slots come from `max_n_frames` (one file per dominant
    species, computed by `get_max_n_frames`). Remaining slots are
    padded with files of the event ranked by their highest detection
    confidence (descending), skipping files already chosen.
    """
    chosen: list[str] = []
    seen: set[str] = set()
    for mf in max_n_frames:
        fid = mf["file_id"]
        if fid not in seen:
            chosen.append(fid)
            seen.add(fid)
        if len(chosen) >= COLLAGE_TILE_LIMIT:
            return chosen

    def file_top_confidence(f: File) -> float:
        return max((d.confidence for d in f.detections), default=0.0)

    padded = sorted(sorted_files, key=file_top_confidence, reverse=True)
    for f in padded:
        if f.id in seen:
            continue
        chosen.append(f.id)
        seen.add(f.id)
        if len(chosen) >= COLLAGE_TILE_LIMIT:
            break
    return chosen


def _event_min_cls_subquery():
    """Correlated scalar subquery: MIN(label_confidence) across all detections
    on any file in the event. Used for the cls_low sort.
    """
    return (
        select(func.min(Detection.label_confidence))
        .select_from(event_files)
        .join(File, File.id == event_files.c.file_id)
        .join(Detection, Detection.file_id == File.id)
        .where(event_files.c.event_id == Event.id)
        .correlate(Event)
        .scalar_subquery()
    )


def _event_sort_spec(sort: str, seed: int | None):
    """Map (sort, seed) to (sort_key_expression, descending, nulls_last).

    The sort_key is a single SQLAlchemy expression that defines display order
    for the Events tab. The list query orders by it; the adjacent endpoint
    derives next/prev predicates from the same key.
    """
    if sort == "newest":
        return Event.event_start_local, True, False
    if sort == "oldest":
        return Event.event_start_local, False, False
    if sort == "random":
        if seed is None:
            raise ValueError("random sort requires a seed")
        return func.seeded_hash(Event.id, seed), False, False
    if sort == "cls_low":
        return _event_min_cls_subquery(), False, True
    raise ValueError(f"unknown sort: {sort}")


def _sort_order_by_clauses(sort_key, id_col, *, descending: bool, nulls_last: bool):
    """ORDER BY clauses for a sort spec. NULLs go last when nulls_last is set."""
    clauses = []
    if nulls_last:
        # False (non-NULL) sorts before True (NULL) when ascending.
        clauses.append(sort_key.is_(None).asc())
    if descending:
        clauses.extend([sort_key.desc(), id_col.desc()])
    else:
        clauses.extend([sort_key.asc(), id_col.asc()])
    return clauses


def _sort_adjacency_predicates(
    sort_key,
    id_col,
    current_value,
    current_id: str,
    *,
    descending: bool,
    nulls_last: bool,
):
    """Build (next_pred, prev_pred, next_order, prev_order) for adjacency.

    `next` = the row immediately AFTER current in the displayed order.
    `prev` = the row immediately BEFORE current.

    For ASC + nulls_last (cls_low), NULL rows display at the end. The
    predicates handle current-is-NULL and current-is-non-NULL separately.
    """
    if not nulls_last:
        if descending:
            next_pred = (sort_key < current_value) | (
                (sort_key == current_value) & (id_col < current_id)
            )
            prev_pred = (sort_key > current_value) | (
                (sort_key == current_value) & (id_col > current_id)
            )
            next_order = [sort_key.desc(), id_col.desc()]
            prev_order = [sort_key.asc(), id_col.asc()]
        else:
            next_pred = (sort_key > current_value) | (
                (sort_key == current_value) & (id_col > current_id)
            )
            prev_pred = (sort_key < current_value) | (
                (sort_key == current_value) & (id_col < current_id)
            )
            next_order = [sort_key.asc(), id_col.asc()]
            prev_order = [sort_key.desc(), id_col.desc()]
        return next_pred, prev_pred, next_order, prev_order

    # ASC + nulls_last: non-NULL rows ASC, then NULL rows by id ASC.
    if current_value is None:
        # Current is in the NULL bucket. After = NULL rows with larger id.
        # Before = any non-NULL row, or NULL rows with smaller id.
        next_pred = sort_key.is_(None) & (id_col > current_id)
        prev_pred = sort_key.isnot(None) | (
            sort_key.is_(None) & (id_col < current_id)
        )
    else:
        # Current is non-NULL. After = non-NULL rows with larger key, or any
        # NULL row. Before = non-NULL rows with smaller key.
        next_pred = (
            (sort_key.isnot(None) & (sort_key > current_value))
            | (sort_key.isnot(None) & (sort_key == current_value) & (id_col > current_id))
            | sort_key.is_(None)
        )
        prev_pred = sort_key.isnot(None) & (
            (sort_key < current_value)
            | ((sort_key == current_value) & (id_col < current_id))
        )

    next_order = [sort_key.is_(None).asc(), sort_key.asc(), id_col.asc()]
    prev_order = [sort_key.is_(None).desc(), sort_key.desc(), id_col.desc()]
    return next_pred, prev_pred, next_order, prev_order


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
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
):
    """Apply shared filters to an event query. Expects Event already joined to Deployment.

    `flagged` / `favorited` filter at the file level (EXISTS a file in the
    event with File.flagged / File.favorited set). Per the decided mental
    model, flag and heart live on files; an event is flagged only in the
    sense that it contains at least one flagged file.

    `min_label_confidence` / `max_label_confidence` filter on the
    classifier score (Detection.label_confidence). NULL classifications
    are excluded automatically when the bounds are set (correct
    behaviour: a NULL cannot satisfy a range).

    `project_floor` is the project's `detection_threshold`, applied as
    `(Detection.confidence >= floor OR Detection.verified == True)`. This
    is the global override rule. `min_confidence` (the user's slider) is
    applied LITERALLY without OR-verified — a verified low-confidence
    detection passes the floor but cannot satisfy a narrower user filter.

    `empty`:
      - "show_only": every file in the event has observation_type == "blank"
        (i.e. NOT EXISTS a non-blank file). Skips the labels and confidence
        gates because empty events have no detections by definition.
      - "hide": at least one file is non-blank
      - any other value (or None): no filter
    """
    from app.api.crud.deployment import site_ids_filter

    if empty == "show_only":
        labels = None
        min_confidence = None
        max_confidence = None
        min_label_confidence = None
        max_label_confidence = None
        project_floor = None

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
        if project_floor is not None:
            label_subq = label_subq.where(
                or_(Detection.confidence >= project_floor, Detection.verified == True)  # noqa: E712
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
        min_confidence is not None
        or max_confidence is not None
        or min_label_confidence is not None
        or max_label_confidence is not None
        or project_floor is not None
    ):
        # Standalone confidence filter: event has at least one detection in range
        conf_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .join(Detection, Detection.file_id == File.id)
            .where(event_files.c.event_id == Event.id)
        )
        if project_floor is not None:
            conf_subq = conf_subq.where(
                or_(Detection.confidence >= project_floor, Detection.verified == True)  # noqa: E712
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
        query = query.filter(exists(conf_subq))

    if verification in ("verified", "unverified"):
        # AddaxAI rule: an event is verified when all its MaxN frames are
        # verified; for blank events (no MaxN) the fallback is "any file
        # verified". Matches the is_verified summary field and the stats
        # bar, so the three surfaces always agree.
        UnverifiedMaxNFile = aliased(File)
        unverified_maxn_exists = exists(
            select(EventObservation.id)
            .join(
                UnverifiedMaxNFile,
                UnverifiedMaxNFile.id == EventObservation.max_n_file_id,
            )
            .where(EventObservation.event_id == Event.id)
            .where(EventObservation.max_n_file_id.isnot(None))
            .where(UnverifiedMaxNFile.verified == False)  # noqa: E712
        )
        any_maxn_exists = exists(
            select(EventObservation.id).where(
                and_(
                    EventObservation.event_id == Event.id,
                    EventObservation.max_n_file_id.isnot(None),
                )
            )
        )
        AnyVerifiedFile = aliased(File)
        any_verified_file_exists = exists(
            select(event_files.c.event_id)
            .join(AnyVerifiedFile, AnyVerifiedFile.id == event_files.c.file_id)
            .where(event_files.c.event_id == Event.id)
            .where(AnyVerifiedFile.verified == True)  # noqa: E712
        )
        verified_clause = or_(
            and_(any_maxn_exists, ~unverified_maxn_exists),
            and_(~any_maxn_exists, any_verified_file_exists),
        )
        if verification == "verified":
            query = query.filter(verified_clause)
        else:
            query = query.filter(~verified_clause)

    if flagged in ("flagged", "not_flagged"):
        flagged_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .where(event_files.c.event_id == Event.id)
            .where(File.flagged == True)  # noqa: E712
        )
        if flagged == "flagged":
            query = query.filter(exists(flagged_subq))
        else:
            query = query.filter(~exists(flagged_subq))

    if favorited in ("favorited", "not_favorited"):
        favorited_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .where(event_files.c.event_id == Event.id)
            .where(File.favorited == True)  # noqa: E712
        )
        if favorited == "favorited":
            query = query.filter(exists(favorited_subq))
        else:
            query = query.filter(~exists(favorited_subq))

    if empty in ("show_only", "hide"):
        non_blank_file_subq = (
            select(event_files.c.event_id)
            .join(File, File.id == event_files.c.file_id)
            .where(event_files.c.event_id == Event.id)
            .where(File.observation_type != "blank")
        )
        if empty == "show_only":
            # All files blank → no non-blank file exists
            query = query.filter(~exists(non_blank_file_subq))
        else:
            # At least one non-blank file
            query = query.filter(exists(non_blank_file_subq))

    return query


def generate_events_for_project(db: Session, project_id: str) -> int:
    """
    Generate events for all deployments in a project.

    Idempotent: deletes existing events before regenerating. Clustering
    logic lives in `app.services.event_clustering.cluster_files_into_events`
    — the single source of truth shared with the smoothing adapter, so
    events and smoother inputs never disagree.

    Returns total event count created.
    """
    from app.models import Project
    from app.services.event_clustering import cluster_files_into_events

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

        for cluster in cluster_files_into_events(files, independence_interval):
            _create_event(db, deployment.id, cluster)
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
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
    sort: str = "newest",
    seed: int | None = None,
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
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        project_floor=project_floor,
    )
    sort_key, descending, nulls_last = _event_sort_spec(sort, seed)
    order_clauses = _sort_order_by_clauses(
        sort_key, Event.id, descending=descending, nulls_last=nulls_last,
    )
    events = (
        query.options(joinedload(Event.files).joinedload(File.detections))
        .order_by(*order_clauses)
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
                meets_floor = (
                    project_floor is None
                    or d.confidence >= project_floor
                    or d.verified
                )
                meets_min = (
                    min_confidence is None or d.confidence >= min_confidence
                )
                meets_max = (
                    max_confidence is None or d.confidence <= max_confidence
                )
                if meets_floor and meets_min and meets_max:
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
        any_file_flagged = any(f.flagged for f in sorted_files)
        any_file_favorited = any(f.favorited for f in sorted_files)

        # MaxN verification counts
        max_n_frames = max_n_by_event.get(event.id, [])
        file_verified_map = {f.id: f.verified for f in sorted_files}
        total_maxn_count = len(max_n_frames)
        verified_maxn_count = sum(
            1
            for mf in max_n_frames
            if file_verified_map.get(mf["file_id"], False)
        )

        # Event verification (AddaxAI rule): all MaxN frames verified.
        # Blank events (no MaxN) fall back to "any file verified" so they
        # require an explicit user confirmation rather than auto-completing.
        if total_maxn_count > 0:
            is_verified = verified_maxn_count == total_maxn_count
        else:
            is_verified = any(f.verified for f in sorted_files)

        # MaxN-derived thumbnail: dominant species' MaxN frame, fallback to first file
        thumbnail_file_id = max_n_frames[0]["file_id"] if max_n_frames else (
            sorted_files[0].id if sorted_files else None
        )

        collage_file_ids = _build_collage_file_ids(max_n_frames, sorted_files)

        summaries.append(
            {
                "id": event.id,
                "deployment_id": event.deployment_id,
                "event_start_local": event.event_start_local,
                "event_end_local": event.event_end_local,
                "file_count": event.file_count,
                "thumbnail_file_id": thumbnail_file_id,
                "collage_file_ids": collage_file_ids,
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
                "is_verified": is_verified,
                "any_file_flagged": any_file_flagged,
                "any_file_favorited": any_file_favorited,
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
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
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
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        project_floor=project_floor,
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
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
    sort: str = "newest",
    seed: int | None = None,
) -> dict:
    """
    Get adjacent event IDs for navigation.

    Returns previous_id, next_id, next_unverified_id, current_index, total_count.
    Order matches `get_events_by_project` for the same `(sort, seed)`. The
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

    sort_key, descending, nulls_last = _event_sort_spec(sort, seed)

    # 1. Get current event's sort key value.
    current_value = (
        db.query(sort_key).filter(Event.id == event_id).scalar()
    )
    current = db.query(Event.id).filter(Event.id == event_id).first()
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
        sort_key, Event.id, current_value, cid,
        descending=descending, nulls_last=nulls_last,
    )

    def base():
        q = (
            db.query(Event.id)
            .join(Deployment)
            .filter(Deployment.project_id == project_id)
        )
        return _apply_event_filters(q, db, **filter_kwargs)

    # 2. Previous (one row above current in display order).
    prev = base().filter(prev_pred).order_by(*prev_order).first()

    # 3. Next (one row below current in display order).
    nxt = base().filter(next_pred).order_by(*next_order).first()

    # 4. Next unverified (next row in display order with at least one
    # unverified file). Uses base() so all active filters apply.
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
        .filter(next_pred)
        .filter(exists(unv_subq))
        .order_by(*next_order)
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

    # 5b. Current index = number of rows that come BEFORE current in display
    # order. This is the prev_pred set, plus current itself if you want
    # 1-based; we use 0-based to match the existing UI.
    idx_q = (
        db.query(func.count(Event.id))
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .filter(prev_pred)
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
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    project_floor: float | None = None,
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
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        project_floor=project_floor,
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

    # Per-detection counts for the Observations verification progress.
    # Denominator and numerator share the same filter so the ratio is
    # meaningful (sharing it with `total_observations`, which is a MaxN
    # sum, would give nonsense like 23 / 12).
    det_total_q = (
        db.query(func.count(Detection.id))
        .join(File, File.id == Detection.file_id)
        .join(event_files, event_files.c.file_id == File.id)
        .filter(event_files.c.event_id.in_(select(event_ids_subq.c.id)))
    )
    det_verified_q = (
        db.query(
            func.sum(func.cast(Detection.verified, Integer)),
        )
        .join(File, File.id == Detection.file_id)
        .join(event_files, event_files.c.file_id == File.id)
        .filter(event_files.c.event_id.in_(select(event_ids_subq.c.id)))
    )
    if project_floor is not None:
        floor_clause = or_(
            Detection.confidence >= project_floor,
            Detection.verified == True,  # noqa: E712
        )
        det_total_q = det_total_q.filter(floor_clause)
        det_verified_q = det_verified_q.filter(floor_clause)
    if min_confidence is not None:
        det_total_q = det_total_q.filter(Detection.confidence >= min_confidence)
        det_verified_q = det_verified_q.filter(Detection.confidence >= min_confidence)
    total_detections = int(det_total_q.scalar() or 0)
    verified_detections = int(det_verified_q.scalar() or 0)

    # Event-level verification: total and fully-verified counts.
    # AddaxAI rule: an event is verified when all its MaxN frames are
    # verified, or (for blank events with no MaxN) when any file in it
    # is verified.
    events_total_q = db.query(func.count(Event.id)).filter(
        Event.id.in_(select(event_ids_subq.c.id))
    )
    events_total = events_total_q.scalar() or 0

    UnverifiedMaxNFile = aliased(File)
    unverified_maxn_exists = exists(
        select(EventObservation.id)
        .join(
            UnverifiedMaxNFile,
            UnverifiedMaxNFile.id == EventObservation.max_n_file_id,
        )
        .where(EventObservation.event_id == Event.id)
        .where(EventObservation.max_n_file_id.isnot(None))
        .where(UnverifiedMaxNFile.verified == False)  # noqa: E712
    )
    any_maxn_exists = exists(
        select(EventObservation.id).where(
            and_(
                EventObservation.event_id == Event.id,
                EventObservation.max_n_file_id.isnot(None),
            )
        )
    )
    AnyVerifiedFile = aliased(File)
    any_verified_file_exists = exists(
        select(event_files.c.event_id)
        .join(AnyVerifiedFile, AnyVerifiedFile.id == event_files.c.file_id)
        .where(event_files.c.event_id == Event.id)
        .where(AnyVerifiedFile.verified == True)  # noqa: E712
    )
    events_fully_verified_clause = or_(
        and_(any_maxn_exists, ~unverified_maxn_exists),
        and_(~any_maxn_exists, any_verified_file_exists),
    )
    events_fully_verified_q = (
        db.query(func.count(Event.id))
        .filter(Event.id.in_(select(event_ids_subq.c.id)))
        .filter(events_fully_verified_clause)
    )
    events_fully_verified = events_fully_verified_q.scalar() or 0

    return {
        "events_fully_verified": events_fully_verified,
        "events_total": events_total,
        "total_files": file_stats[0] or 0,
        "verified_files": int(file_stats[1] or 0),
        "total_max_n_frames": maxn_stats[0] or 0,
        "verified_max_n_frames": int(maxn_stats[1] or 0),
        "total_observations": total_observations,
        "total_detections": total_detections,
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
