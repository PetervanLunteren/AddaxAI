"""
MaxN calculation and storage for event observations.

MaxN is the maximum number of individuals of a species visible in any
single image within an event. Calculated per-species, stored in the
event_observations table.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import delete, func, or_
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.label_exclusion import is_a_real_detection
from app.models import Deployment, Detection, Event, File, Project
from app.models.event import event_files
from app.models.event_observation import EventObservation

logger = get_logger(__name__)


@dataclass
class PriorObs:
    """A snapshot of one prior EventObservation, enough to carry the human
    layer (human_count, human-only rows) and detect an effective-set change.
    Its attribute names mirror EventObservation so `calculate_max_n_for_event`
    can read either interchangeably. Used to carry the human layer of a
    deleted event onto its regenerated replacement (same file set)."""

    label: str | None
    label_taxonomy_id: str | None
    category: str
    human_count: int | None
    effective_count: int


def _threshold_clause(threshold: float):
    """Detection threshold filter: confidence >= threshold OR verified."""
    return or_(
        Detection.confidence >= threshold,
        Detection.verified == True,  # noqa: E712
    )


def calculate_max_n_for_event(
    db: Session,
    event_id: str,
    counting_threshold: float,
    prior: list[PriorObs] | None = None,
) -> list[EventObservation]:
    """
    Recalculate the AI-derived MaxN per species for a single event, while
    preserving the human-authoritative data layered on top.

    `prior` supplies the human layer to carry when the event was just
    (re)created and has no rows of its own yet, e.g. during an interval
    change where a deleted event's replacement covers the same files. When
    None, the event's own current rows are used (the normal relabel/verify
    path).

    Algorithm:
    1. Count detections per (file, frame, taxonomy) and take the maximum
       count per species (= MaxN). Ties break on summed confidence.
       For videos, a species only counts if it appears on the video's best
       frame (or was verified on some frame): non-best-frame labels are
       per-frame classifier noise the user can't see or clean in the Labels
       step, so they must not spawn spurious species rows.
    2. Rebuild the event's `event_observations` rows: one per AI species
       (carrying forward any human_count set for it) plus any human-only
       species the AI did not detect this round (max_n=0, no frame).
    3. Clear `Event.confirmed` when the effective species/count set changed
       (a Labels-page relabel/add/delete that moves a count un-signs the
       event); a pure detection-verify that leaves counts unchanged does not.

    The human layer (`human_count`, human-only rows) is keyed by
    `label_taxonomy_id or label` so it survives the delete-and-rebuild.

    Returns the rebuilt EventObservation rows.
    """
    # Group by label_taxonomy_id (authoritative), falling back to
    # COALESCE(label, category) for display string.
    effective_label = func.coalesce(Detection.label, Detection.category)

    # Count detections per (file, frame, taxonomy) so a video that shows
    # 4 animals in frame A and 4 in frame B reports MaxN=4 instead of
    # collapsing into MaxN=8. For image rows, `frame_number` is NULL and
    # all detections in one file land in a single group (NULL == NULL in
    # GROUP BY semantics), preserving the legacy per-image behaviour.
    # file_type / best_frame_number / any_verified ride along to gate video
    # species to the best frame (below). They are constant per file_id, so
    # adding the two columns to GROUP BY doesn't change the grouping.
    counts = (
        db.query(
            Detection.file_id,
            Detection.frame_number,
            Detection.label_taxonomy_id,
            effective_label.label("eff_label"),
            Detection.category,
            func.count(Detection.id).label("det_count"),
            func.sum(Detection.confidence).label("conf_sum"),
            File.file_type,
            File.best_frame_number,
            func.max(Detection.verified).label("any_verified"),
        )
        .join(File, File.id == Detection.file_id)
        .join(event_files, event_files.c.file_id == File.id)
        .filter(event_files.c.event_id == event_id)
        .filter(_threshold_clause(counting_threshold))
        .filter(is_a_real_detection())
        .group_by(
            Detection.file_id,
            Detection.frame_number,
            Detection.label_taxonomy_id,
            effective_label,
            Detection.category,
            File.file_type,
            File.best_frame_number,
        )
        .all()
    )

    # Snapshot the existing rows so the human layer survives the rebuild
    # and so we can detect whether the effective set changed. A caller can
    # pass `prior` instead (a freshly recreated event has no rows of its own).
    existing = (
        prior
        if prior is not None
        else db.query(EventObservation)
        .filter(EventObservation.event_id == event_id)
        .all()
    )
    prior_human: dict[str, int] = {}
    prior_effective: dict[str, int] = {}
    for r in existing:
        key = r.label_taxonomy_id or r.label
        if key is None:
            continue
        if r.human_count is not None:
            prior_human[key] = r.human_count
        prior_effective[key] = r.effective_count

    # A video species is only suggested if it appears on the video's best
    # frame (the canonical, user-cleanable view) or was verified on some
    # frame. Non-best-frame-only labels are per-frame classifier noise the
    # user can't see or clean in the Labels step, so they must not spawn
    # spurious species rows. Images are never gated (every image detection
    # is visible and cleanable).
    allowed_video_keys: dict[str, set[str]] = defaultdict(set)
    for r in counts:
        if r.file_type != "video":
            continue
        key = r.label_taxonomy_id or r.eff_label
        on_best = (
            r.best_frame_number is not None
            and r.frame_number == r.best_frame_number
        )
        if on_best or r.any_verified:
            allowed_video_keys[r.file_id].add(key)

    # Find MaxN per taxonomy_id (or label string as fallback key). The
    # winning row's `file_id` is stored as max_n_file_id; for videos
    # this is the parent video File row regardless of which frame
    # within it contributed the MaxN count. The UI surfaces the video's
    # `best_frame_path` thumbnail, which is the canonical representative
    # of the file. A gated video species (best-frame absent, unverified) is
    # skipped, but an allowed species still takes its peak across all frames.
    max_n_per_key: dict[str, dict] = {}
    for r in counts:
        key = r.label_taxonomy_id or r.eff_label
        if r.file_type == "video" and key not in allowed_video_keys[r.file_id]:
            continue
        new_score = (r.det_count, r.conf_sum)
        prev = max_n_per_key.get(key)
        if prev is None or new_score > (prev["count"], prev["conf_sum"]):
            max_n_per_key[key] = {
                "count": r.det_count,
                "conf_sum": r.conf_sum,
                "file_id": r.file_id,
                "category": r.category,
                "label": r.eff_label,
                "taxonomy_id": r.label_taxonomy_id,
            }

    # Rebuild: delete then recreate the AI rows (carrying human_count) plus
    # any surviving human-only rows.
    db.execute(
        delete(EventObservation).where(EventObservation.event_id == event_id)
    )

    ai_keys = set(max_n_per_key.keys())
    new_effective: dict[str, int] = {}
    observations: list[EventObservation] = []

    for key, data in max_n_per_key.items():
        hc = prior_human.get(key)
        obs = EventObservation(
            id=str(uuid.uuid4()),
            event_id=event_id,
            label=data["label"],
            label_taxonomy_id=data["taxonomy_id"],
            category=data["category"],
            max_n=data["count"],
            max_n_file_id=data["file_id"],
            human_count=hc,
        )
        db.add(obs)
        observations.append(obs)
        new_effective[key] = hc if hc is not None else data["count"]

    # Human-only species: recorded by a human, not detected by the AI this
    # round. Keep them (max_n=0, no frame). Dedupe by key defensively.
    seen_human_only: set[str] = set()
    for r in existing:
        key = r.label_taxonomy_id or r.label
        if (
            key is None
            or key in ai_keys
            or key in seen_human_only
            or r.human_count is None
        ):
            continue
        seen_human_only.add(key)
        obs = EventObservation(
            id=str(uuid.uuid4()),
            event_id=event_id,
            label=r.label,
            label_taxonomy_id=r.label_taxonomy_id,
            category=r.category,
            max_n=0,
            max_n_file_id=None,
            human_count=r.human_count,
        )
        db.add(obs)
        observations.append(obs)
        new_effective[key] = r.human_count

    if prior_effective != new_effective:
        event = db.get(Event, event_id)
        if event is not None and event.confirmed:
            event.confirmed = False

    return observations


def recalculate_max_n_for_project(
    db: Session,
    project_id: str,
) -> int:
    """
    Recalculate MaxN for all events in a project.

    Returns the total number of EventObservation rows created.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    threshold = project.counting_threshold

    # Get all event IDs for this project
    event_ids = (
        db.query(Event.id)
        .join(Deployment, Deployment.id == Event.deployment_id)
        .filter(Deployment.project_id == project_id)
        .all()
    )

    total_observations = 0
    for (event_id,) in event_ids:
        obs = calculate_max_n_for_event(db, event_id, threshold)
        total_observations += len(obs)

    logger.info(
        f"Recalculated MaxN for project {project_id}: "
        f"{len(event_ids)} events, {total_observations} observations"
    )
    return total_observations


def recalculate_max_n_for_events(
    db: Session,
    event_ids: list[str],
    counting_threshold: float,
) -> None:
    """Recalculate MaxN for specific events (after verify/relabel)."""
    for event_id in event_ids:
        calculate_max_n_for_event(db, event_id, counting_threshold)


def get_event_ids_for_detections(
    db: Session,
    detection_ids: list[str],
) -> list[str]:
    """Get event IDs that contain files with the given detections."""
    rows = (
        db.query(event_files.c.event_id)
        .join(File, File.id == event_files.c.file_id)
        .join(Detection, Detection.file_id == File.id)
        .filter(Detection.id.in_(detection_ids))
        .distinct()
        .all()
    )
    return [row[0] for row in rows]


def get_thumbnail_file_id(db: Session, event_id: str) -> str | None:
    """Get the max_n_file_id of the dominant species (highest max_n)."""
    obs = (
        db.query(EventObservation.max_n_file_id)
        .filter(EventObservation.event_id == event_id)
        .order_by(EventObservation.max_n.desc())
        .first()
    )
    return obs[0] if obs else None


def get_max_n_frames(db: Session, event_id: str) -> list[dict]:
    """Get all MaxN frames for an event, ordered by max_n descending.

    Only rows with a frame (`max_n_file_id`) are returned, so human-only
    species (no AI frame) are excluded here. `effective_count` is the
    human-authoritative count (`human_count` if set, else `max_n`).
    """
    rows = (
        db.query(
            EventObservation.max_n_file_id,
            EventObservation.label,
            EventObservation.max_n,
            EventObservation.human_count,
            EventObservation.label_taxonomy_id,
        )
        .filter(EventObservation.event_id == event_id)
        .filter(EventObservation.max_n_file_id.isnot(None))
        .order_by(EventObservation.max_n.desc())
        .all()
    )
    return [
        {
            "file_id": row[0],
            "label": row[1],
            "max_n": row[2],
            "effective_count": row[3] if row[3] is not None else row[2],
            "label_taxonomy_id": row[4],
        }
        for row in rows
    ]


def list_event_observations(
    db: Session, event_id: str
) -> list[EventObservation]:
    """All observation rows for an event (AI + human-only).

    Ordered by the AI MaxN (highest first), then label. The key is
    deliberately independent of `human_count` so editing a count never
    reshuffles the rows under the user's cursor; human-only rows (max_n=0)
    fall to the bottom alphabetically.
    """
    rows = (
        db.query(EventObservation)
        .filter(EventObservation.event_id == event_id)
        .all()
    )
    return sorted(rows, key=lambda o: (-o.max_n, o.label or ""))


def set_human_count(
    db: Session, event_observation_id: str, count: int | None
) -> EventObservation | None:
    """Set (or clear, when count is None) the human count on one row.

    Clears the event's sign-off, since the counts just changed. Returns
    the updated row, or None when the id is unknown.
    """
    obs = (
        db.query(EventObservation)
        .filter(EventObservation.id == event_observation_id)
        .first()
    )
    if obs is None:
        return None
    obs.human_count = count
    event = db.get(Event, obs.event_id)
    if event is not None:
        event.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def add_human_species(
    db: Session,
    event_id: str,
    category: str,
    count: int,
    label: str | None = None,
    label_taxonomy_id: str | None = None,
) -> EventObservation:
    """Record a species the AI missed entirely (or bump an existing row).

    If a row already matches the species (by taxonomy id, else label) its
    human_count is set; otherwise a human-only row is created (max_n=0, no
    frame). Clears the event's sign-off. Returns the row.
    """
    # Resolve the taxonomy id from the label when not supplied, so a
    # human-added species keys to the same row the AI would produce and
    # doesn't split into a duplicate if the AI later detects it.
    if label_taxonomy_id is None and label:
        from app.ml.taxonomy_db import resolve_taxonomy_id

        project_id = (
            db.query(Deployment.project_id)
            .join(Event, Event.deployment_id == Deployment.id)
            .filter(Event.id == event_id)
            .scalar()
        )
        if project_id:
            label_taxonomy_id = resolve_taxonomy_id(label, project_id, db)

    query = db.query(EventObservation).filter(
        EventObservation.event_id == event_id
    )
    if label_taxonomy_id is not None:
        existing = query.filter(
            EventObservation.label_taxonomy_id == label_taxonomy_id
        ).first()
    else:
        existing = query.filter(
            EventObservation.label_taxonomy_id.is_(None),
            EventObservation.label == label,
            EventObservation.category == category,
        ).first()

    if existing is not None:
        existing.human_count = count
        obs = existing
    else:
        obs = EventObservation(
            id=str(uuid.uuid4()),
            event_id=event_id,
            label=label,
            label_taxonomy_id=label_taxonomy_id,
            category=category,
            max_n=0,
            max_n_file_id=None,
            human_count=count,
        )
        db.add(obs)

    event = db.get(Event, event_id)
    if event is not None:
        event.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def relabel_observation(
    db: Session,
    event_observation_id: str,
    category: str,
    label: str | None = None,
    label_taxonomy_id: str | None = None,
) -> EventObservation | None:
    """Change the species of one count row, carrying its count to the target.

    Count-level relabel: the source row is removed the same way the panel's
    X does (a human-only row is deleted, an AI row keeps its boxes but its
    human_count drops to 0 so it hides and survives a MaxN recompute), and
    the source's effective count is moved onto the target species. If the
    target species already has a row in the event, the counts SUM (bird(5)
    relabelled to deer, with deer already 1, gives deer 6); otherwise a
    human-only row is created for it. This edits counts only, not the
    underlying detections, exactly like add/remove on this panel. Clears the
    event's sign-off. Returns the target row, or None when the id is unknown.
    """
    source = (
        db.query(EventObservation)
        .filter(EventObservation.id == event_observation_id)
        .first()
    )
    if source is None:
        return None
    event_id = source.event_id
    source_count = source.effective_count

    # Resolve the target taxonomy id from the label when not supplied, so the
    # relabel keys to the same row the AI would produce.
    if label_taxonomy_id is None and label:
        from app.ml.taxonomy_db import resolve_taxonomy_id

        project_id = (
            db.query(Deployment.project_id)
            .join(Event, Event.deployment_id == Deployment.id)
            .filter(Event.id == event_id)
            .scalar()
        )
        if project_id:
            label_taxonomy_id = resolve_taxonomy_id(label, project_id, db)

    # No-op when relabelling to the species it already is.
    same_species = (
        (
            label_taxonomy_id is not None
            and label_taxonomy_id == source.label_taxonomy_id
        )
        or (
            label_taxonomy_id is None
            and source.label_taxonomy_id is None
            and label == source.label
            and category == source.category
        )
    )
    if same_species:
        return source

    # Current count on the target species (0 if it has no row yet), read
    # before we touch the source so the sum is correct even when source and
    # target sit in the same event.
    target_query = db.query(EventObservation).filter(
        EventObservation.event_id == event_id
    )
    if label_taxonomy_id is not None:
        target = target_query.filter(
            EventObservation.label_taxonomy_id == label_taxonomy_id
        ).first()
    else:
        target = target_query.filter(
            EventObservation.label_taxonomy_id.is_(None),
            EventObservation.label == label,
            EventObservation.category == category,
        ).first()
    target_existing = target.effective_count if target is not None else 0
    summed = target_existing + source_count

    # Remove the source (same semantics as the panel's X / count-to-zero).
    if source.max_n == 0:
        db.delete(source)
    else:
        source.human_count = 0

    # Move the count onto the target. add_human_species merges into the
    # existing row (setting human_count to the summed total) or creates a
    # human-only row, resolves taxonomy, clears the sign-off, and commits.
    return add_human_species(
        db,
        event_id,
        category=category,
        count=summed,
        label=label,
        label_taxonomy_id=label_taxonomy_id,
    )


def delete_event_observation(
    db: Session, event_observation_id: str
) -> str | None:
    """Remove the human contribution to one observation row.

    A human-only row (the AI detected nothing: max_n=0) is deleted
    outright; an AI row keeps its box-derived MaxN but drops the human
    override. Clears the event's sign-off. Returns the event id, or None
    when the id is unknown.
    """
    obs = (
        db.query(EventObservation)
        .filter(EventObservation.id == event_observation_id)
        .first()
    )
    if obs is None:
        return None
    event_id = obs.event_id
    if obs.max_n == 0:
        db.delete(obs)
    else:
        obs.human_count = None
    event = db.get(Event, event_id)
    if event is not None:
        event.confirmed = False
    db.commit()
    return event_id


def set_event_confirmed(
    db: Session, event_id: str, confirmed: bool
) -> Event | None:
    """Set the human confirmation on an event's species and counts."""
    event = db.get(Event, event_id)
    if event is None:
        return None
    event.confirmed = confirmed
    db.commit()
    db.refresh(event)
    return event


def reset_event_to_ai(db: Session, event_id: str) -> Event | None:
    """Drop every human edit to the event's counts, back to the AI proposal.

    Clears `human_count` on the AI rows and deletes the human-only rows
    (the species the AI never detected). Clears the event's sign-off.
    Returns the event, or None when the id is unknown.
    """
    event = db.get(Event, event_id)
    if event is None:
        return None
    rows = (
        db.query(EventObservation)
        .filter(EventObservation.event_id == event_id)
        .all()
    )
    for obs in rows:
        if obs.max_n == 0:
            db.delete(obs)
        else:
            obs.human_count = None
    event.confirmed = False
    db.commit()
    db.refresh(event)
    return event


def get_project_threshold_for_detections(
    db: Session,
    detection_ids: list[str],
) -> float:
    """The project ``counting_threshold`` owning the given detections.

    Requires at least one id, and requires it to resolve. Both used to
    fall back to ``0.0``, which is not a neutral value: it is the
    threshold at which every detection passes, including MegaDetector's
    near-noise tail down to its 0.01 output cap. A failed lookup therefore
    rebuilt an event's counts against the wrong floor and said nothing.

    Callers must capture the threshold *before* deleting detections; the
    join runs through the detection rows themselves, so afterwards there
    is nothing left to resolve.
    """
    if not detection_ids:
        raise ValueError(
            "get_project_threshold_for_detections needs at least one "
            "detection id; there is no sensible threshold for none."
        )
    row = (
        db.query(Project.counting_threshold)
        .join(Deployment, Deployment.project_id == Project.id)
        .join(File, File.deployment_id == Deployment.id)
        .join(Detection, Detection.file_id == File.id)
        .filter(Detection.id.in_(detection_ids))
        .first()
    )
    if row is None:
        raise ValueError(
            f"No project reachable from detections {detection_ids[:5]}"
            f"{'...' if len(detection_ids) > 5 else ''}. Either they were "
            f"already deleted (capture the threshold first) or the "
            f"database is corrupt. Refusing to guess a threshold."
        )
    return row[0]
