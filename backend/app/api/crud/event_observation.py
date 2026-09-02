"""
MaxN calculation and storage for event observations.

MaxN is the maximum number of individuals of a species visible in any
single image within an event. Calculated per-species, stored in the
event_observations table.

A row is one cohort (see the model). The AI seeds one row per species
with no demographics; a person labels that row in place or splits it.
The rebuild below carries all of that: see DEVELOPERS.md, "Observation
cohorts".
"""

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, or_, text
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.label_exclusion import is_a_real_detection
from app.models import Deployment, Detection, Event, File, Project
from app.models.event import event_files
from app.models.event_observation import EventObservation

logger = get_logger(__name__)

# The three demographic columns, in one place: the seed rule, the row
# matching and the confirmed-reset compare all walk them.
DEMOGRAPHIC_FIELDS = ("sex", "life_stage", "behavior")

# "Not passed" marker for partial updates, so a caller can clear a field
# with None and leave another alone by not naming it.
UNSET: Any = object()


@dataclass
class PriorObs:
    """A snapshot of one prior EventObservation, enough to carry the human
    layer (human_count, demographics, human-only rows) and detect an
    effective-set change. Its attribute names mirror EventObservation so
    `calculate_max_n_for_event` can read either interchangeably. Used to
    carry the human layer of a deleted event onto its regenerated
    replacement (same file set)."""

    label: str | None
    label_taxonomy_id: str | None
    category: str
    max_n: int
    human_count: int | None
    effective_count: int
    sex: str | None = None
    life_stage: str | None = None
    behavior: str | None = None


def _species_key(row: Any) -> str | None:
    """What identifies a species across a rebuild: the taxonomy id, else
    the label. Category-only rows (person, vehicle) key on their label,
    which `calculate_max_n_for_event` sets to the category."""
    return row.label_taxonomy_id or row.label


def _has_demographics(row: Any) -> bool:
    return any(getattr(row, f) is not None for f in DEMOGRAPHIC_FIELDS)


def _cohort_signature(row: Any) -> tuple:
    """What the confirmed-reset compare looks at: species, demographics and
    the effective count. Notes are commentary, not the observation, so a
    note edit never unconfirms."""
    return (
        _species_key(row),
        *(getattr(row, f) for f in DEMOGRAPHIC_FIELDS),
        row.effective_count,
    )


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
    2. Rebuild the event's `event_observations` rows: one per AI species,
       carrying the human layer of that species' *seed* row (its
       human_count, sex, life stage and behaviour), plus every other
       human-only row (max_n=0, no frame) recreated as it was: the cohorts a
       person split off, and the species the AI did not detect this round.
    3. Clear `Event.confirmed` when the set of (species, demographics,
       count) changed (a Labels-page relabel/add/delete that moves a count
       un-signs the event); a pure detection-verify that leaves it unchanged
       does not.

    The seed of a species is its prior AI row (max_n > 0), else a human-only
    row without demographics (a species added by hand that the AI now finds
    merges into the AI row, as before). Species are keyed by
    `label_taxonomy_id or label` so the layer survives the delete-and-rebuild.

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
    existing: list[Any] = (
        prior
        if prior is not None
        else _rows_in_stored_order(db, event_id)
    )
    # A multiset, not a sorted list: the tuples hold None and str.
    prior_signature = Counter(
        _cohort_signature(r) for r in existing if _species_key(r) is not None
    )

    # The seed per species: the prior AI row, else a human-only row without
    # demographics. Its human layer lands on the new AI row. Rows with
    # demographics are never seeds: they are cohorts and stay their own row.
    seeds: dict[str, Any] = {}
    for r in existing:
        key = _species_key(r)
        if key is not None and r.max_n > 0:
            seeds[key] = r
    for r in existing:
        key = _species_key(r)
        if (
            key is not None
            and key not in seeds
            and r.human_count is not None
            and not _has_demographics(r)
        ):
            seeds[key] = r

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

    # Rebuild: delete then recreate the AI rows (carrying the seed's human
    # layer) plus every other human row. Insertion order is the display
    # order (see `_rows_in_stored_order`), so AI rows go first and the human
    # rows follow in the order they had.
    db.execute(
        delete(EventObservation).where(EventObservation.event_id == event_id)
    )

    consumed: set[int] = set()
    observations: list[EventObservation] = []

    for key, data in max_n_per_key.items():
        seed = seeds.get(key)
        if seed is not None:
            consumed.add(id(seed))
        obs = EventObservation(
            id=str(uuid.uuid4()),
            event_id=event_id,
            label=data["label"],
            label_taxonomy_id=data["taxonomy_id"],
            category=data["category"],
            max_n=data["count"],
            max_n_file_id=data["file_id"],
            **_human_layer(seed),
        )
        db.add(obs)
        observations.append(obs)

    # Everything a person recorded that no AI row absorbed: split-off
    # cohorts, and species the AI did not detect this round. Recreated as
    # human-only rows (max_n=0, no frame), duplicates included: they are
    # the user's. A row that only carries demographics (an AI row someone
    # labelled, whose species the AI then stopped seeing) counts as
    # recorded too; its count becomes a human count so it survives the
    # same way a count override does.
    for r in existing:
        if (
            _species_key(r) is None
            or id(r) in consumed
            or (r.human_count is None and not _has_demographics(r))
        ):
            continue
        layer = _human_layer(r)
        if layer["human_count"] is None:
            layer["human_count"] = r.effective_count
        obs = EventObservation(
            id=str(uuid.uuid4()),
            event_id=event_id,
            label=r.label,
            label_taxonomy_id=r.label_taxonomy_id,
            category=r.category,
            max_n=0,
            max_n_file_id=None,
            **layer,
        )
        db.add(obs)
        observations.append(obs)

    new_signature = Counter(_cohort_signature(o) for o in observations)
    if prior_signature != new_signature:
        event = db.get(Event, event_id)
        if event is not None and event.confirmed:
            event.confirmed = False

    return observations


def _human_layer(row: Any | None) -> dict[str, Any]:
    """The columns a person owns on a row, as constructor kwargs. All None
    when there is no prior row to carry."""
    if row is None:
        return {
            "human_count": None,
            "sex": None,
            "life_stage": None,
            "behavior": None,
        }
    return {
        "human_count": row.human_count,
        "sex": row.sex,
        "life_stage": row.life_stage,
        "behavior": row.behavior,
    }


def _rows_in_stored_order(db: Session, event_id: str) -> list[EventObservation]:
    """An event's rows in insertion order (SQLite rowid).

    That order is the display order of the cohorts under a species, and
    the rebuild reinserts rows in this same order, so a Labels-page
    relabel never reshuffles the rows under the user's cursor. The rows
    have no timestamp of their own and the ids are random, so rowid is the
    one thing that carries the order. SQLite only, like the app."""
    return (
        db.query(EventObservation)
        .filter(EventObservation.event_id == event_id)
        .order_by(text("event_observations.rowid"))
        .all()
    )


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


def get_event_ids_for_files(db: Session, file_ids: list[str]) -> list[str]:
    """Event IDs that contain any of the given files.

    By file, not by detection: a file whose boxes were just deleted has
    nothing left to join through.
    """
    if not file_ids:
        return []
    rows = (
        db.query(event_files.c.event_id)
        .filter(event_files.c.file_id.in_(file_ids))
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
    species total across all its cohort rows (the card chip reads it), not
    the frame row's own count: a species split into 1 + 1 says 2.
    """
    rows = (
        db.query(EventObservation)
        .filter(EventObservation.event_id == event_id)
        .order_by(EventObservation.max_n.desc())
        .all()
    )
    total_by_key: dict[str | None, int] = defaultdict(int)
    for row in rows:
        total_by_key[_species_key(row)] += row.effective_count
    return [
        {
            "file_id": row.max_n_file_id,
            "label": row.label,
            "max_n": row.max_n,
            "effective_count": total_by_key[_species_key(row)],
            "label_taxonomy_id": row.label_taxonomy_id,
        }
        for row in rows
        if row.max_n_file_id is not None
    ]


def _row(
    db: Session, event_observation_id: str, event_id: str | None
) -> EventObservation | None:
    """One row by id. With `event_id` the row must belong to that event, so
    a request whose path names another event touches nothing."""
    query = db.query(EventObservation).filter(
        EventObservation.id == event_observation_id
    )
    if event_id is not None:
        query = query.filter(EventObservation.event_id == event_id)
    return query.first()


def list_event_observations(
    db: Session, event_id: str
) -> list[EventObservation]:
    """All observation rows for an event (AI + human-only), grouped by
    species.

    Species are ordered by the AI MaxN (highest first), then label; species
    the AI did not detect come after, alphabetically. Within a species the
    AI row comes first and the cohorts follow in the order they were made
    (rowid). The key is deliberately independent of `human_count` and of the
    demographics, so editing a count or a dropdown never reshuffles the rows
    under the user's cursor.
    """
    rows = _rows_in_stored_order(db, event_id)
    ai_order = sorted(
        (o for o in rows if o.max_n > 0), key=lambda o: (-o.max_n, o.label or "")
    )
    rank: dict[str | None, tuple[int, str]] = {}
    for i, o in enumerate(ai_order):
        rank.setdefault(_species_key(o), (i, ""))
    for o in rows:
        rank.setdefault(_species_key(o), (len(ai_order), o.label or ""))
    # Python's sort is stable, so rowid order survives inside a species.
    return sorted(rows, key=lambda o: (*rank[_species_key(o)], o.max_n == 0))


def set_human_count(
    db: Session,
    event_observation_id: str,
    count: int | None,
    event_id: str | None = None,
) -> EventObservation | None:
    """Set (or clear, when count is None) the human count on one row.

    Clears the event's sign-off, since the counts just changed. Returns
    the updated row, or None when the id is unknown (or not in `event_id`).
    """
    obs = _row(db, event_observation_id, event_id)
    if obs is None:
        return None
    obs.human_count = count
    event = db.get(Event, obs.event_id)
    if event is not None:
        event.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def _resolve_taxonomy_id(
    db: Session, event_id: str, label: str | None, label_taxonomy_id: str | None
) -> str | None:
    """Resolve the taxonomy id from the label when not supplied, so a
    human-added species keys to the same row the AI would produce and
    doesn't split into a duplicate if the AI later detects it."""
    if label_taxonomy_id is not None or not label:
        return label_taxonomy_id
    from app.ml.taxonomy_db import resolve_taxonomy_id

    project_id = (
        db.query(Deployment.project_id)
        .join(Event, Event.deployment_id == Deployment.id)
        .filter(Event.id == event_id)
        .scalar()
    )
    if not project_id:
        return None
    return resolve_taxonomy_id(label, project_id, db)


def _find_cohort(
    db: Session,
    event_id: str,
    category: str,
    label: str | None,
    label_taxonomy_id: str | None,
    demographics: dict[str, str | None],
) -> EventObservation | None:
    """The event's row for this species with exactly these demographics
    (all None = the plain row), in stored order, or None."""
    query = db.query(EventObservation).filter(
        EventObservation.event_id == event_id
    )
    if label_taxonomy_id is not None:
        query = query.filter(
            EventObservation.label_taxonomy_id == label_taxonomy_id
        )
    else:
        query = query.filter(
            EventObservation.label_taxonomy_id.is_(None),
            EventObservation.label == label,
            EventObservation.category == category,
        )
    for field in DEMOGRAPHIC_FIELDS:
        value = demographics.get(field)
        column = getattr(EventObservation, field)
        query = query.filter(column.is_(None) if value is None else column == value)
    return query.order_by(text("event_observations.rowid")).first()


def add_human_species(
    db: Session,
    event_id: str,
    category: str,
    count: int,
    label: str | None = None,
    label_taxonomy_id: str | None = None,
    sex: str | None = None,
    life_stage: str | None = None,
    behavior: str | None = None,
) -> EventObservation:
    """Record a species the AI missed entirely (or bump an existing row).

    If a row already matches the species (by taxonomy id, else label) with
    the same demographics its human_count is set; otherwise a human-only row
    is created (max_n=0, no frame) carrying the demographics.
    Clears the event's sign-off. Returns the row.
    """
    label_taxonomy_id = _resolve_taxonomy_id(db, event_id, label, label_taxonomy_id)
    demographics = {"sex": sex, "life_stage": life_stage, "behavior": behavior}
    existing = _find_cohort(
        db, event_id, category, label, label_taxonomy_id, demographics
    )

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
            **demographics,
        )
        db.add(obs)

    event = db.get(Event, event_id)
    if event is not None:
        event.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def set_observation_attributes(
    db: Session,
    event_observation_id: str,
    *,
    sex: str | None = UNSET,
    life_stage: str | None = UNSET,
    behavior: str | None = UNSET,
    event_id: str | None = None,
) -> EventObservation | None:
    """Set sex / life stage / behaviour on one row. A field left UNSET is
    not touched; None clears it.

    A change clears the event's sign-off, because it changes the recorded
    observation. Returns the row, or None when the id is unknown (or not
    in `event_id`).
    """
    obs = _row(db, event_observation_id, event_id)
    if obs is None:
        return None
    demographics_changed = False
    for field, value in (
        ("sex", sex), ("life_stage", life_stage), ("behavior", behavior)
    ):
        if value is not UNSET and getattr(obs, field) != value:
            setattr(obs, field, value)
            demographics_changed = True
    if demographics_changed:
        event = db.get(Event, obs.event_id)
        if event is not None:
            event.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def split_observation(
    db: Session, event_observation_id: str, event_id: str | None = None
) -> EventObservation | None:
    """Split one row into two cohorts of the same species.

    The source keeps its count minus one (never below one) and the new
    human-only row starts at one, with no demographics, so the
    user then sets what makes it different. Splitting a row at count one
    still gives two rows of one; the user corrects the numbers. Clears the
    event's sign-off. Returns the new row, or None when the id is unknown
    (or not in `event_id`).
    """
    source = _row(db, event_observation_id, event_id)
    if source is None:
        return None
    source.human_count = max(1, source.effective_count - 1)
    obs = EventObservation(
        id=str(uuid.uuid4()),
        event_id=source.event_id,
        label=source.label,
        label_taxonomy_id=source.label_taxonomy_id,
        category=source.category,
        max_n=0,
        max_n_file_id=None,
        human_count=1,
    )
    db.add(obs)
    event = db.get(Event, source.event_id)
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
    event_id: str | None = None,
) -> EventObservation | None:
    """Change the species of one count row, carrying its count to the target.

    Count-level relabel: the source row is removed the same way the panel's
    X does (a human-only row is deleted, an AI row keeps its boxes but its
    human_count drops to 0 so it hides and survives a MaxN recompute), and
    the source's effective count is moved onto the target species, keeping
    the source's demographics. If the target species already has
    a row with those demographics in the event, the counts SUM (bird(5)
    relabelled to deer, with deer already 1, gives deer 6); otherwise a
    human-only row is created for it. This edits counts only, not the
    underlying detections, exactly like add/remove on this panel. Clears the
    event's sign-off. Returns the target row, or None when the id is unknown
    (or not in `event_id`).
    """
    source = _row(db, event_observation_id, event_id)
    if source is None:
        return None
    event_id = source.event_id
    source_count = source.effective_count
    demographics = {f: getattr(source, f) for f in DEMOGRAPHIC_FIELDS}

    label_taxonomy_id = _resolve_taxonomy_id(db, event_id, label, label_taxonomy_id)

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

    # Current count on the target cohort (0 if it has no row yet), read
    # before we touch the source so the sum is correct even when source and
    # target sit in the same event.
    target = _find_cohort(
        db, event_id, category, label, label_taxonomy_id, demographics
    )
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
        **demographics,
    )


def delete_event_observation(
    db: Session, event_observation_id: str, event_id: str | None = None
) -> str | None:
    """Remove the human contribution to one observation row.

    A human-only row (the AI detected nothing: max_n=0) is deleted
    outright; an AI row keeps its box-derived MaxN but drops the human
    override. Clears the event's sign-off. Returns the event id, or None
    when the id is unknown (or not in `event_id`).
    """
    obs = _row(db, event_observation_id, event_id)
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


def set_event_notes(
    db: Session, event_id: str, notes: str | None
) -> Event | None:
    """Set a person's free text on an event. Never touches the sign-off:
    a note is commentary, not the recorded observation."""
    event = db.get(Event, event_id)
    if event is None:
        return None
    event.notes = (notes or "").strip() or None
    db.commit()
    db.refresh(event)
    return event


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

    Clears `human_count` and the demographics on the AI rows and
    deletes the human-only rows (split-off cohorts and the species the AI
    never detected). Clears the event's sign-off. Returns the event, or None
    when the id is unknown.
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
            for field, value in _human_layer(None).items():
                setattr(obs, field, value)
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
