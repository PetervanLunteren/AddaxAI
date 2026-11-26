"""
MaxN calculation and storage for sequence observations.

MaxN is the maximum number of individuals of a species visible in any
single image within a sequence. Calculated per-species, stored in the
sequence_observations table.

A row is one cohort (see the model). The AI seeds one row per species
with no demographics; a person labels that row in place or splits it.
The rebuild below carries all of that: see DEVELOPERS.md, "Observation
cohorts".
"""

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.label_exclusion import is_a_real_detection, threshold_or_verified
from app.models import Deployment, Detection, File, Project, Sequence, Track
from app.models.sequence import sequence_files
from app.models.sequence_observation import SequenceObservation
from app.utils.media_dates import frame_time

logger = get_logger(__name__)

# The three demographic columns, in one place: the seed rule, the row
# matching and the confirmed-reset compare all walk them.
DEMOGRAPHIC_FIELDS = ("sex", "life_stage", "behavior")

# "Not passed" marker for partial updates, so a caller can clear a field
# with None and leave another alone by not naming it.
UNSET: Any = object()


@dataclass
class PriorObs:
    """A snapshot of one prior SequenceObservation, enough to carry the human
    layer (human_count, demographics, human-only rows) and detect an
    effective-set change. Its attribute names mirror SequenceObservation so
    `calculate_max_n_for_sequence` can read either interchangeably. Used to
    carry the human layer of a deleted sequence onto its regenerated
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


def count_individuals_by_class(db: Session, deployment_ids: list[str]) -> dict[str, int]:
    """Individuals per detector class over these deployments' sequences
    (the confirmed count where a person set one, else the AI's MaxN),
    largest first, classes with nothing left out. For the deployment and
    site info sheets (`ClassCounts.classes`)."""
    total = func.sum(SequenceObservation.effective_count)
    rows = db.execute(
        select(SequenceObservation.category, total)
        .join(Sequence, Sequence.id == SequenceObservation.sequence_id)
        .where(Sequence.deployment_id.in_(deployment_ids))
        .group_by(SequenceObservation.category)
        .having(total > 0)
        .order_by(total.desc(), SequenceObservation.category)
    ).all()
    return {category: int(n) for category, n in rows}


def _species_key(row: Any) -> str | None:
    """What identifies a species across a rebuild: the taxonomy id, else
    the label. Category-only rows (person, vehicle) key on their label,
    which `calculate_max_n_for_sequence` sets to the category."""
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
    """The shared scope rule; kept as a local name for its two callers."""
    return threshold_or_verified(threshold)


def calculate_max_n_for_sequence(
    db: Session,
    sequence_id: str,
    counting_threshold: float,
    prior: list[PriorObs] | None = None,
) -> list[SequenceObservation]:
    """
    Recalculate the AI-derived MaxN per species for a single sequence, while
    preserving the human-authoritative data layered on top.

    `prior` supplies the human layer to carry when the sequence was just
    (re)created and has no rows of its own yet, e.g. during an interval
    change where a deleted sequence's replacement covers the same files. When
    None, the sequence's own current rows are used (the normal relabel/verify
    path).

    Algorithm:
    1. Count detections per (file, frame, taxonomy) and take the maximum
       count per species (= MaxN). Ties break on summed confidence.
       For videos a species only counts when it has a card (a box on one
       of its tracks' representative frames) or a person put that label on
       a frame themselves: any other per-frame label is classifier noise
       the user can't see or clean in the Labels step, so it must not
       spawn a spurious species row.
    2. Rebuild the sequence's `sequence_observations` rows: one per AI species,
       carrying the human layer of that species' *seed* row (its
       human_count, sex, life stage and behaviour), plus every other
       human-only row (max_n=0, no frame) recreated as it was: the cohorts a
       person split off, and the species the AI did not detect this round.
    3. Clear `Sequence.confirmed` when the set of (species, demographics,
       count) changed (a Labels-page relabel/add/delete that moves a count
       un-signs the sequence); a pure detection-verify that leaves it unchanged
       does not.

    The seed of a species is its prior AI row (max_n > 0), else a human-only
    row without demographics (a species added by hand that the AI now finds
    merges into the AI row, as before). Species are keyed by
    `label_taxonomy_id or label` so the layer survives the delete-and-rebuild.

    Returns the rebuilt SequenceObservation rows.
    """
    # Group by label_taxonomy_id (authoritative), falling back to
    # COALESCE(label, category) for display string.
    effective_label = func.coalesce(Detection.label, Detection.category)

    # Count detections per (file, frame, taxonomy) so a video that shows
    # 4 animals in frame A and 4 in frame B reports MaxN=4 instead of
    # collapsing into MaxN=8. For image rows, `frame_number` is NULL and
    # all detections in one file land in a single group (NULL == NULL in
    # GROUP BY semantics), preserving the legacy per-image behaviour.
    # file_type / best_frame_number ride along to gate video species
    # (below). They are constant per file_id, so adding the two columns to
    # GROUP BY doesn't change the grouping.
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
            # True when a person put this label on a frame themselves.
            # A file sign-off verifies without touching labels, so this
            # cannot be mistaken for one (see the gate below).
            func.max(Detection.classification_method == "human").label("any_human"),
            # True when this frame carries the representative box of a
            # track of this species: that box has a card, so the species
            # is reviewable and may spawn a row (see the gate below).
            func.max(Track.representative_frame_number == Detection.frame_number).label(
                "any_representative"
            ),
        )
        .join(File, File.id == Detection.file_id)
        .join(sequence_files, sequence_files.c.file_id == File.id)
        .outerjoin(Track, Track.id == Detection.track_id)
        .filter(sequence_files.c.sequence_id == sequence_id)
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
        # Frame order, so an equal count later in the clip never replaces
        # the first one: MaxN is taken at its first occurrence, the
        # convention EventMeasure defaults to.
        .order_by(Detection.file_id, Detection.frame_number)
        .all()
    )

    # Snapshot the existing rows so the human layer survives the rebuild
    # and so we can detect whether the effective set changed. A caller can
    # pass `prior` instead (a freshly recreated sequence has no rows of its own).
    existing: list[Any] = (
        prior
        if prior is not None
        else _rows_in_stored_order(db, sequence_id)
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

    # A video species is suggested when it has a card (a box on the
    # representative frame of one of its tracks, one card per track on
    # the Labels page) **or** when a person put that label on a frame
    # themselves. Any other per-frame label is classifier noise the user
    # cannot see or clean in the Labels step, so it must not spawn a
    # spurious species row. Images are never gated (every image detection
    # is visible and cleanable). Hand copy of the rule in
    # ml/detection_visibility.py, kept here to keep the grouping.
    #
    # The second half is what makes opening a track work. Relabelling the
    # frames of a merged track that followed a different animal is the
    # documented way to put it right, and the card keeps the old species,
    # so on the card test alone the new species had no row anywhere: the
    # person's verdict lived in the database and in no count. It cannot
    # let noise through, because confirming a card now writes that card's
    # label over every frame nobody touched (`apply_card_labels`).
    allowed_video_keys: dict[str, set[str]] = defaultdict(set)
    for r in counts:
        if r.file_type == "video" and (r.any_representative or r.any_human):
            allowed_video_keys[r.file_id].add(r.label_taxonomy_id or r.eff_label)

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
                "frame_number": r.frame_number,
                "category": r.category,
                "label": r.eff_label,
                "taxonomy_id": r.label_taxonomy_id,
            }

    # Rebuild: delete then recreate the AI rows (carrying the seed's human
    # layer) plus every other human row. Insertion order is the display
    # order (see `_rows_in_stored_order`), so AI rows go first and the human
    # rows follow in the order they had.
    #
    # "fetch" is not optional. The default, "evaluate", tests the WHERE in
    # Python against every object in the session after the delete. Sequence
    # generation holds the whole deployment's files in its session and
    # calls this once per sequence, which made it files x sequences: 168 s for
    # 54K photos, an estimated 13 hours for 900K. "fetch" gets the deleted
    # ids back from SQLite (RETURNING) and expires only those objects.
    db.execute(
        delete(SequenceObservation)
        .where(SequenceObservation.sequence_id == sequence_id)
        .execution_options(synchronize_session="fetch")
    )

    consumed: set[int] = set()
    observations: list[SequenceObservation] = []

    for key, data in max_n_per_key.items():
        seed = seeds.get(key)
        if seed is not None:
            consumed.add(id(seed))
        obs = SequenceObservation(
            id=str(uuid.uuid4()),
            sequence_id=sequence_id,
            label=data["label"],
            label_taxonomy_id=data["taxonomy_id"],
            category=data["category"],
            max_n=data["count"],
            max_n_file_id=data["file_id"],
            max_n_frame_number=data["frame_number"],
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
        obs = SequenceObservation(
            id=str(uuid.uuid4()),
            sequence_id=sequence_id,
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
        sequence = db.get(Sequence, sequence_id)
        if sequence is not None and sequence.confirmed:
            sequence.confirmed = False

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


def _rows_in_stored_order(db: Session, sequence_id: str) -> list[SequenceObservation]:
    """A sequence's rows in insertion order (SQLite rowid).

    That order is the display order of the cohorts under a species, and
    the rebuild reinserts rows in this same order, so a Labels-page
    relabel never reshuffles the rows under the user's cursor. The rows
    have no timestamp of their own and the ids are random, so rowid is the
    one thing that carries the order. SQLite only, like the app."""
    return (
        db.query(SequenceObservation)
        .filter(SequenceObservation.sequence_id == sequence_id)
        .order_by(text("sequence_observations.rowid"))
        .all()
    )


def recalculate_max_n_for_project(
    db: Session,
    project_id: str,
) -> int:
    """
    Recalculate MaxN for all sequences in a project.

    Returns the total number of SequenceObservation rows created.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    threshold = project.counting_threshold

    # Get all sequence IDs for this project
    sequence_ids = (
        db.query(Sequence.id)
        .join(Deployment, Deployment.id == Sequence.deployment_id)
        .filter(Deployment.project_id == project_id)
        .all()
    )

    total_observations = 0
    for (sequence_id,) in sequence_ids:
        obs = calculate_max_n_for_sequence(db, sequence_id, threshold)
        total_observations += len(obs)

    logger.info(
        f"Recalculated MaxN for project {project_id}: "
        f"{len(sequence_ids)} sequences, {total_observations} observations"
    )
    return total_observations


def recalculate_max_n_for_sequences(
    db: Session,
    sequence_ids: list[str],
    counting_threshold: float,
) -> None:
    """Recalculate MaxN for specific sequences (after verify/relabel)."""
    for sequence_id in sequence_ids:
        calculate_max_n_for_sequence(db, sequence_id, counting_threshold)


def get_sequence_ids_for_detections(
    db: Session,
    detection_ids: list[str],
) -> list[str]:
    """Get sequence IDs that contain files with the given detections."""
    rows = (
        db.query(sequence_files.c.sequence_id)
        .join(File, File.id == sequence_files.c.file_id)
        .join(Detection, Detection.file_id == File.id)
        .filter(Detection.id.in_(detection_ids))
        .distinct()
        .all()
    )
    return [row[0] for row in rows]


def get_sequence_ids_for_files(db: Session, file_ids: list[str]) -> list[str]:
    """Sequence IDs that contain any of the given files.

    By file, not by detection: a file whose boxes were just deleted has
    nothing left to join through.
    """
    if not file_ids:
        return []
    rows = (
        db.query(sequence_files.c.sequence_id)
        .filter(sequence_files.c.file_id.in_(file_ids))
        .distinct()
        .all()
    )
    return [row[0] for row in rows]


def get_max_n_frames(db: Session, sequence_id: str) -> list[dict]:
    """Get all MaxN frames for a sequence, ordered by max_n descending.

    Only rows with a frame (`max_n_file_id`) are returned, so human-only
    species (no AI frame) are excluded here. `effective_count` is the
    species total across all its cohort rows (the card chip reads it), not
    the frame row's own count: a species split into 1 + 1 says 2.
    """
    rows = (
        db.query(SequenceObservation)
        .filter(SequenceObservation.sequence_id == sequence_id)
        .order_by(SequenceObservation.max_n.desc())
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
            "max_n_frame_number": row.max_n_frame_number,
            "max_n_time": frame_time(row.max_n_file, row.max_n_frame_number),
        }
        for row in rows
        if row.max_n_file_id is not None
    ]


def first_arrivals(
    db: Session,
    counting_threshold: float,
    *,
    sequence_id: str | None = None,
    project_id: str | None = None,
    deployment_ids: list[str] | None = None,
) -> dict[str, dict[str | None, datetime]]:
    """When each species was first seen, per sequence: ``{sequence_id: {key:
    time}}`` with the key as the observation rows use it
    (`label_taxonomy_id`, else the label).

    The earliest passing, real box of the species over the sequence's files:
    for a photo that is the photo's time, for a video the frame's time
    (`frame_time`). Every box counts, not only the visible ones, because
    the first frame a tracker saw the animal is rarely the frame its card
    shows. One grouped query for the whole scope (a sequence, or a project's
    sequences, optionally limited to some deployments): the earliest frame
    per file and species, which is a handful of rows per sequence however
    many boxes a video holds. Derived on request; nothing stored. Species
    with no timed file are absent.
    """
    effective_label = func.coalesce(Detection.label, Detection.category)
    query = (
        db.query(
            sequence_files.c.sequence_id,
            Detection.label_taxonomy_id,
            effective_label.label("eff_label"),
            File.captured_at_local,
            File.frame_rate,
            func.min(Detection.frame_number).label("frame_number"),
        )
        .join(File, File.id == Detection.file_id)
        .join(sequence_files, sequence_files.c.file_id == File.id)
        .filter(_threshold_clause(counting_threshold))
        .filter(is_a_real_detection())
        .group_by(
            sequence_files.c.sequence_id,
            Detection.file_id,
            Detection.label_taxonomy_id,
            effective_label,
            File.captured_at_local,
            File.frame_rate,
        )
    )
    if sequence_id is not None:
        query = query.filter(sequence_files.c.sequence_id == sequence_id)
    if project_id is not None:
        query = query.join(Sequence, Sequence.id == sequence_files.c.sequence_id).join(
            Deployment, Deployment.id == Sequence.deployment_id
        )
        query = query.filter(Deployment.project_id == project_id)
        if deployment_ids is not None:
            query = query.filter(Deployment.id.in_(deployment_ids))

    first: dict[str, dict[str | None, datetime]] = defaultdict(dict)
    for r in query.all():
        seen = frame_time(r, r.frame_number)
        if seen is None:
            continue
        key = r.label_taxonomy_id or r.eff_label
        per_sequence = first[r.sequence_id]
        if key not in per_sequence or seen < per_sequence[key]:
            per_sequence[key] = seen
    return first


def _row(
    db: Session, sequence_observation_id: str, sequence_id: str | None
) -> SequenceObservation | None:
    """One row by id. With `sequence_id` the row must belong to that sequence, so
    a request whose path names another sequence touches nothing."""
    query = db.query(SequenceObservation).filter(
        SequenceObservation.id == sequence_observation_id
    )
    if sequence_id is not None:
        query = query.filter(SequenceObservation.sequence_id == sequence_id)
    return query.first()


def list_sequence_observations(
    db: Session, sequence_id: str
) -> list[SequenceObservation]:
    """All observation rows for a sequence (AI + human-only), grouped by
    species.

    Species are ordered by the AI MaxN (highest first), then label; species
    the AI did not detect come after, alphabetically. Within a species the
    AI row comes first and the cohorts follow in the order they were made
    (rowid). The key is deliberately independent of `human_count` and of the
    demographics, so editing a count or a dropdown never reshuffles the rows
    under the user's cursor.
    """
    rows = _rows_in_stored_order(db, sequence_id)
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
    sequence_observation_id: str,
    count: int | None,
    sequence_id: str | None = None,
) -> SequenceObservation | None:
    """Set (or clear, when count is None) the human count on one row.

    Clears the sequence's sign-off, since the counts just changed. Returns
    the updated row, or None when the id is unknown (or not in `sequence_id`).
    """
    obs = _row(db, sequence_observation_id, sequence_id)
    if obs is None:
        return None
    obs.human_count = count
    sequence = db.get(Sequence, obs.sequence_id)
    if sequence is not None:
        sequence.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def _resolve_taxonomy_id(
    db: Session, sequence_id: str, label: str | None, label_taxonomy_id: str | None
) -> str | None:
    """Resolve the taxonomy id from the label when not supplied, so a
    human-added species keys to the same row the AI would produce and
    doesn't split into a duplicate if the AI later detects it."""
    if label_taxonomy_id is not None or not label:
        return label_taxonomy_id
    from app.ml.taxonomy_db import resolve_taxonomy_id

    project_id = (
        db.query(Deployment.project_id)
        .join(Sequence, Sequence.deployment_id == Deployment.id)
        .filter(Sequence.id == sequence_id)
        .scalar()
    )
    if not project_id:
        return None
    return resolve_taxonomy_id(label, project_id, db)


def _human_species(
    db: Session,
    sequence_id: str,
    category: str | None,
    label: str | None,
    label_taxonomy_id: str | None,
) -> tuple[str, str | None, str | None]:
    """The (category, label, label_taxonomy_id) a row a person adds is
    stored with, in the shape `calculate_max_n_for_sequence` gives AI rows.

    The category follows `category_after_human_edit`: a species takes the
    detector's primary class, a picked detector class is itself. A class
    without a species gets ``label = category`` and its builtin row, as an
    AI row of that class has, so the two merge and the row keeps its key
    through a MaxN rebuild instead of being dropped by it.
    """
    from app.ml.primary_class import category_after_human_edit, project_primary_class
    from app.ml.taxonomy_db import ensure_builtin_labels

    def primary() -> str:
        project = (
            db.query(Project)
            .join(Deployment, Deployment.project_id == Project.id)
            .join(Sequence, Sequence.deployment_id == Deployment.id)
            .filter(Sequence.id == sequence_id)
            .one()
        )
        return project_primary_class(project)

    category = category_after_human_edit(None, category, label, primary)
    if not label and label_taxonomy_id is None:
        return category, category, ensure_builtin_labels(db, [category])[category]
    return category, label, _resolve_taxonomy_id(db, sequence_id, label, label_taxonomy_id)


def _find_cohort(
    db: Session,
    sequence_id: str,
    category: str,
    label: str | None,
    label_taxonomy_id: str | None,
    demographics: dict[str, str | None],
) -> SequenceObservation | None:
    """The sequence's row for this species with exactly these demographics
    (all None = the plain row), in stored order, or None."""
    query = db.query(SequenceObservation).filter(
        SequenceObservation.sequence_id == sequence_id
    )
    if label_taxonomy_id is not None:
        query = query.filter(
            SequenceObservation.label_taxonomy_id == label_taxonomy_id
        )
    else:
        query = query.filter(
            SequenceObservation.label_taxonomy_id.is_(None),
            SequenceObservation.label == label,
            SequenceObservation.category == category,
        )
    for field in DEMOGRAPHIC_FIELDS:
        value = demographics.get(field)
        column = getattr(SequenceObservation, field)
        query = query.filter(column.is_(None) if value is None else column == value)
    return query.order_by(text("sequence_observations.rowid")).first()


def add_human_species(
    db: Session,
    sequence_id: str,
    category: str | None,
    count: int,
    label: str | None = None,
    label_taxonomy_id: str | None = None,
    sex: str | None = None,
    life_stage: str | None = None,
    behavior: str | None = None,
) -> SequenceObservation:
    """Record a species the AI missed entirely.

    A human-only row is created (max_n=0, no frame) carrying the
    demographics, and the sequence's sign-off is cleared. When the sequence
    already has a row for this species with the same demographics,
    nothing changes and that row is returned: adding a species that is
    already counted must not overwrite its count (an AI MaxN of 3 used to
    become 1). The count is changed with + and - on the row itself.
    """
    category, label, label_taxonomy_id = _human_species(
        db, sequence_id, category, label, label_taxonomy_id
    )
    demographics = {"sex": sex, "life_stage": life_stage, "behavior": behavior}
    existing = _find_cohort(
        db, sequence_id, category, label, label_taxonomy_id, demographics
    )
    if existing is not None:
        return existing

    obs = SequenceObservation(
        id=str(uuid.uuid4()),
        sequence_id=sequence_id,
        label=label,
        label_taxonomy_id=label_taxonomy_id,
        category=category,
        max_n=0,
        max_n_file_id=None,
        human_count=count,
        **demographics,
    )
    db.add(obs)

    sequence = db.get(Sequence, sequence_id)
    if sequence is not None:
        sequence.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def set_observation_attributes(
    db: Session,
    sequence_observation_id: str,
    *,
    sex: str | None = UNSET,
    life_stage: str | None = UNSET,
    behavior: str | None = UNSET,
    sequence_id: str | None = None,
) -> SequenceObservation | None:
    """Set sex / life stage / behaviour on one row. A field left UNSET is
    not touched; None clears it.

    A change clears the sequence's sign-off, because it changes the recorded
    observation. Returns the row, or None when the id is unknown (or not
    in `sequence_id`).
    """
    obs = _row(db, sequence_observation_id, sequence_id)
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
        sequence = db.get(Sequence, obs.sequence_id)
        if sequence is not None:
            sequence.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def split_observation(
    db: Session, sequence_observation_id: str, sequence_id: str | None = None
) -> SequenceObservation | None:
    """Split one row into two cohorts of the same species.

    The source keeps its count minus one (never below one) and the new
    human-only row starts at one, with no demographics, so the
    user then sets what makes it different. Splitting a row at count one
    still gives two rows of one; the user corrects the numbers. Clears the
    sequence's sign-off. Returns the new row, or None when the id is unknown
    (or not in `sequence_id`).
    """
    source = _row(db, sequence_observation_id, sequence_id)
    if source is None:
        return None
    source.human_count = max(1, source.effective_count - 1)
    obs = SequenceObservation(
        id=str(uuid.uuid4()),
        sequence_id=source.sequence_id,
        label=source.label,
        label_taxonomy_id=source.label_taxonomy_id,
        category=source.category,
        max_n=0,
        max_n_file_id=None,
        human_count=1,
    )
    db.add(obs)
    sequence = db.get(Sequence, source.sequence_id)
    if sequence is not None:
        sequence.confirmed = False
    db.commit()
    db.refresh(obs)
    return obs


def relabel_observation(
    db: Session,
    sequence_observation_id: str,
    category: str | None,
    label: str | None = None,
    label_taxonomy_id: str | None = None,
    sequence_id: str | None = None,
) -> SequenceObservation | None:
    """Change the species of one count row, carrying its count to the target.

    Count-level relabel: the source row is removed the same way the panel's
    X does (a human-only row is deleted, an AI row keeps its boxes but its
    human_count drops to 0 so it hides and survives a MaxN recompute), and
    the source's effective count is moved onto the target species, keeping
    the source's demographics. If the target species already has
    a row with those demographics in the sequence, the counts SUM (bird(5)
    relabelled to deer, with deer already 1, gives deer 6); otherwise a
    human-only row is created for it. This edits counts only, not the
    underlying detections, exactly like add/remove on this panel. Clears the
    sequence's sign-off. Returns the target row, or None when the id is unknown
    (or not in `sequence_id`).
    """
    source = _row(db, sequence_observation_id, sequence_id)
    if source is None:
        return None
    sequence_id = source.sequence_id
    source_count = source.effective_count
    demographics = {f: getattr(source, f) for f in DEMOGRAPHIC_FIELDS}

    category, label, label_taxonomy_id = _human_species(
        db, sequence_id, category, label, label_taxonomy_id
    )

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
    # target sit in the same sequence.
    target = _find_cohort(
        db, sequence_id, category, label, label_taxonomy_id, demographics
    )
    target_existing = target.effective_count if target is not None else 0
    summed = target_existing + source_count

    # Remove the source (same semantics as the panel's X / count-to-zero).
    if source.max_n == 0:
        db.delete(source)
    else:
        source.human_count = 0

    # Move the count onto the target: the summed total on its existing
    # row, or a new human-only row via add_human_species, which also clears
    # the sign-off and commits.
    if target is not None:
        target.human_count = summed
        sequence = db.get(Sequence, sequence_id)
        if sequence is not None:
            sequence.confirmed = False
        db.commit()
        db.refresh(target)
        return target
    return add_human_species(
        db,
        sequence_id,
        category=category,
        count=summed,
        label=label,
        label_taxonomy_id=label_taxonomy_id,
        **demographics,
    )


def delete_sequence_observation(
    db: Session, sequence_observation_id: str, sequence_id: str | None = None
) -> str | None:
    """Remove the human contribution to one observation row.

    A human-only row (the AI detected nothing: max_n=0) is deleted
    outright; an AI row keeps its box-derived MaxN but drops the human
    override. Clears the sequence's sign-off. Returns the sequence id, or None
    when the id is unknown (or not in `sequence_id`).
    """
    obs = _row(db, sequence_observation_id, sequence_id)
    if obs is None:
        return None
    sequence_id = obs.sequence_id
    if obs.max_n == 0:
        db.delete(obs)
    else:
        obs.human_count = None
    sequence = db.get(Sequence, sequence_id)
    if sequence is not None:
        sequence.confirmed = False
    db.commit()
    return sequence_id


def set_sequence_notes(
    db: Session, sequence_id: str, notes: str | None
) -> Sequence | None:
    """Set a person's free text on a sequence. Never touches the sign-off:
    a note is commentary, not the recorded observation."""
    sequence = db.get(Sequence, sequence_id)
    if sequence is None:
        return None
    sequence.notes = (notes or "").strip() or None
    db.commit()
    db.refresh(sequence)
    return sequence


def set_sequence_confirmed(
    db: Session, sequence_id: str, confirmed: bool
) -> Sequence | None:
    """Set the human confirmation on a sequence's species and counts."""
    sequence = db.get(Sequence, sequence_id)
    if sequence is None:
        return None
    sequence.confirmed = confirmed
    db.commit()
    db.refresh(sequence)
    return sequence


def _clear_human_layer(db: Session, sequence_id: str) -> list[SequenceObservation]:
    """Drop every human edit to the sequence's counts in the session, no
    commit: delete the human-only rows (max_n == 0), clear `human_count`
    and the demographics on the AI rows. Returns the surviving AI rows in
    stored order."""
    ai_rows: list[SequenceObservation] = []
    for obs in _rows_in_stored_order(db, sequence_id):
        if obs.max_n == 0:
            db.delete(obs)
        else:
            for field, value in _human_layer(None).items():
                setattr(obs, field, value)
            ai_rows.append(obs)
    return ai_rows


def reset_sequence_to_ai(db: Session, sequence_id: str) -> Sequence | None:
    """Drop every human edit to the sequence's counts, back to the AI proposal.

    Clears `human_count` and the demographics on the AI rows and
    deletes the human-only rows (split-off cohorts and the species the AI
    never detected). Clears the sequence's sign-off. Returns the sequence, or None
    when the id is unknown.
    """
    sequence = db.get(Sequence, sequence_id)
    if sequence is None:
        return None
    _clear_human_layer(db, sequence_id)
    sequence.confirmed = False
    db.commit()
    db.refresh(sequence)
    return sequence


def set_human_observations(
    db: Session, sequence_id: str, rows: list[dict[str, Any]]
) -> Sequence | None:
    """Replace the sequence's human layer so its visible cohorts become
    exactly `rows` ("Same as last" on the Counts page).

    A demographic-free row lands on the sequence's AI row of that species
    when it has one, so the AI linkage and its MaxN survive; every other
    row becomes a human-only row. AI rows for species the rows do not
    cover get a human count of 0, which hides them and survives a
    recompute, the same way the X button removes a species. Creation is
    deliberate appends, never `_find_cohort`: matching would merge two
    identical cohorts (two "1 male adult" split rows) into one.

    Clears the sequence's sign-off. One commit. Returns the sequence, or None
    when the id is unknown.
    """
    sequence = db.get(Sequence, sequence_id)
    if sequence is None:
        return None
    ai_rows = _clear_human_layer(db, sequence_id)
    ai_by_key: dict[str | None, SequenceObservation] = {}
    for obs in ai_rows:
        obs.human_count = 0
        ai_by_key.setdefault(_species_key(obs), obs)
    seeded: set[str] = set()
    for row in rows:
        category, label, label_taxonomy_id = _human_species(
            db, sequence_id, row.get("category"), row.get("label"),
            row.get("label_taxonomy_id"),
        )
        demographics = {f: row.get(f) for f in DEMOGRAPHIC_FIELDS}
        key = label_taxonomy_id or label
        seed = ai_by_key.get(key)
        if (
            seed is not None
            and seed.id not in seeded
            and not any(demographics.values())
        ):
            seed.human_count = row["count"]
            seeded.add(seed.id)
        else:
            db.add(
                SequenceObservation(
                    id=str(uuid.uuid4()),
                    sequence_id=sequence_id,
                    label=label,
                    label_taxonomy_id=label_taxonomy_id,
                    category=category,
                    max_n=0,
                    max_n_file_id=None,
                    human_count=row["count"],
                    **demographics,
                )
            )
    sequence.confirmed = False
    db.commit()
    db.refresh(sequence)
    return sequence


def get_project_threshold_for_detections(
    db: Session,
    detection_ids: list[str],
) -> float:
    """The project ``counting_threshold`` owning the given detections.

    Requires at least one id, and requires it to resolve. Both used to
    fall back to ``0.0``, which is not a neutral value: it is the
    threshold at which every detection passes, including MegaDetector's
    near-noise tail down to its 0.01 output cap. A failed lookup therefore
    rebuilt a sequence's counts against the wrong floor and said nothing.

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
