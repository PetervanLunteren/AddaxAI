"""
CRUD operations for sequences.

Sequences are time-clustered groups of files within a deployment.
"""

import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time
from typing import NamedTuple

from sqlalchemy import Integer, delete, exists, func, insert, select, text
from sqlalchemy.orm import Session, joinedload, selectinload

from app.core.logging_config import get_logger
from app.db.sql_params import iter_id_chunks
from app.ml.detection_visibility import on_visible_frame, visible_detections
from app.ml.label_exclusion import (
    classification_score_in_range,
    threshold_or_verified,
)
from app.ml.observation_type import derive_observation_type
from app.models import Deployment, Detection, File, Project, Sequence
from app.models.sequence import sequence_files
from app.models.sequence_observation import SequenceObservation

logger = get_logger(__name__)

VERIFY_SORT_VALUES = {"newest", "oldest", "random", "cls_low"}

COLLAGE_TILE_LIMIT = 4


def _build_collage_file_ids(
    max_n_frames: list[dict],
    sorted_files: list[File],
) -> list[str]:
    """Pick up to four representative file IDs for a sequence card collage.

    The first slots come from `max_n_frames` (one file per dominant
    species, computed by `get_max_n_frames`). Remaining slots are
    padded with files of the sequence ranked by their highest detection
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


def _sequence_min_cls_subquery():
    """Correlated scalar subquery: MIN(label_confidence) across all detections
    on any file in the sequence. Used for the cls_low sort.
    """
    return (
        select(func.min(Detection.label_confidence))
        .select_from(sequence_files)
        .join(File, File.id == sequence_files.c.file_id)
        .join(Detection, Detection.file_id == File.id)
        .where(sequence_files.c.sequence_id == Sequence.id)
        .correlate(Sequence)
        .scalar_subquery()
    )


def _sequence_sort_spec(sort: str, seed: int | None):
    """Map (sort, seed) to (sort_key_expression, descending, nulls_last).

    The sort_key is a single SQLAlchemy expression that defines display order
    for the Sequences tab. The list query orders by it; the adjacent endpoint
    derives next/prev predicates from the same key.
    """
    if sort == "newest":
        return Sequence.sequence_start_local, True, False
    if sort == "oldest":
        return Sequence.sequence_start_local, False, False
    if sort == "random":
        if seed is None:
            raise ValueError("random sort requires a seed")
        return func.seeded_hash(Sequence.id, seed), False, False
    if sort == "cls_low":
        return _sequence_min_cls_subquery(), False, True
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
        # SQLite sorts NULL before every value: the NULL rows lead an ASC
        # list and trail a DESC one, ordered by id among themselves. The
        # predicates say so, because a plain comparison with NULL is
        # NULL: a sequence without a capture time (most BRUV clips) made
        # the adjacent endpoint answer 500 and the modal's arrows dead
        # (2026-11-03), and the NULL rows were unreachable from the
        # dated ones.
        is_null = sort_key.is_(None)
        not_null = sort_key.isnot(None)
        if current_value is None:
            if descending:
                next_pred = is_null & (id_col < current_id)
                prev_pred = not_null | (is_null & (id_col > current_id))
            else:
                next_pred = (is_null & (id_col > current_id)) | not_null
                prev_pred = is_null & (id_col < current_id)
        else:
            # Built only here: comparing a column with None is not SQL.
            after = (sort_key > current_value) | (
                (sort_key == current_value) & (id_col > current_id)
            )
            before = (sort_key < current_value) | (
                (sort_key == current_value) & (id_col < current_id)
            )
            if descending:
                next_pred = (not_null & before) | is_null
                prev_pred = not_null & after
            else:
                next_pred = not_null & after
                prev_pred = (not_null & before) | is_null
        if descending:
            next_order = [sort_key.desc(), id_col.desc()]
            prev_order = [sort_key.asc(), id_col.asc()]
        else:
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


def _sequence_box_clauses(
    project_floor: float | None,
    min_confidence: float | None,
    max_confidence: float | None,
    min_label_confidence: float | None,
    max_label_confidence: float | None,
) -> list:
    """What makes a box count for the Counts page's label and confidence
    filters: the project floor through `threshold_or_verified` (the
    global override rule), the user's detector range applied LITERALLY,
    and the classifier-score range on classified boxes only. Shared by
    the sequence filters and the label tree's counts."""
    clauses = []
    if project_floor is not None:
        clauses.append(threshold_or_verified(project_floor))
    if min_confidence is not None:
        clauses.append(Detection.confidence >= min_confidence)
    if max_confidence is not None:
        clauses.append(Detection.confidence <= max_confidence)
    if min_label_confidence is not None or max_label_confidence is not None:
        clauses.append(
            classification_score_in_range(min_label_confidence, max_label_confidence)
        )
    return clauses


def _apply_sequence_filters(
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
    """Apply shared filters to a sequence query. Expects Sequence already joined to Deployment.

    `flagged` / `favorited` filter at the file level (EXISTS a file in the
    sequence with File.flagged / File.favorited set). Per the decided mental
    model, flag and heart live on files; a sequence is flagged only in the
    sense that it contains at least one flagged file.

    `min_label_confidence` / `max_label_confidence` filter on the
    classifier score (Detection.label_confidence) through
    `classification_score_in_range`: classified boxes only, an
    unclassified box passes the range.

    `project_floor` is the project's `counting_threshold`, applied via
    `threshold_or_verified` (the global override rule). `min_confidence`
    (the user's slider) is
    applied LITERALLY without OR-verified — a verified low-confidence
    detection passes the floor but cannot satisfy a narrower user filter.

    `empty`:
      - "show_only": every file in the sequence has observation_type == "blank"
        (i.e. NOT EXISTS a non-blank file). Skips the labels and confidence
        gates because empty sequences have no detections by definition.
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
        query = query.filter(Sequence.sequence_start_local >= date_from)

    if date_to is not None:
        # Include the entire end-of-day
        end_of_day = (
            datetime.combine(date_to.date(), time.max) if isinstance(date_to, datetime) else date_to
        )
        query = query.filter(Sequence.sequence_start_local <= end_of_day)

    box_clauses = _sequence_box_clauses(
        project_floor, min_confidence, max_confidence,
        min_label_confidence, max_label_confidence,
    )
    if labels:
        # EXISTS subquery: sequence has at least one file with a detection
        # matching the given labels. Labels are taxonomy UUIDs.
        label_subq = (
            select(sequence_files.c.sequence_id)
            .join(File, File.id == sequence_files.c.file_id)
            .join(Detection, Detection.file_id == File.id)
            .where(sequence_files.c.sequence_id == Sequence.id)
            .where(Detection.label_taxonomy_id.in_(labels))
            .where(*box_clauses)
        )
        query = query.filter(exists(label_subq))
    elif box_clauses:
        # Standalone confidence filter: sequence has at least one detection in range
        conf_subq = (
            select(sequence_files.c.sequence_id)
            .join(File, File.id == sequence_files.c.file_id)
            .join(Detection, Detection.file_id == File.id)
            .where(sequence_files.c.sequence_id == Sequence.id)
            .where(*box_clauses)
        )
        query = query.filter(exists(conf_subq))

    if verification in ("verified", "unverified"):
        # Sequence verification is an explicit human sign-off stored on
        # Sequence.confirmed (set on the Observations page when the species
        # and counts are confirmed). Distinct from Detection.verified.
        if verification == "verified":
            query = query.filter(Sequence.confirmed == True)  # noqa: E712
        else:
            query = query.filter(Sequence.confirmed == False)  # noqa: E712

    if flagged in ("flagged", "not_flagged"):
        flagged_subq = (
            select(sequence_files.c.sequence_id)
            .join(File, File.id == sequence_files.c.file_id)
            .where(sequence_files.c.sequence_id == Sequence.id)
            .where(File.flagged == True)  # noqa: E712
        )
        if flagged == "flagged":
            query = query.filter(exists(flagged_subq))
        else:
            query = query.filter(~exists(flagged_subq))

    if favorited in ("favorited", "not_favorited"):
        favorited_subq = (
            select(sequence_files.c.sequence_id)
            .join(File, File.id == sequence_files.c.file_id)
            .where(sequence_files.c.sequence_id == Sequence.id)
            .where(File.favorited == True)  # noqa: E712
        )
        if favorited == "favorited":
            query = query.filter(exists(favorited_subq))
        else:
            query = query.filter(~exists(favorited_subq))

    if empty in ("show_only", "hide"):
        non_blank_file_subq = (
            select(sequence_files.c.sequence_id)
            .join(File, File.id == sequence_files.c.file_id)
            .where(sequence_files.c.sequence_id == Sequence.id)
            .where(File.observation_type != "blank")
        )
        if empty == "show_only":
            # All files blank → no non-blank file exists
            query = query.filter(~exists(non_blank_file_subq))
        else:
            # At least one non-blank file
            query = query.filter(exists(non_blank_file_subq))

    return query


@dataclass
class _SequenceCarry:
    """The human layer of a sequence, kept across a regeneration so a new
    sequence covering the same files can inherit it."""

    confirmed: bool
    prior: list  # list[PriorObs]

    @property
    def worked_on(self) -> bool:
        """A person confirmed the sequence or set a count, sex, age or
        behaviour on it. A note alone does not count: notes survive any
        regroup anyway."""
        return self.confirmed or any(
            p.human_count is not None or p.sex or p.life_stage or p.behavior
            for p in self.prior
        )


@dataclass
class _RegroupCarry:
    """What survives a regeneration, with two different rules.

    Counts and the sign-off (`by_fileset`) are claims about one exact file
    set, so they carry only onto a new sequence with the same files; a merged
    or split sequence gets the AI's numbers back and is unconfirmed. Notes are
    free text nobody can rebuild, so they are never lost: a new sequence
    inherits the notes of every old sequence it shares a file with. A split
    duplicates the note onto every child, a merge joins the notes in time
    order, one per line."""

    by_fileset: dict[frozenset, _SequenceCarry]
    old_sequence_by_file: dict[str, str]
    notes_by_old_sequence: dict[str, tuple[datetime | None, str, str]]

    def inherited_notes(self, file_ids: Iterable[str]) -> str | None:
        old_ids = {
            self.old_sequence_by_file[f]
            for f in file_ids
            if f in self.old_sequence_by_file
        }
        found = sorted(
            self.notes_by_old_sequence[i] for i in old_ids if i in self.notes_by_old_sequence
        )
        if not found:
            return None
        return "\n".join(note for _start, _id, note in found)


def _snapshot_sequence_carry(
    db: Session, deployment_ids: list[str]
) -> _RegroupCarry:
    """Snapshot each sequence's confirmed flag + observation rows, keyed by its
    exact set of file IDs (files are partitioned across sequences, so the keys
    are unique), plus every note keyed by file so it can be carried by
    overlap. See `_RegroupCarry` for the two rules."""
    from app.api.crud.sequence_observation import PriorObs

    carry = _RegroupCarry({}, {}, {})
    if not deployment_ids:
        return carry

    sequences = (
        db.query(Sequence.id, Sequence.confirmed, Sequence.notes, Sequence.sequence_start_local)
        .filter(Sequence.deployment_id.in_(deployment_ids))
        .all()
    )
    if not sequences:
        return carry
    sequence_ids = [e.id for e in sequences]
    for e in sequences:
        if e.notes:
            carry.notes_by_old_sequence[e.id] = (e.sequence_start_local, e.id, e.notes)

    # Chunk the sequence-id filters: one `IN (?, ?, ...)` over every sequence blows
    # SQLite's bound-parameter limit on a large project (see app/db/sql_params).
    files_by_sequence: dict[str, set[str]] = defaultdict(set)
    for chunk in iter_id_chunks(sequence_ids):
        for eid, fid in db.query(
            sequence_files.c.sequence_id, sequence_files.c.file_id
        ).filter(sequence_files.c.sequence_id.in_(chunk)):
            files_by_sequence[eid].add(fid)

    obs_by_sequence: dict[str, list] = defaultdict(list)
    for chunk in iter_id_chunks(sequence_ids):
        # rowid order: the cohorts of a species keep their order through
        # the regeneration (see `_rows_in_stored_order`).
        for o in (
            db.query(SequenceObservation)
            .filter(SequenceObservation.sequence_id.in_(chunk))
            .order_by(text("sequence_observations.rowid"))
        ):
            obs_by_sequence[o.sequence_id].append(
                PriorObs(
                    label=o.label,
                    label_taxonomy_id=o.label_taxonomy_id,
                    category=o.category,
                    max_n=o.max_n,
                    human_count=o.human_count,
                    effective_count=o.effective_count,
                    sex=o.sex,
                    life_stage=o.life_stage,
                    behavior=o.behavior,
                )
            )

    for e in sequences:
        fileset = frozenset(files_by_sequence.get(e.id, set()))
        if fileset:
            carry.by_fileset[fileset] = _SequenceCarry(
                confirmed=e.confirmed, prior=obs_by_sequence.get(e.id, [])
            )
        for fid in fileset:
            carry.old_sequence_by_file[fid] = e.id
    return carry


def generate_sequences_for_project(db: Session, project_id: str) -> int:
    """
    Generate sequences for all deployments in a project.

    Idempotent: deletes existing sequences before regenerating. Clustering
    logic lives in `app.services.sequence_clustering.cluster_files_into_sequences`
    — the single source of truth shared with the smoothing adapter, so
    sequences and smoother inputs never disagree.

    Human confirmations (`Sequence.confirmed`) and manual counts
    (`SequenceObservation.human_count`) are carried onto any regenerated sequence
    whose file set is unchanged, so a reprocess that doesn't actually
    regroup a sequence never loses its count verification. Sequences that merge
    or split find no match and reset to unconfirmed AI counts.

    Returns total sequence count created.
    """
    from app.models import Project

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    threshold = project.counting_threshold

    # Get all deployments for this project
    deployments = (
        db.query(Deployment).filter(Deployment.project_id == project_id).all()
    )
    deployment_ids = [d.id for d in deployments]

    # Snapshot the human layer before deleting, then re-attach by file set.
    carry = _snapshot_sequence_carry(db, deployment_ids)

    # Delete existing sequences for all deployments in this project. Chunked
    # like the sequence-id filters above: a project with enough deployments
    # would otherwise blow SQLite's bound-parameter limit.
    for chunk in iter_id_chunks(deployment_ids):
        db.execute(delete(Sequence).where(Sequence.deployment_id.in_(chunk)))

    total_sequences = 0
    for deployment in deployments:
        total_sequences += _regenerate_deployment_sequences(
            db, deployment, threshold, carry
        )

    db.flush()
    db.commit()
    # Fill the events cache now, not on the next page visit: a project-wide
    # regroup would otherwise make that visit wait for all of it.
    from app.api.crud.independent_events import refresh

    refresh(db, project_id)
    return total_sequences


def generate_sequences_for_deployment(db: Session, deployment: Deployment) -> int:
    """
    Regenerate the sequences of one deployment, carrying the human layer
    like `generate_sequences_for_project` does. Used at the end of an
    analysis, which only touches its own deployment, and when a
    deployment-level setting that changes the grouping (`paired_cameras`,
    camera offsets) is edited.

    Does NOT commit: the caller owns the transaction. Returns the number
    of sequences created.
    """
    project = deployment.project
    carry = _snapshot_sequence_carry(db, [deployment.id])
    db.execute(delete(Sequence).where(Sequence.deployment_id == deployment.id))
    count = _regenerate_deployment_sequences(
        db, deployment, project.counting_threshold, carry
    )
    db.flush()
    return count


def _regenerate_deployment_sequences(
    db: Session,
    deployment: Deployment,
    threshold: float,
    carry: _RegroupCarry,
) -> int:
    """Rebuild one deployment's sequences from its files. The caller has
    already snapshotted `carry` and deleted the old Sequence rows. No commit."""
    from app.api.crud.sequence_observation import calculate_max_n_for_sequence
    from app.services.sequence_clustering import cluster_files_into_sequences

    files = (
        db.query(File)
        .options(joinedload(File.source_video))
        .filter(File.deployment_id == deployment.id)
        .filter(File.file_type.in_(["image", "video"]))
        .all()
    )

    clusters = _keep_worked_on_whole(
        cluster_files_into_sequences(files, paired_cameras=deployment.paired_cameras),
        carry,
    )
    count = 0
    for cluster in clusters:
        sequence = _create_sequence(db, deployment, cluster)
        file_ids = [f.id for f in cluster]
        carried = carry.by_fileset.get(frozenset(file_ids))
        if carried is not None:
            sequence.confirmed = carried.confirmed
        sequence.notes = carry.inherited_notes(file_ids)
        # Rebuild MaxN, carrying the human layer for a matched sequence.
        # calculate_max_n_for_sequence still clears `confirmed` if the
        # species/count set changed (e.g. smoothing relabelled a crop).
        calculate_max_n_for_sequence(
            db,
            sequence.id,
            threshold,
            prior=carried.prior if carried is not None else None,
        )
        count += 1
    return count


def _keep_worked_on_whole(
    clusters: list[list[File]], carry: _RegroupCarry
) -> list[list[File]]:
    """Never cut a group somebody worked on.

    Where the new rule would only split an old group that was confirmed or
    has a manual count or demographics (its files land in more than one
    new cluster and in nothing else), the clusters are joined back into
    that group, so the existing carry finds the same file set and keeps
    the person's work. This is what keeps counts confirmed before
    sequences existed (30-minute groups) whole through every later
    regroup. A merge or a mixed change (files added, removed, or joined
    with another group) is not a split and resets as before.
    """
    cluster_of = {f.id: i for i, cluster in enumerate(clusters) for f in cluster}
    joined: dict[int, int] = {}  # cluster index -> index it was joined into
    for fileset, old in carry.by_fileset.items():
        if not old.worked_on or not fileset <= cluster_of.keys():
            continue
        touched = sorted({cluster_of[f] for f in fileset})
        if len(touched) < 2 or sum(len(clusters[i]) for i in touched) != len(fileset):
            continue
        for i in touched:
            joined[i] = touched[0]

    if not joined:
        return clusters
    out: dict[int, list[File]] = {}
    for i, cluster in enumerate(clusters):
        out.setdefault(joined.get(i, i), []).extend(cluster)
    for i in set(joined.values()):
        out[i].sort(key=lambda f: (f.captured_at_local or datetime.max, f.file_path))
    return list(out.values())


def _create_sequence(db: Session, deployment: Deployment, files: list[File]) -> Sequence:
    """Create a sequence with its junction table entries. It ends when its
    last file ends: a video at its start plus its length. A dateless file is
    a sequence of its own, with no bounds."""
    from app.services.sequence_clustering import camera_key, file_end

    dated = [f for f in files if f.captured_at_local is not None]
    sequence = Sequence(
        id=str(uuid.uuid4()),
        deployment_id=deployment.id,
        camera_key=camera_key(files[0], paired_cameras=deployment.paired_cameras),
        sequence_start_local=dated[0].captured_at_local if dated else None,
        sequence_end_local=max(map(file_end, dated)) if dated else None,
        file_count=len(files),
    )
    db.add(sequence)
    db.flush()  # Get sequence.id assigned

    # Insert junction table rows
    for seq, file in enumerate(files):
        db.execute(
            insert(sequence_files).values(
                sequence_id=sequence.id,
                file_id=file.id,
                position=seq,
            )
        )

    return sequence


def get_sequences_by_project(
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
    Get sequence summaries for a project.

    Returns sequence data with representative file, label list,
    observation type, and verification progress.
    """
    query = (
        db.query(Sequence)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
    )
    query = _apply_sequence_filters(
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
    sort_key, descending, nulls_last = _sequence_sort_spec(sort, seed)
    order_clauses = _sort_order_by_clauses(
        sort_key, Sequence.id, descending=descending, nulls_last=nulls_last,
    )
    sequences = (
        query.options(selectinload(Sequence.files).selectinload(File.detections))
        .order_by(*order_clauses)
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Deduplicate sequences (joinedload can produce duplicates)
    seen_ids: set[str] = set()
    unique_sequences: list[Sequence] = []
    for sequence in sequences:
        if sequence.id not in seen_ids:
            seen_ids.add(sequence.id)
            unique_sequences.append(sequence)

    # Batch-load MaxN frames for all sequences in this page
    from app.api.crud.sequence_observation import get_max_n_frames

    sequence_ids = [e.id for e in unique_sequences]
    max_n_by_sequence: dict[str, list[dict]] = {}
    for eid in sequence_ids:
        max_n_by_sequence[eid] = get_max_n_frames(db, eid)

    # Gallery-card chips must match the Counts panel's species. Source the
    # allowed species per sequence from its SequenceObservation rows, which
    # already apply the best-frame gate for videos (see
    # calculate_max_n_for_sequence). Without this, non-best-frame per-frame
    # classifier noise leaks into the card chips even though it's correctly
    # excluded from the count suggestion.
    allowed_keys_by_sequence: dict[str, set[str]] = defaultdict(set)
    if sequence_ids:
        obs_rows = (
            db.query(
                SequenceObservation.sequence_id,
                SequenceObservation.label_taxonomy_id,
                SequenceObservation.label,
            )
            .filter(SequenceObservation.sequence_id.in_(sequence_ids))
            .all()
        )
        for eid, tid, lbl in obs_rows:
            key = tid or lbl
            if key is not None:
                allowed_keys_by_sequence[eid].add(key)

    summaries = []
    for sequence in unique_sequences:
        # Sort files by sequence within sequence. Same-second bursts break
        # ties on file_path (sequential camera filenames = capture order).
        sorted_files = sorted(
            sequence.files,
            key=lambda f: (f.captured_at_local, f.file_path),
        )

        # Collect unique taxonomy IDs across all files. Both name maps are
        # keyed the same way so the client can switch without a refetch.
        label_set: set[str] = set()
        label_to_scientific: dict[str, str] = {}
        label_to_common: dict[str, str] = {}
        allowed_keys = allowed_keys_by_sequence.get(sequence.id, set())
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
                    # A box without a species points at its class's
                    # builtin row, so a box with no row has a label.
                    tid = d.label_taxonomy_id
                    raw = d.label
                    # Same gate as the count suggestion: only species the
                    # sequence's SequenceObservation rows kept (best-frame gated
                    # for videos) become chips.
                    if (tid or raw) not in allowed_keys:
                        continue
                    if tid:
                        label_set.add(tid)
                        if tid not in label_to_scientific:
                            sci = d.scientific_name or d.label
                            if sci:
                                label_to_scientific[tid] = sci
                        if tid not in label_to_common:
                            common = d.common_name or d.label
                            if common:
                                label_to_common[tid] = common
                    else:
                        label_set.add(raw)
                        if d.scientific_name and raw not in label_to_scientific:
                            label_to_scientific[raw] = d.scientific_name
                        if d.common_name and raw not in label_to_common:
                            label_to_common[raw] = d.common_name

        # The sequence's dominant observation type is the same rule applied
        # one level up: the category of the strongest detection anywhere
        # in the sequence. This used to be a second priority table with its
        # own weights ({"animal": 4, "human": 3, ...}), a duplicate of
        # the one in observation_type.py that could drift from it. Now
        # there is one rule and no vocabulary to keep in step.
        observation_types_set: set[str] = {
            f.observation_type for f in sorted_files if f.observation_type
        }
        # Each file contributes only its visible surface, so a video is
        # represented by its best frame here too. Flattening the raw
        # relationship would let an off-frame box name the whole sequence,
        # which the species chips built just above already refuse to do.
        dominant_type = derive_observation_type(
            (
                d
                for f in sorted_files
                for d in visible_detections(f, f.detections)
            ),
            project_floor if project_floor is not None else 0.0,
        )

        # Count files by type and verification. Post-frame-row-removal
        # (2026-05) only "image" and "video" file types exist. The
        # `frame_count` field is kept on the response shape so the
        # frontend doesn't break; it counts distinct frame_numbers from
        # video detections, which is the natural successor metric.
        image_count = sum(1 for f in sorted_files if f.file_type == "image")
        video_count = sum(1 for f in sorted_files if f.file_type == "video")
        frame_count = len(
            {
                (d.file_id, d.frame_number)
                for f in sorted_files
                if f.file_type == "video"
                for d in (f.detections or [])
                if d.frame_number is not None
            }
        )
        verified_count = sum(1 for f in sorted_files if f.verified)
        any_file_flagged = any(f.flagged for f in sorted_files)
        any_file_favorited = any(f.favorited for f in sorted_files)

        # MaxN verification counts
        max_n_frames = max_n_by_sequence.get(sequence.id, [])
        file_verified_map = {f.id: f.verified for f in sorted_files}
        total_maxn_count = len(max_n_frames)
        verified_maxn_count = sum(
            1
            for mf in max_n_frames
            if file_verified_map.get(mf["file_id"], False)
        )

        # Sequence verification is the explicit human sign-off on species and
        # counts, stored on Sequence.confirmed (set on the Observations page).
        # The file-level verified_* counts above stay as secondary detail.
        is_confirmed = sequence.confirmed

        # MaxN-derived thumbnail: dominant species' MaxN frame, fallback to first file
        thumbnail_file_id = max_n_frames[0]["file_id"] if max_n_frames else (
            sorted_files[0].id if sorted_files else None
        )

        collage_file_ids = _build_collage_file_ids(max_n_frames, sorted_files)

        summaries.append(
            {
                "id": sequence.id,
                "deployment_id": sequence.deployment_id,
                "sequence_start_local": sequence.sequence_start_local,
                "sequence_end_local": sequence.sequence_end_local,
                "file_count": sequence.file_count,
                "thumbnail_file_id": thumbnail_file_id,
                "collage_file_ids": collage_file_ids,
                "max_n_frames": max_n_frames,
                "site_name": sequence.deployment.site.name
                if sequence.deployment and sequence.deployment.site
                else None,
                "labels": sorted(label_set),
                "scientific_labels": dict(label_to_scientific),
                "common_labels": dict(label_to_common),
                "observation_type": dominant_type,
                "observation_types": sorted(observation_types_set),
                "image_count": image_count,
                "frame_count": frame_count,
                "video_count": video_count,
                "verified_count": verified_count,
                "total_count": len(sorted_files),
                "verified_maxn_count": verified_maxn_count,
                "total_maxn_count": total_maxn_count,
                "is_confirmed": is_confirmed,
                "any_file_flagged": any_file_flagged,
                "any_file_favorited": any_file_favorited,
            }
        )

    return summaries


def get_sequence_with_files(db: Session, sequence_id: str) -> Sequence | None:
    """
    Get sequence with its files and their tracks. The boxes are not loaded
    here: the router hands each file to `file_payload` in the "cards"
    scope, and the viewer fetches the full file it focuses on itself.
    Until 2026-11-02 every box of every file came along, 28 MB for one
    BRUV clip, and the Counts modal waited 15 s on it.

    Files and tracks load as IN-selects, never joined onto the sequence
    row: joined, a tracked hour of video (12,700 boxes) took 30 seconds
    to come back, and the sequences list did the same per page.
    """
    sequence = (
        db.query(Sequence)
        .options(selectinload(Sequence.files).selectinload(File.tracks))
        .options(joinedload(Sequence.deployment).joinedload(Deployment.site))
        .filter(Sequence.id == sequence_id)
        .first()
    )
    return sequence


class SequenceCounts(NamedTuple):
    total: int
    confirmed: int


def get_sequence_counts(
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
) -> SequenceCounts:
    """How many sequences pass the filters, and how many of those a person
    signed off on the Counts page (`Sequence.confirmed`). One statement."""
    query = (
        db.query(
            func.count(Sequence.id),
            func.coalesce(func.sum(func.cast(Sequence.confirmed, Integer)), 0),
        )
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
    )
    query = _apply_sequence_filters(
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
    total, confirmed = query.one()
    return SequenceCounts(total=total, confirmed=int(confirmed))


def get_adjacent_sequences(
    db: Session,
    sequence_id: str,
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
    Get adjacent sequence IDs for navigation.

    Returns previous_id, next_id, next_unconfirmed_id, current_index, total_count.
    Order matches `get_sequences_by_project` for the same `(sort, seed)`. The
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

    sort_key, descending, nulls_last = _sequence_sort_spec(sort, seed)

    # 1. Get current sequence's sort key value.
    current_value = (
        db.query(sort_key).filter(Sequence.id == sequence_id).scalar()
    )
    current = db.query(Sequence.id).filter(Sequence.id == sequence_id).first()
    if not current:
        return {
            "previous_id": None,
            "next_id": None,
            "next_unconfirmed_id": None,
            "current_index": 0,
            "total_count": 0,
        }

    cid = current.id

    next_pred, prev_pred, next_order, prev_order = _sort_adjacency_predicates(
        sort_key, Sequence.id, current_value, cid,
        descending=descending, nulls_last=nulls_last,
    )

    def base():
        q = (
            db.query(Sequence.id)
            .join(Deployment)
            .filter(Deployment.project_id == project_id)
        )
        return _apply_sequence_filters(q, db, **filter_kwargs)

    # 2. Previous (one row above current in display order).
    prev = base().filter(prev_pred).order_by(*prev_order).first()

    # 3. Next (one row below current in display order).
    nxt = base().filter(next_pred).order_by(*next_order).first()

    # 4. Next unconfirmed (next row in display order whose species and
    # counts have not been confirmed). Uses base() so all active filters
    # apply. This drives the "next sequence to confirm" jump on the
    # Observations page.
    nxt_unconf = (
        base()
        .filter(next_pred)
        .filter(Sequence.confirmed == False)  # noqa: E712
        .order_by(*next_order)
        .first()
    )

    # 5a. Total count
    total_q = (
        db.query(func.count(Sequence.id))
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
    )
    total_q = _apply_sequence_filters(total_q, db, **filter_kwargs)
    total = total_q.scalar() or 0

    # 5b. Current index = number of rows that come BEFORE current in display
    # order. This is the prev_pred set, plus current itself if you want
    # 1-based; we use 0-based to match the existing UI.
    idx_q = (
        db.query(func.count(Sequence.id))
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .filter(prev_pred)
    )
    idx_q = _apply_sequence_filters(idx_q, db, **filter_kwargs)
    idx = idx_q.scalar() or 0

    return {
        "previous_id": prev[0] if prev else None,
        "next_id": nxt[0] if nxt else None,
        "next_unconfirmed_id": nxt_unconf[0] if nxt_unconf else None,
        "current_index": idx,
        "total_count": total,
    }


def get_sequence_verification_stats(
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
    """The reprocess dialog's "what is protected" numbers: the filtered
    sequences and how many are confirmed, and the cards on those sequences and
    how many are verified. A caller that needs the sequences only uses
    `get_sequence_counts`, which skips the pass over the detections."""
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

    # Base: filtered sequence IDs
    sequence_ids_q = (
        db.query(Sequence.id)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
    )
    sequence_ids_q = _apply_sequence_filters(sequence_ids_q, db, **filter_kwargs)
    sequence_ids_subq = sequence_ids_q.subquery()

    # Per-detection counts. Denominator and numerator share the same
    # filter so the ratio is meaningful.
    #
    # Same visible surface as the Labels grid these numbers describe. A
    # video stores a detection per sampled frame but only its best frame
    # is renderable, so without this the denominator counts labels the
    # user can never open: a video project sat at 19% with every card on
    # screen already verified, and could not have reached 100%. Verified
    # detections pass on any frame, so only the denominator moves.
    det_q = (
        db.query(
            func.count(Detection.id),
            func.sum(func.cast(Detection.verified, Integer)),
        )
        .join(File, File.id == Detection.file_id)
        .join(sequence_files, sequence_files.c.file_id == File.id)
        .filter(sequence_files.c.sequence_id.in_(select(sequence_ids_subq.c.id)))
        .filter(on_visible_frame())
    )
    if project_floor is not None:
        det_q = det_q.filter(threshold_or_verified(project_floor))
    if min_confidence is not None:
        det_q = det_q.filter(Detection.confidence >= min_confidence)
    total_detections, verified_detections = det_q.one()

    sequences = get_sequence_counts(db, project_id, **filter_kwargs)

    return {
        "sequences_confirmed": sequences.confirmed,
        "sequences_total": sequences.total,
        "total_detections": int(total_detections or 0),
        "verified_detections": int(verified_detections or 0),
    }


def _present_boxes(query, project_id: str, threshold: float):
    """Narrow a query over `Detection` to the boxes that make a label
    present in the project: labelled, at or above ``threshold`` or
    verified, on the visible frame. The one definition, shared by
    `present_label_rows` and the filter options."""
    return (
        query.join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(Detection.label_taxonomy_id.isnot(None))
        .filter(threshold_or_verified(threshold))
        # Same visible surface as the grid. Without it the label filter
        # offers species that live only on frames the grid never renders,
        # so picking one returns nothing.
        .filter(on_visible_frame())
    )


def present_label_rows(
    db: Session, project_id: str, threshold: float, *columns
) -> list:
    """Distinct ``label_taxonomy_id`` values the project's grid can show.

    The one definition of "species present in this project"
    (`_present_boxes`). The label filter and the species colour map both
    read it, so a species can never be offered as a filter without a
    colour or the other way round. Extra ``columns`` ride along in the
    same distinct query; the taxonomy id is always the first column.
    """
    return (
        _present_boxes(
            db.query(Detection.label_taxonomy_id, *columns), project_id, threshold
        )
        .distinct()
        .all()
    )


def get_filter_options(db: Session, project_id: str) -> dict:
    """Everything about the labels a project has, for the filter bars and
    the species colours: which labels (with both names, the colour map
    and the filter tree), the sequence date range, and the lowest
    classification score among the labels' boxes.

    One pass over the project's boxes. The counts per label under a
    page's filters are a separate request (`label_tree.
    count_labels_under_filters`), because they change with every filter
    and this does not.
    """
    from app.api.crud.label_colors import colors_for_labels
    from app.api.crud.label_tree import build_label_filter_tree

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id!r} not found")
    threshold = project.counting_threshold

    scientific = func.coalesce(Detection.scientific_name, Detection.label)
    common = func.coalesce(Detection.common_name, Detection.label)
    label_rows = (
        _present_boxes(
            db.query(
                Detection.label_taxonomy_id,
                scientific,
                common,
                func.min(Detection.label_confidence),
            ),
            project_id,
            threshold,
        )
        .group_by(Detection.label_taxonomy_id, scientific, common)
        .all()
    )

    # Name maps (taxonomy id -> name), one per display preference.
    scientific_labels: dict[str, str] = {}
    common_labels: dict[str, str] = {}
    for tid, sci, com, _ in label_rows:
        if sci and tid not in scientific_labels:
            scientific_labels[tid] = sci
        if com and tid not in common_labels:
            common_labels[tid] = com
    label_ids = sorted({row[0] for row in label_rows})

    # The lowest classification score among the boxes the pages can show.
    # The filter bars clamp the classification range slider there:
    # dragging below it would select nothing. None when nothing is
    # classified yet.
    scores = [row[3] for row in label_rows if row[3] is not None]
    min_label_confidence = min(scores) if scores else None

    date_row = (
        db.query(
            func.min(Sequence.sequence_start_local),
            func.max(Sequence.sequence_start_local),
        )
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .first()
    )
    date_range = None
    if date_row and date_row[0] and date_row[1]:
        date_range = {"min": date_row[0], "max": date_row[1]}

    return {
        "labels": label_ids,
        "date_range": date_range,
        "scientific_labels": scientific_labels,
        "common_labels": common_labels,
        "min_label_confidence": min_label_confidence,
        "label_colors": colors_for_labels(db, label_ids),
        "label_tree": build_label_filter_tree(db, set(label_ids)),
    }


def count_sequences_by_label(
    db: Session,
    project_id: str,
    *,
    project_floor: float | None,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    verification: str | None = None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
) -> dict[str, int]:
    """Sequences per label under every filter except the label filter: what
    the Counts page would list if that label were ticked alone. The
    sequence set comes from the page's own filters, the boxes from its own
    clauses, so the two cannot disagree.

    "Show only empty" is the one exception: the page then ignores the
    label filter altogether (an empty sequence has no species), so every
    label reads 0 rather than the number of empty sequences.
    """
    if empty == "show_only":
        return {}
    sequences = _apply_sequence_filters(
        db.query(Sequence).join(Deployment).filter(Deployment.project_id == project_id),
        db,
        site_ids=site_ids,
        date_from=date_from,
        date_to=date_to,
        labels=None,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        project_floor=project_floor,
    ).with_entities(Sequence.id).subquery()
    rows = (
        db.query(
            Detection.label_taxonomy_id,
            func.count(func.distinct(sequence_files.c.sequence_id)),
        )
        .join(File, File.id == Detection.file_id)
        .join(sequence_files, sequence_files.c.file_id == File.id)
        .filter(sequence_files.c.sequence_id.in_(select(sequences.c.id)))
        .filter(Detection.label_taxonomy_id.isnot(None))
        .filter(
            *_sequence_box_clauses(
                project_floor, min_confidence, max_confidence,
                min_label_confidence, max_label_confidence,
            )
        )
        .group_by(Detection.label_taxonomy_id)
        .all()
    )
    return {tid: n for tid, n in rows}
