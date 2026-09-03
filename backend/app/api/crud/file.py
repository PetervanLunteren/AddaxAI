"""
CRUD operations for files.
"""

from collections.abc import Iterable
from datetime import UTC, datetime, time
from typing import NamedTuple

from sqlalchemy import Integer, and_, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.api.crud.detection import mark_detections_false
from app.api.crud.event_observation import (
    get_event_ids_for_files,
    recalculate_max_n_for_events,
)
from app.api.schemas.file import FileUpdate
from app.core.confidence import effective_floor
from app.ml.detection_visibility import on_visible_frame, on_visible_frame_of
from app.ml.label_exclusion import is_a_real_detection, threshold_or_verified
from app.ml.observation_type import derive_observation_type
from app.models import Deployment, Detection, Event, File, Project
from app.models.event import event_files


def _project_threshold_for_file(db: Session, file: File) -> float:
    """The detection threshold of the project a file belongs to.

    ``observation_type`` counts only detections at or above this (or
    verified), so the threshold has to come from the owning project.

    Raises when the chain is broken. It cannot be: ``File.deployment_id``
    and ``Deployment.project_id`` are both ``NOT NULL`` with
    ``ON DELETE CASCADE``, and ``PRAGMA foreign_keys=ON`` is set on every
    connection, so a file without a project means a corrupt database.
    This used to return ``0.0``, which is not a neutral fallback: it is
    the value at which *every* detection passes, including MegaDetector's
    near-noise tail down to its 0.01 output cap. A broken lookup silently
    reclassified files and recomputed counts against the wrong floor
    instead of saying anything.
    """
    row = (
        db.query(Project.counting_threshold)
        .join(Deployment, Deployment.project_id == Project.id)
        .filter(Deployment.id == file.deployment_id)
        .first()
    )
    if row is None:
        raise ValueError(
            f"File {file.id} has no reachable project via deployment "
            f"{file.deployment_id}. Refusing to guess a detection "
            f"threshold."
        )
    return row[0]


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
    # file_path tiebreak (here and in the sibling queries below): burst
    # shots share one second-resolution timestamp, and offset pagination
    # over a non-unique sort key can skip or repeat rows across pages.
    return (
        query.order_by(File.captured_at_local.desc(), File.file_path.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


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
    return (
        query.order_by(File.captured_at_local.desc(), File.file_path.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


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
    return (
        query.order_by(File.captured_at_local.desc(), File.file_path.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


LABELS_FILES_SORT_VALUES = frozenset({"path", "newest", "oldest", "random"})


def get_labels_files(
    db: Session,
    project_id: str,
    *,
    floor: float,
    empty: str = "all",
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    verification: str | None = None,
    flagged: str | None = None,
    favorited: str | None = None,
    labels: list[str] | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    sort: str = "path",
    seed: int | None = None,
    skip: int = 0,
    limit: int = 48,
    find: str | None = None,
) -> tuple[int, list[File], int | None]:
    """The project's files for the Files tab, and how many there are.

    ``empty`` narrows by whether anything on the file's visible surface
    passes at ``floor``: ``"show_only"`` keeps the files where nothing
    does, ``"hide"`` the files where something does, ``"all"`` both. That
    is ``derive_observation_type(...) == BLANK`` computed live instead of
    read from the stored ``observation_type`` column, so the answer
    follows the grid's confidence slider rather than the project
    setting. Both sides use ``effective_floor``, so with the slider at
    rest the two agree exactly, and the two halves partition the
    project (``tests/api/test_labels_files.py``).

    Returns ``(total, page, find_index)`` — the uncapped count of
    matching files, the requested slice, and, when ``find`` names a file
    id, that file's 0-based position in the full ordering (``None`` when
    it does not match the filters). The position is what lets the
    Detections modal's "Open in files view" land the viewer on the file
    inside the list instead of showing it alone.

    ``sort='path'`` is the default because it groups one camera's photos
    together: ``file_path`` is absolute, so it begins with the
    deployment folder. Reviewing files means scanning the same scene
    over and over, and capture-time order interleaves cameras.
    """
    from app.api.crud.deployment import site_ids_filter

    passing = (
        select(Detection.id)
        .where(Detection.file_id == File.id)
        .where(on_visible_frame())
        .where(is_a_real_detection())
        .where(
            or_(
                Detection.confidence >= floor,
                Detection.verified == True,  # noqa: E712
            )
        )
    )
    query = (
        db.query(File)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
    )
    if empty == "show_only":
        query = query.filter(~passing.exists())
    elif empty == "hide":
        query = query.filter(passing.exists())

    # Box-level filters, the Detections tab's rules lifted to files: a
    # file shows when ONE box matches all of them together (species AND
    # confidence range), on the same `passing` scope as the empty
    # partition, so an invisible box can never match. One combined
    # EXISTS, not one per filter: a file with dog@0.9 and cat@0.3 must
    # not match "dogs below 40%" off two different boxes.
    #
    # `min_confidence` at or below the floor is a no-op: the Files
    # slider is clamped at the project threshold ("empty" is defined by
    # that one setting, never by a transient slider), so a lower value
    # can only arrive from a stale URL. Above the floor it is literal,
    # exactly as the sort worker applies it, so a verified
    # low-confidence box cannot satisfy a narrowed range there either.
    # Files without any box only appear while no box filter is active
    # (or through "Show only empty").
    box_filters = []
    if labels:
        box_filters.append(Detection.label_taxonomy_id.in_(labels))
    if min_confidence is not None and min_confidence > floor:
        box_filters.append(Detection.confidence >= min_confidence)
    if max_confidence is not None:
        box_filters.append(Detection.confidence <= max_confidence)
    # Classifier-score range. NULL label_confidence fails the comparison
    # in SQL, so unclassified boxes are excluded automatically once a
    # bound is set, the same behaviour as the Detections sort worker.
    if min_label_confidence is not None:
        box_filters.append(Detection.label_confidence >= min_label_confidence)
    if max_label_confidence is not None:
        box_filters.append(Detection.label_confidence <= max_label_confidence)
    if box_filters:
        query = query.filter(passing.where(*box_filters).exists())

    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.filter(site_clause)
    if date_from is not None:
        query = query.filter(File.captured_at_local >= date_from)
    if date_to is not None:
        # Include the whole end day, matching the event filters.
        query = query.filter(
            File.captured_at_local
            <= datetime.combine(date_to.date(), time.max)
        )
    if verification == "verified":
        query = query.filter(File.verified.is_(True))
    elif verification == "unverified":
        query = query.filter(File.verified.is_(False))
    # Triage marks live on the file itself, so unlike the events
    # endpoint no EXISTS is needed. Anything else (None, "all") means
    # no clause, matching `get_events`.
    if flagged in ("flagged", "not_flagged"):
        query = query.filter(File.flagged.is_(flagged == "flagged"))
    if favorited in ("favorited", "not_favorited"):
        query = query.filter(File.favorited.is_(favorited == "favorited"))

    total = query.count()

    # File.id breaks every tie, so paging is stable. Files with no
    # capture time sort last rather than wherever SQLite puts NULL.
    undated_last = File.captured_at_local.is_(None).asc()
    # Same-second ties resolve by file name, not by row id: a burst shot
    # within one second must come back in shooting order, and camera
    # filenames are sequential. Lexicographic, which equals human order
    # for the fixed-width names cameras write (IMG_0102); a true natural
    # sort would need a computed key and no real data has asked for it.
    name_break = [File.file_path.asc(), File.id.asc()]
    if sort == "path":
        order = [File.file_path.asc(), File.id.asc()]
    elif sort == "newest":
        order = [undated_last, File.captured_at_local.desc(), *name_break]
    elif sort == "oldest":
        order = [undated_last, File.captured_at_local.asc(), *name_break]
    elif sort == "random":
        if seed is None:
            raise ValueError("random sort requires a seed")
        order = [func.seeded_hash(File.id, seed).asc(), File.id.asc()]
    elif sort == "events":
        # Bursts contiguous: events newest first (as on the Detections
        # tab), inside an event the files in shooting order. Correlated
        # scalar subqueries rather than a join: a file in more than one
        # event would duplicate rows under a join, and LIMIT 1 keeps the
        # sort key single-valued. Files outside any event sort last.
        event_id_sq = (
            select(event_files.c.event_id)
            .where(event_files.c.file_id == File.id)
            .order_by(event_files.c.event_id)
            .limit(1)
            .correlate(File)
            .scalar_subquery()
        )
        event_start_sq = (
            select(Event.event_start_local)
            .join(event_files, event_files.c.event_id == Event.id)
            .where(event_files.c.file_id == File.id)
            .order_by(event_files.c.event_id)
            .limit(1)
            .correlate(File)
            .scalar_subquery()
        )
        order = [
            event_start_sq.is_(None).asc(),
            event_start_sq.desc(),
            event_id_sq.asc(),
            undated_last,
            File.captured_at_local.asc(),
            *name_break,
        ]
    else:
        raise ValueError(f"unknown sort: {sort}")

    find_index: int | None = None
    if find is not None:
        # The file's rank under the same filters and order the page
        # itself uses, so "position N" and "page N // limit" can never
        # disagree with what the grid shows. row_number over the
        # filtered set (thousands of rows, one window pass), not a
        # COUNT of rows sorting before it, which would mean restating
        # every order key as a comparison.
        ranked = query.with_entities(
            File.id.label("fid"),
            func.row_number().over(order_by=order).label("pos"),
        ).subquery()
        pos = (
            db.query(ranked.c.pos).filter(ranked.c.fid == find).scalar()
        )
        find_index = pos - 1 if pos is not None else None

    page = query.order_by(*order).offset(skip).limit(limit).all()
    return total, page, find_index


class LabelCounts(NamedTuple):
    """The two halves of the progress bar, kept apart as well as summed.

    A "crop label" is a detection above the threshold, one card in the
    Detections tab. An "empty label" is a file with nothing above it, one
    "nothing here" call. They never overlap, which is what lets one bar
    cover the page.

    ``files`` / ``files_verified`` count every file in scope and how many
    are signed off, for the Files tab's chip. That unit overlaps with the
    crop labels on purpose: the Files tab lists files with boxes too.
    """

    crop_labels: int
    crop_labels_verified: int
    empty_labels: int
    empty_labels_verified: int
    files: int
    files_verified: int

    @property
    def total(self) -> int:
        return self.crop_labels + self.empty_labels

    @property
    def verified(self) -> int:
        return self.crop_labels_verified + self.empty_labels_verified


def get_label_progress(
    db: Session,
    project_id: str,
    *,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    min_confidence: float | None = None,
) -> LabelCounts:
    """How many of the project's labels a human has checked.

    Counted in **labels**, where a label is one thing a person has to
    make a call on:

    - every detection that passes the project threshold, which is one
      card in the Crops tab, and
    - every file where nothing passes, which is one card in the Empties
      tab carrying the label "nothing here".

    The two never overlap and together they cover the project, because a
    file either has a passing detection or it does not. So this total is
    exactly the number of cards across both tabs, and 100% means every
    one of them has been looked at.

    That is what makes one bar work for a page with two halves, and what
    stops it reading 100% while the empty files are untouched. Counting
    files instead would have understated the crop work, since one file
    can carry several labels.

    The floor follows the grid's confidence slider, through the same
    ``effective_floor`` the two grids use, so this can never disagree
    with what is on screen beside it.

    It used to stay pinned to the project threshold, on the argument
    that a denominator moving while you drag would be unreadable. That
    was the wrong trade: the counts feed the chips on the tab switch,
    which sit next to grids that DO follow the slider, so at 1% the chip
    read "Empties 220" above a grid header saying "68 files". Two
    numbers for one thing on one screen is worse than a number that
    moves when you deliberately change what counts. The dashboard passes
    no slider, so its bar is unaffected either way.

    The two halves are returned separately as well as summed, so each
    tab can say how much is waiting in the other one. Site and date
    filters DO apply: they carry across the tab switch, so a count that
    ignored them would promise more than the user is about to see.
    """
    from app.api.crud.deployment import site_ids_filter

    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        return LabelCounts(0, 0, 0, 0)
    floor = effective_floor(project.counting_threshold, min_confidence)
    passes = and_(
        is_a_real_detection(),
        or_(
            Detection.confidence >= floor,
            Detection.verified == True,  # noqa: E712
        ),
    )

    def in_scope(query):
        query = query.join(
            Deployment, Deployment.id == File.deployment_id
        ).filter(Deployment.project_id == project_id)
        site_clause = site_ids_filter(site_ids)
        if site_clause is not None:
            query = query.filter(site_clause)
        if date_from is not None:
            query = query.filter(File.captured_at_local >= date_from)
        if date_to is not None:
            query = query.filter(
                File.captured_at_local
                <= datetime.combine(date_to.date(), time.max)
            )
        return query

    verified_sum = func.coalesce(
        func.sum(func.cast(Detection.verified, Integer)), 0
    )
    det_total, det_verified = in_scope(
        db.query(func.count(Detection.id), verified_sum)
        .select_from(Detection)
        .join(File, File.id == Detection.file_id)
        .filter(on_visible_frame())
        .filter(passes)
    ).one()

    has_passing = (
        select(Detection.id)
        .where(Detection.file_id == File.id)
        .where(on_visible_frame())
        .where(passes)
    )
    empty_total, empty_verified = in_scope(
        db.query(
            func.count(File.id),
            func.coalesce(func.sum(func.cast(File.verified, Integer)), 0),
        )
        .select_from(File)
        .filter(~has_passing.exists())
    ).one()

    files_total, files_verified = in_scope(
        db.query(
            func.count(File.id),
            func.coalesce(func.sum(func.cast(File.verified, Integer)), 0),
        ).select_from(File)
    ).one()

    return LabelCounts(
        crop_labels=int(det_total or 0),
        crop_labels_verified=int(det_verified or 0),
        empty_labels=int(empty_total or 0),
        empty_labels_verified=int(empty_verified or 0),
        files=int(files_total or 0),
        files_verified=int(files_verified or 0),
    )


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


def recompute_file_verified(db: Session, file_ids: Iterable[str]) -> None:
    """Maintain `File.verified` as the observation-level rollup.

    A file is verified when every reviewable detection on it is verified.
    "Reviewable" = confidence >= the project's detection threshold OR
    already verified (the standard threshold-or-verified rule).

    Files with no reviewable detections (empty / blank) are left
    untouched: there are no observations to roll up, so `File.verified`
    is owned directly by the file-verify action (a human reviewing the
    empty frame).

    Call this after any change to a detection's verified status. Every
    user-facing surface (file badge, file filter, "next unverified"
    navigation, the event MaxN rollup) reads `File.verified`, so keeping
    it in sync here is the single point where the upward cascade lives.
    No commit — the caller owns the transaction.

    It does flush, though, and that is load-bearing rather than tidiness.
    The app's session runs with ``autoflush=False`` (``db/base.py``), so
    a caller that has just set ``det.verified = True`` in Python still
    has it pending; the query below would read the old values from the
    database, decide "not all verified", and leave the flag stale until
    some later edit happened to fix it. Measured: a relabelled file
    exported ``is_verified = FALSE``, and relabelling the same detection
    a second time corrected it. Flushing here rather than at each call
    site means no caller can forget.
    """
    db.flush()
    ids = [fid for fid in dict.fromkeys(file_ids)]
    if not ids:
        return
    files = db.query(File).filter(File.id.in_(ids)).all()
    if not files:
        return

    # One threshold lookup per deployment, shared across its files.
    threshold_cache: dict[str, float] = {}
    now = datetime.now(UTC)
    for f in files:
        dep_id = f.deployment_id
        if dep_id not in threshold_cache:
            threshold_cache[dep_id] = _project_threshold_for_file(db, f)
        floor = threshold_cache[dep_id]

        rows = (
            db.query(Detection.verified)
            .filter(Detection.file_id == f.id)
            # A video is only its best frame, the same rule every other
            # surface applies. Without this a clip counted the detections
            # on its other sampled frames, which have no card in the grid
            # and no way to be verified, so any video with them could
            # never roll up to verified however much work the user did.
            # Measured on a real database: 24 of 26 videos were stuck.
            .filter(on_visible_frame_of(f))
            .filter(threshold_or_verified(floor))
            .all()
        )
        if not rows:
            # No reviewable detections: leave File.verified as-is (owned
            # by the file-verify action for empty frames).
            continue

        new_verified = all(r[0] for r in rows)
        if new_verified != f.verified:
            f.verified = new_verified
            f.verified_at_utc = now if new_verified else None


def recompute_file_verified_for_detections(
    db: Session, detection_ids: Iterable[str]
) -> None:
    """Recompute `File.verified` for the files owning the given detections."""
    ids = list(dict.fromkeys(detection_ids))
    if not ids:
        return
    file_ids = [
        fid
        for (fid,) in db.query(Detection.file_id)
        .filter(Detection.id.in_(ids))
        .distinct()
        .all()
    ]
    recompute_file_verified(db, file_ids)


def set_file_verified(db: Session, file: File, verified: bool) -> None:
    """Sign a file off, or take the sign-off back. No commit.

    Verifying a file says "the boxes you can see are all there is". So on
    the file's visible frame, every box the person could not see (below
    the threshold and not verified) is rejected the way the X key rejects
    a box (marked "false detection", verified), and every box they could
    see is verified. One rule for every file, empty or not, and for every
    box, whoever drew it: a drawn box has confidence 1.0, so it is always
    visible and never rejected.

    Rejecting the weak boxes rather than leaving them is what keeps
    "verified" true at every threshold: left alone, a 3% smudge came back
    the moment the confidence slider dropped, still under a verified
    flag. The rejected rows stay out of every user-facing scope through
    ``threshold_or_verified`` (`ml/label_exclusion.py`). They used to be
    deleted instead, which meant reprocess reported them as missing and
    nothing could ever bring them back; kept as rejected rows they match
    their ``results.json`` boxes again, and an unverify hands them back
    to the machine on the next reprocess.

    Scoped to the visible frame, which for a video is its best frame plus
    verified boxes on any frame. Boxes on frames nobody saw are neither
    rejected nor verified. A video with no best frame has no visible
    surface, so verifying it sets the flag and touches no boxes.

    Unverifying clears every box on the file, drawn ones included.

    Idempotent, so re-verifying picks up boxes added since. Both paths
    re-derive ``observation_type`` and the event MaxN, as the detection
    endpoints do. The caller commits.
    """
    threshold = _project_threshold_for_file(db, file)
    now = datetime.now(UTC)
    # Keep the file_id clause: the video branches of the predicate carry
    # only the frame clause.
    on_frame = and_(Detection.file_id == file.id, on_visible_frame_of(file))
    if verified:
        weak = (
            db.query(Detection)
            .filter(
                on_frame,
                Detection.verified == False,  # noqa: E712
                Detection.confidence < threshold,
            )
            .all()
        )
        mark_detections_false(db, weak)
        db.query(Detection).filter(
            on_frame,
            Detection.verified == False,  # noqa: E712
        ).update(
            {"verified": True, "verified_at_utc": now},
            synchronize_session=False,
        )
        file.verified = True
        file.verified_at_utc = now
    else:
        db.query(Detection).filter(Detection.file_id == file.id).update(
            {"verified": False, "verified_at_utc": None},
            synchronize_session=False,
        )
        file.verified = False
        file.verified_at_utc = None
    db.flush()
    _set_observation_type(db, file, threshold)
    event_ids = get_event_ids_for_files(db, [file.id])
    if event_ids:
        recalculate_max_n_for_events(db, event_ids, threshold)


def update_file(db: Session, file_id: str, update: FileUpdate) -> File | None:
    """Update a file's verification status, notes, favorited or flagged."""
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return None

    if update.verified is not None:
        set_file_verified(db, file, update.verified)

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


def recalculate_observation_type(db: Session, file_id: str) -> None:
    """
    Re-derive observation_type from the file's *passing* detections.

    Passing = over the project threshold OR verified (see
    ``derive_observation_type``), and on the file's visible surface (see
    ``on_visible_frame_of``), which for a video is its best frame. Called
    after any detection create / update / delete / verify so the summary
    stays consistent with what the verify grid shows.
    """
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return

    _set_observation_type(db, file, _project_threshold_for_file(db, file))
    db.commit()


def _set_observation_type(db: Session, file: File, threshold: float) -> None:
    """The derivation behind ``recalculate_observation_type``, no commit."""
    detections = (
        db.query(Detection)
        .filter(Detection.file_id == file.id)
        # Keep the file_id filter above: the video branches of this
        # predicate carry only the frame clause.
        .filter(on_visible_frame_of(file))
        .all()
    )
    file.observation_type = derive_observation_type(detections, threshold)


def recalculate_observation_types_for_project(
    db: Session, project_id: str
) -> int:
    """Re-derive observation_type for every file in a project.

    Run when the project detection threshold changes: the threshold feeds
    the passing rule, so a file whose only detection now falls below it
    flips to ``"blank"`` (and vice-versa). Returns the number of files
    whose observation_type actually changed. One pass, one commit.
    """
    project = db.get(Project, project_id)
    if project is None:
        return 0
    threshold = project.counting_threshold

    files = (
        db.query(File)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .all()
    )
    detections = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        # A video is only its best frame; File is joined, so the column
        # form of the rule applies directly.
        .filter(on_visible_frame())
        .all()
    )
    by_file: dict[str, list[Detection]] = {}
    for det in detections:
        by_file.setdefault(det.file_id, []).append(det)

    changed = 0
    for file in files:
        new_type = derive_observation_type(
            by_file.get(file.id, []), threshold
        )
        if file.observation_type != new_type:
            file.observation_type = new_type
            changed += 1
    db.commit()
    return changed
