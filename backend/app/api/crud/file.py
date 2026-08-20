"""
CRUD operations for files.
"""

from collections.abc import Iterable
from datetime import UTC, datetime, time
from typing import NamedTuple

from sqlalchemy import Integer, and_, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.api.schemas.file import FileUpdate
from app.core.confidence import effective_floor
from app.ml.detection_visibility import on_visible_frame, on_visible_frame_of
from app.ml.label_exclusion import is_a_real_detection
from app.ml.observation_type import derive_observation_type
from app.models import Deployment, Detection, File, Project


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
    near-noise tail down to 0.005. A broken lookup silently reclassified
    files and recomputed counts against the wrong floor instead of
    saying anything.
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


EMPTIES_SORT_VALUES = frozenset({"path", "newest", "oldest", "random"})


def get_empty_files(
    db: Session,
    project_id: str,
    *,
    floor: float,
    site_ids: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    verification: str | None = None,
    sort: str = "path",
    seed: int | None = None,
    skip: int = 0,
    limit: int = 48,
) -> tuple[int, list[File]]:
    """The project's empty photos at ``floor``, and how many there are.

    A file is empty when nothing on its visible surface passes: no
    detection at or above ``floor``, and none verified. That is
    ``derive_observation_type(...) == BLANK`` computed live instead of
    read from the stored ``observation_type`` column, so the answer
    follows the grid's confidence slider rather than the project
    setting. Both sides use ``effective_floor``, so with the slider at
    rest the two agree exactly.

    The consequence users are told about: every photo in the project is
    in exactly one of two places, the crop grid via a passing detection
    or this list. ``tests/api/test_empties.py`` pins that partition.

    Returns ``(total, page)`` — the uncapped count of matching files and
    the requested slice, so the toolbar can say how much is left to do.

    ``sort='path'`` is the default because it groups one camera's photos
    together: ``file_path`` is absolute, so it begins with the
    deployment folder. Reviewing empties means scanning the same scene
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
        .filter(~passing.exists())
    )

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

    total = query.count()

    # File.id breaks every tie, so paging is stable. Files with no
    # capture time sort last rather than wherever SQLite puts NULL.
    undated_last = File.captured_at_local.is_(None).asc()
    if sort == "path":
        order = [File.file_path.asc(), File.id.asc()]
    elif sort == "newest":
        order = [undated_last, File.captured_at_local.desc(), File.id.asc()]
    elif sort == "oldest":
        order = [undated_last, File.captured_at_local.asc(), File.id.asc()]
    elif sort == "random":
        if seed is None:
            raise ValueError("random sort requires a seed")
        order = [func.seeded_hash(File.id, seed).asc(), File.id.asc()]
    else:
        raise ValueError(f"unknown sort: {sort}")

    page = query.order_by(*order).offset(skip).limit(limit).all()
    return total, page


class LabelCounts(NamedTuple):
    """The two halves of the Labels page, kept apart as well as summed.

    A "crop label" is a detection above the threshold, one card in the
    Crops tab. An "empty label" is a file with nothing above it, one
    card in the Empties tab. They never overlap.
    """

    crop_labels: int
    crop_labels_verified: int
    empty_labels: int
    empty_labels_verified: int

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

    return LabelCounts(
        crop_labels=int(det_total or 0),
        crop_labels_verified=int(det_verified or 0),
        empty_labels=int(empty_total or 0),
        empty_labels_verified=int(empty_verified or 0),
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
            .filter(
                or_(Detection.confidence >= floor, Detection.verified == True)  # noqa: E712
            )
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


def _discard_detector_boxes(db: Session, file: File) -> None:
    """Remove the detector's boxes from a file a person has called empty.

    Verifying an empty frame is a statement about the picture: there is
    no animal in it. Every box the detector left behind is therefore a
    false positive, so they go.

    Keeping them made "empty" true only at one threshold. Drop the
    confidence slider and the file reappeared carrying a 3% smudge while
    still flagged verified, and a later change to the project threshold
    made it export as ``is_verified = TRUE`` beside a species nobody had
    confirmed.

    This is only defensible because the viewer draws no boxes over the
    frame: the person is judging the photograph, not a threshold. If
    weak boxes are ever drawn there again, revisit this, because then
    the verdict becomes threshold-dependent and deleting on it does not
    follow.

    Boxes the person drew are kept. That clause is insurance, not a live
    path: ``on_visible_frame_of`` passes verified detections on *any*
    frame, so a drawn box keeps its file reviewable even on a video where
    it sits off the best frame, and the caller never reaches here while
    one exists (pinned by
    ``test_a_file_holding_a_drawn_box_is_never_treated_as_empty``).

    The NULL case is spelled out on purpose: ``classification_method`` is
    nullable and a bare ``!= "human"`` matches no NULLs in SQL, which
    would leave every unclassified box behind, and that is most of them.

    Scoped to the file's visible surface, which for a video is its best
    frame. The person judged the one frame the app shows them, and a clip
    reads empty on that frame alone, so boxes on the other sampled frames
    were never on screen and are not theirs to discard. Without this the
    verdict on one frame threw away every box in the clip, including a
    confident animal on a frame nobody had seen.

    Nothing is truly lost. ``results.json`` on disk is never modified and
    still holds every box, which is what a re-analysis reads back.
    """
    db.query(Detection).filter(
        Detection.file_id == file.id,
        on_visible_frame_of(file),
        or_(
            Detection.classification_method.is_(None),
            Detection.classification_method != "human",
        ),
    ).delete(synchronize_session=False)


def update_file(db: Session, file_id: str, update: FileUpdate) -> File | None:
    """
    Update a file's verification status and/or notes.

    Sets verified_at_utc to current time when verified changes to True,
    clears it when verified changes to False.
    """
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return None

    if update.verified is not None:
        # The file-verify action (modal Enter). Sets File.verified and
        # cascades down to the detections. recompute keeps File.verified
        # consistent for files that have detections; for empty frames it
        # is a no-op, so the flag we set here stands.
        if update.verified and not file.verified:
            now = datetime.now(UTC)
            file.verified = True
            file.verified_at_utc = now
            threshold = _project_threshold_for_file(db, file)
            # Is there anything here a person could have been judging?
            # The same "reviewable" rule the rollup uses, visible frame
            # and all, so the two can never disagree about what counts
            # as empty.
            reviewable = (
                db.query(Detection.id)
                .filter(Detection.file_id == file_id)
                .filter(on_visible_frame_of(file))
                .filter(
                    or_(
                        Detection.confidence >= threshold,
                        Detection.verified == True,  # noqa: E712
                    )
                )
                .first()
            )
            if reviewable is None:
                # An empty frame. The verdict is "nothing here", so the
                # detector's leftovers go with it.
                _discard_detector_boxes(db, file)
            else:
                # Only verify detections above the project's detection
                # threshold (below-threshold ones are not visible).
                db.query(Detection).filter(
                    Detection.file_id == file_id,
                    Detection.verified == False,  # noqa: E712
                    Detection.confidence >= threshold,
                ).update({"verified": True, "verified_at_utc": now})
                recompute_file_verified(db, [file_id])
        elif not update.verified and file.verified:
            file.verified = False
            file.verified_at_utc = None
            db.query(Detection).filter(
                Detection.file_id == file_id,
            ).update({"verified": False, "verified_at_utc": None})
            recompute_file_verified(db, [file_id])

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

    detections = (
        db.query(Detection)
        .filter(Detection.file_id == file_id)
        # Keep the file_id filter above: the video branches of this
        # predicate carry only the frame clause.
        .filter(on_visible_frame_of(file))
        .all()
    )
    threshold = _project_threshold_for_file(db, file)
    file.observation_type = derive_observation_type(detections, threshold)
    db.commit()


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
