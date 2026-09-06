"""Which detections the user can actually see.

A video's detections sit on every sampled frame, but only the best frame
is written to disk as a JPEG. So the app shows exactly one frame per
video, and a detection on any other frame has no picture anywhere: not in
the Labels grid, not on an event card, not in the annotated still a
folder run writes. Images have no frames, so every image detection is
visible.

    an untracked video detection is visible only when
    Detection.frame_number == File.best_frame_number,
    a tracked one only on its track's representative frame

A tracked video has one more still per track, at the frame of the
track's highest-confidence box (`Track.representative_frame_number`,
written by the same pass as the best frame), so that box has a picture
too and is the card the person reviews the whole track through. It is
the track's only card: a sibling box that happens to sit on the file's
best frame is the same animal, so it must not become a second card.

Verified untracked detections are the exception and pass on any frame.
A human decision must never end up out of reach, which is the same
escape hatch `calculate_max_n_for_event` uses to let a species verified
on some frame into the counts. A tracked box needs no such hatch: a
verdict on one box reaches the whole track (`expand_to_tracks`), so the
card on the representative frame always carries it. Letting verified
tracked boxes through would turn every frame of a verified track into a
row of its own, in the grid counts and the detection exports.

**Two things must apply this: anything that counts detections for the
user, and anything that decides what the media outputs contain.**

Counting without it promises rows the UI cannot show. The label filter
said "person 62" over a grid holding 4, and offered a "chimpanzee (2)"
branch that led to a blank screen, because both of those detections live
on a frame nobody can open.

Placing without it is the same bug wearing different clothes. A video is
written to disk as its best-frame JPEG, so deciding its folder, or
whether to drop it, from a detection on some other frame files a picture
under a label that picture does not show.

**The spreadsheet exports apply it; the archival ones do not.**
`addaxai-detections.csv` and the XLSX detections sheet used to carry
every box on every frame, on the argument that they are the complete
record and the user filters them downstream. In practice the file gave
them nothing to filter *on*: it carries `frame_number` but not the
video's best frame, so the comparison cannot be made from the file at
all. What users saw was a species list holding animals they could not
find, select or relabel anywhere in the app. So these tables now hold
what the Labels grid holds, in both projects mode and folder runs, which
also makes them agree with `addaxai-files.csv` and `counts.csv` beside
them.

The complete record is `addaxai-recognitions.json`, which keeps every
stored detection with its frame number, and the CamTrap DP export, whose
per-box rows must not drop a video whose best frame happens to be empty.
Both are read by other software rather than by a person, which is why
they keep the boxes with no picture.

**Two lanes, one rule.** Have a query? Use a predicate. Have the
detections already in memory? Use `visible_detections`. A parity test
(`tests/ml/test_detection_visibility.py`) pins that the two agree, which
is what makes having two of them safe.

Places that cannot use any of these and keep a hand-written copy:
`calculate_max_n_for_event` filters after the query to keep its grouping,
`similarity_script` is a subprocess with no `app.*` on its path, and
`shouldDrawBbox` in the frontend is TypeScript. There are more (`labels.py`
twice, `embedding_utils`, `crop_service`, `annotated_copies` twice), which
is itself the argument for reaching for a helper here rather than writing
a tenth. Keep them in step.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

from sqlalchemy import and_, exists, or_
from sqlalchemy.sql.elements import ColumnElement

from app.models import Detection, File, Track


def on_representative_frame() -> ColumnElement[bool]:
    """The box that stands for its track: the one on the track's
    representative frame. A correlated EXISTS, so it drops into any query
    that has ``Detection`` without asking the caller to join ``tracks``.
    False for every untracked box."""
    return exists().where(
        and_(
            Track.id == Detection.track_id,
            Track.representative_frame_number == Detection.frame_number,
        )
    )


def _untracked() -> ColumnElement[bool]:
    return Detection.track_id.is_(None)


def on_pixel_surface() -> ColumnElement[bool]:
    """The frames that have a JPEG and stand for the box: an image, an
    untracked box on the video's best frame, or a track's representative
    frame. Verified boxes elsewhere are visible but have no picture,
    which is why the embedding paths use this and not
    ``on_visible_frame``. Needs ``File`` joined to ``Detection``."""
    return or_(
        File.file_type == "image",
        and_(
            File.file_type == "video",
            _untracked(),
            Detection.frame_number == File.best_frame_number,
        ),
        on_representative_frame(),
    )


def on_visible_frame() -> ColumnElement[bool]:
    """Predicate for a query that has ``File`` joined to ``Detection``.

    Combine with the usual threshold-or-verified clause; this one is
    only about *frames*, not confidence.
    """
    return or_(
        File.file_type != "video",
        and_(
            _untracked(),
            or_(
                Detection.frame_number == File.best_frame_number,
                Detection.verified == True,  # noqa: E712
            ),
        ),
        on_representative_frame(),
    )


def on_visible_frame_of(file: File) -> ColumnElement[bool]:
    """Same rule for a query already scoped to one known ``File``.

    Used where the caller holds the ORM object and does not join
    ``File``, so the frame number is a plain value rather than a column.
    A video with no best frame has no visible surface beyond whatever a
    human verified, so only verified detections pass.

    The video branches return **only** the frame clause. A caller that is
    not otherwise scoped to this file must keep its own
    ``Detection.file_id == file.id`` filter; without it a video's
    detections would be drawn from every file sharing that frame number.
    """
    if file.file_type != "video":
        return Detection.file_id == file.id
    if file.best_frame_number is None:
        return or_(
            and_(_untracked(), Detection.verified == True),  # noqa: E712
            on_representative_frame(),
        )
    return or_(
        and_(
            _untracked(),
            or_(
                Detection.frame_number == file.best_frame_number,
                Detection.verified == True,  # noqa: E712
            ),
        ),
        on_representative_frame(),
    )


class _Track(Protocol):
    representative_frame_number: int


class _FramedDetection(Protocol):
    frame_number: int | None
    verified: bool
    track: _Track | None


_D = TypeVar("_D", bound=_FramedDetection)


def visible_detections(file: File, detections: Iterable[_D]) -> list[_D]:
    """The same rule again, for detections already in memory.

    The Python twin of ``on_visible_frame_of``: use this where the caller
    holds a list rather than a query it can filter. Input order is
    preserved, because ``strongest_passing_detection`` makes a stable
    order the caller's contract.

    Takes the ``File`` rather than ``is_video`` / ``best_frame_number``
    keywords on purpose. The ``file_type == "video"`` test is a third of
    the rule, and passing it in as a flag would hand-copy that third to
    every call site, which is the duplication this module exists to stop.
    """
    if file.file_type != "video":
        return list(detections)
    best = file.best_frame_number
    return [
        det
        for det in detections
        if (
            det.frame_number == det.track.representative_frame_number
            if det.track is not None
            else det.verified or (best is not None and det.frame_number == best)
        )
    ]
