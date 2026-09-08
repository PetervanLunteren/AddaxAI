"""Which detections the user can actually see.

Images have no frames, so every image detection is visible. A video's
detections sit on every sampled frame and every one of them belongs to
a track (the tracker's boxes, a drawn box as a one-frame track, the
boxes of a video analysed before tracking became the standard, migrated
into one-frame tracks). The track is the unit of review, and it has one
card:

    a video detection is visible on its track's representative frame

That frame is the one the track's highest-confidence box sits on
(`Track.representative_frame_number`), the box the tracking script kept
a crop of. It is the track's only card: a sibling box on any other
frame, the video's cover frame included, is the same animal, so it must
not become a second card. A verdict on the card reaches every box of
the track (`expand_to_tracks`), which is why no "verified anywhere"
escape hatch exists: letting verified boxes through on other frames
would turn every frame of a verified track into a row of its own, in
the grid counts and the detection exports. An untracked video box (an
unverified box off the cover of a pre-tracking run) is visible nowhere,
like the sub-threshold noise, and kept only so a reprocess of that run
still matches its JSON.

**Two things must apply this: anything that counts detections for the
user, and anything that decides what the media outputs contain.**

Counting without it promises rows the UI cannot show. The label filter
said "person 62" over a grid holding 4, and offered a "chimpanzee (2)"
branch that led to a blank screen, because both of those detections live
on a frame with no card.

Placing without it is the same bug wearing different clothes. The still
beside a copied clip is its cover frame, so deciding its folder from a
box that is not a card files a picture under a label nobody reviewed.

**The spreadsheet exports apply it; the archival ones do not.**
`addaxai-detections.csv` and the XLSX detections sheet hold what the
Labels grid holds, one row per card, so they agree with
`addaxai-files.csv` and `counts.csv` beside them. The complete record is
`addaxai-recognitions.json`, which keeps every stored box with its frame
and track. The CamTrap DP export writes one observation per track, with
the track's span and its representative box.

**Two lanes, one rule.** Have a query? Use a predicate. Have the
detections already in memory? Use `visible_detections`. A parity test
(`tests/ml/test_detection_visibility.py`) pins that the two agree, which
is what makes having two of them safe.

Places that cannot use any of these and keep a hand-written copy:
`calculate_max_n_for_event` filters after the query to keep its grouping,
`similarity_script` is a subprocess with no `app.*` on its path, and
`shouldDrawBbox` in the frontend is TypeScript (it draws the boxes of
the frame on screen, which for a card is the representative frame).
Keep them in step.
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
    False for an untracked box."""
    return exists().where(
        and_(
            Track.id == Detection.track_id,
            Track.representative_frame_number == Detection.frame_number,
        )
    )


def on_visible_frame() -> ColumnElement[bool]:
    """Predicate for a query that has ``File`` joined to ``Detection``.

    Combine with the usual threshold-or-verified clause; this one is
    only about *frames*, not confidence.
    """
    return or_(File.file_type != "video", on_representative_frame())


def on_visible_frame_of(file: File) -> ColumnElement[bool]:
    """Same rule for a query already scoped to one known ``File``.

    Used where the caller holds the ORM object and does not join
    ``File``. The video branch returns **only** the frame clause. A
    caller that is not otherwise scoped to this file must keep its own
    ``Detection.file_id == file.id`` filter; without it a video's
    detections would be drawn from every file.
    """
    if file.file_type != "video":
        return Detection.file_id == file.id
    return on_representative_frame()


class _Track(Protocol):
    representative_frame_number: int


class _FramedDetection(Protocol):
    frame_number: int | None
    track: _Track | None


_D = TypeVar("_D", bound=_FramedDetection)


def visible_detections(file: File, detections: Iterable[_D]) -> list[_D]:
    """The same rule again, for detections already in memory.

    The Python twin of ``on_visible_frame_of``: use this where the caller
    holds a list rather than a query it can filter. Input order is
    preserved, because ``strongest_passing_detection`` makes a stable
    order the caller's contract.

    Takes the ``File`` rather than an ``is_video`` flag on purpose: the
    ``file_type == "video"`` test is half of the rule, and passing it in
    would hand-copy that half to every call site, which is the
    duplication this module exists to stop.
    """
    if file.file_type != "video":
        return list(detections)
    return [
        det
        for det in detections
        if det.track is not None
        and det.frame_number == det.track.representative_frame_number
    ]
