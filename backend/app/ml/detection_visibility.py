"""Which detections the user can actually see.

A video's detections sit on every sampled frame, but only the best frame
is written to disk as a JPEG. So the app shows exactly one frame per
video, and a detection on any other frame has no picture anywhere: not in
the Labels grid, not on an event card, not in the annotated still a
folder run writes. Images have no frames, so every image detection is
visible.

    a video detection is visible only when
    Detection.frame_number == File.best_frame_number

Verified detections are the exception and pass on any frame. A human
decision must never end up out of reach, which is the same escape hatch
`calculate_max_n_for_event` uses to let a species verified on some frame
into the counts.

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

**This does NOT apply to the data exports.** `addaxai-detections.csv`,
the XLSX and the recognition JSON carry every detection on every frame by
design: they are the complete record of the run, so the user can do their
own filtering downstream in their own tools.

Three places cannot use these helpers and keep a hand-written copy:
`calculate_max_n_for_event` filters after the query to keep its grouping,
`similarity_script` is a subprocess with no `app.*` on its path, and
`shouldDrawBbox` in the frontend is TypeScript. Keep them in step.
"""

from __future__ import annotations

from sqlalchemy import or_
from sqlalchemy.sql.elements import ColumnElement

from app.models import Detection, File


def on_visible_frame() -> ColumnElement[bool]:
    """Predicate for a query that has ``File`` joined to ``Detection``.

    Combine with the usual threshold-or-verified clause; this one is
    only about *frames*, not confidence.
    """
    return or_(
        File.file_type != "video",
        Detection.frame_number == File.best_frame_number,
        Detection.verified == True,  # noqa: E712
    )


def on_visible_frame_of(file: File) -> ColumnElement[bool]:
    """Same rule for a query already scoped to one known ``File``.

    Used where the caller holds the ORM object and does not join
    ``File``, so the frame number is a plain value rather than a column.
    A video with no best frame has no visible surface at all, and the
    literal ``False`` says so rather than silently matching everything.
    """
    if file.file_type != "video":
        return Detection.file_id == file.id
    if file.best_frame_number is None:
        return Detection.verified == True  # noqa: E712
    return or_(
        Detection.frame_number == file.best_frame_number,
        Detection.verified == True,  # noqa: E712
    )
