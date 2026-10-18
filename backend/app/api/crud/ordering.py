"""
The one order in which files, and the boxes on a file, are listed.

Shared by the Labels Files tab, the /api/files lists, every export table
and the recognition JSON, so the same photos never come out in a
different order on a different surface.
"""

from sqlalchemy import ColumnElement

from app.models import Detection, File


def file_time_order(*, newest_first: bool) -> list[ColumnElement]:
    """Files by capture time.

    Files with no capture time come last, wherever SQLite would put NULL.
    Same-second ties resolve by file path, not by row id: a burst shot
    within one second must come back in shooting order, and camera file
    names are sequential. Lexicographic, which equals human order for the
    fixed-width names cameras write (RCNX0001); a true natural sort would
    need a computed key and no real data has asked for it. The id last
    makes the order total, so paging is stable.
    """
    captured = (
        File.captured_at_local.desc()
        if newest_first
        else File.captured_at_local.asc()
    )
    return [
        File.captured_at_local.is_(None).asc(),
        captured,
        File.file_path.asc(),
        File.id.asc(),
    ]


def detection_order_in_file() -> list[ColumnElement]:
    """Boxes on one file: frame order first, so a video's boxes read
    sequentially (NULL for images, so they tie), then strongest first,
    the order MegaDetector's own output uses. The id makes it total."""
    return [
        Detection.frame_number.asc(),
        Detection.confidence.desc(),
        Detection.id.asc(),
    ]
