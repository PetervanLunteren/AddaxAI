"""`frame_time`: the wall-clock time a video frame stands for.

Derived, never stored: file time plus frame over frame rate. It is what
the Counts page shows as time of MaxN and what the counts export writes,
so the rules for missing inputs matter more than the arithmetic.
"""

from datetime import datetime
from types import SimpleNamespace

from app.utils.media_dates import frame_time


def _file(captured=datetime(2026, 6, 1, 9, 0, 0), frame_rate=30.0):
    return SimpleNamespace(captured_at_local=captured, frame_rate=frame_rate)


def test_a_frame_is_offset_from_the_file_time_at_the_frame_rate():
    assert frame_time(_file(), 900) == datetime(2026, 6, 1, 9, 0, 30)
    assert frame_time(_file(frame_rate=29.97), 2997) == datetime(2026, 6, 1, 9, 1, 40)
    assert frame_time(_file(), 0) == datetime(2026, 6, 1, 9, 0, 0)


def test_an_image_is_its_own_frame():
    assert frame_time(_file(frame_rate=None), None) == datetime(2026, 6, 1, 9, 0, 0)


def test_no_capture_time_means_no_frame_time():
    assert frame_time(_file(captured=None), 900) is None
    assert frame_time(None, 900) is None


def test_an_unknown_frame_rate_falls_back_to_the_file_time():
    """Better a time that is a little early than none: the file's own
    time is what every other surface shows for that video."""
    assert frame_time(_file(frame_rate=None), 900) == datetime(2026, 6, 1, 9, 0, 0)
    assert frame_time(_file(frame_rate=0.0), 900) == datetime(2026, 6, 1, 9, 0, 0)
