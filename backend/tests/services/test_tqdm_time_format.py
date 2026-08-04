"""The video frame selection phase times itself, so it has to speak tqdm.

Every other analysis phase is driven by a subprocess and passes tqdm's own
elapsed / remaining strings straight through to the frontend. Video frame
selection is a plain Python loop, so it does that arithmetic itself, and
the frontend's `tqdmTimeToSeconds` (frontend/src/lib/duration.ts) accepts
exactly two shapes: ``M:SS`` and ``H:MM:SS``. Anything else falls through to
being rendered as raw text, so the row would read "Remaining time: 92.4"
instead of "about 2 min".

These pin the format against that parser, which is the contract nothing
else checks.
"""

from app.workers.detection_worker import _tqdm_time


def test_under_a_minute_is_zero_minutes():
    assert _tqdm_time(0) == "0:00"
    assert _tqdm_time(7) == "0:07"
    assert _tqdm_time(59) == "0:59"


def test_minutes_pad_seconds_to_two_digits():
    assert _tqdm_time(60) == "1:00"
    assert _tqdm_time(65) == "1:05"
    assert _tqdm_time(600) == "10:00"
    assert _tqdm_time(3599) == "59:59"


def test_an_hour_switches_to_three_parts():
    """`tqdmTimeToSeconds` reads 3 parts as H:MM:SS, so minutes pad too."""
    assert _tqdm_time(3600) == "1:00:00"
    assert _tqdm_time(3661) == "1:01:01"
    assert _tqdm_time(7325) == "2:02:05"


def test_seconds_are_floored_not_rounded():
    """Never overstate elapsed time; the frontend humanises from here."""
    assert _tqdm_time(59.9) == "0:59"
    assert _tqdm_time(0.9) == "0:00"


def test_negative_is_clamped():
    """A remaining estimate can go slightly negative on the last item."""
    assert _tqdm_time(-5) == "0:00"


def test_every_output_has_the_shape_the_frontend_parses():
    """Two or three colon-separated integer parts, nothing else."""
    for seconds in (0, 1, 59, 60, 3599, 3600, 86399, 90061):
        parts = _tqdm_time(seconds).split(":")
        assert len(parts) in (2, 3), f"{seconds}s -> {_tqdm_time(seconds)}"
        assert all(p.isdigit() for p in parts)
        # Everything after the first part is zero-padded to two digits,
        # which is what makes 1:5 impossible to misread as 1 min 5 sec.
        assert all(len(p) == 2 for p in parts[1:])
