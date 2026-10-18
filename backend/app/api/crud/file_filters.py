"""The file-level filter predicates, defined once.

The Labels page (``crud/file.py``) and the media export
(``ml/postprocessing_outputs/_scope.py``) both narrow files by capture
date and by the three per-file marks. The clauses live here, a leaf
module with no imports beyond SQLAlchemy and the models, so both sides
read the same rule and neither has to import the other.

The one rule worth stating: a date range is inclusive at both ends, on
whole days. ``date_to`` covers the entire end day. A file with no
capture time (``captured_at_local IS NULL``) fails both comparisons,
so it drops out as soon as either date is set, and stays in when
neither is.
"""

from __future__ import annotations

from datetime import date, datetime, time

from app.models import File


def _day_start(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time.min)


def _day_end(value: date | datetime) -> datetime:
    day = value.date() if isinstance(value, datetime) else value
    return datetime.combine(day, time.max)


def file_flag_clauses(
    *,
    date_from: date | datetime | None = None,
    date_to: date | datetime | None = None,
    verified: bool | None = None,
    flagged: bool | None = None,
    favorited: bool | None = None,
) -> list:
    """WHERE clauses on ``File`` for the given filters; ``None`` means no
    clause for that filter.

    ``date_from`` as a datetime is used as is (the Labels page passes
    the parsed query value); as a date it means the start of that day.
    ``date_to`` always means the end of its day.
    """
    clauses: list = []
    if date_from is not None:
        clauses.append(File.captured_at_local >= _day_start(date_from))
    if date_to is not None:
        clauses.append(File.captured_at_local <= _day_end(date_to))
    if verified is not None:
        clauses.append(File.verified.is_(verified))
    if flagged is not None:
        clauses.append(File.flagged.is_(flagged))
    if favorited is not None:
        clauses.append(File.favorited.is_(favorited))
    return clauses


def tri_state(value: str | None, true_word: str, false_word: str) -> bool | None:
    """Map the Labels page's three-valued query strings onto a bool or
    ``None``: ``"verified"`` / ``"unverified"``, ``"flagged"`` /
    ``"not_flagged"``, ``"favorited"`` / ``"not_favorited"``. Anything
    else (``None``, ``"all"``) means no filter."""
    if value == true_word:
        return True
    if value == false_word:
        return False
    return None
