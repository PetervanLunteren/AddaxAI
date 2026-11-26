"""
Datetime serialization helpers.

Observational datetimes (File.captured_at_local, Sequence.sequence_start_local,
etc.) are stored as naive wall-clock time in the project's local camera
timezone. When sending them over the wire, we attach the UTC offset that
applies on that specific calendar date in the project timezone so that
DST is resolved per-file. This makes the wire format self-describing
(browsers parse it natively with new Date(...)) and matches Camtrap-DP's
ISO 8601 + offset requirement.

See the "Datetime conventions" section in DEVELOPERS.md for the full rule.
"""

from contextvars import ContextVar
from datetime import datetime
from zoneinfo import ZoneInfo


class _ActiveTimezone:
    """The project timezone of one request, filled in by the endpoint."""

    def __init__(self) -> None:
        self.name = "UTC"


# Per-request active project timezone used by Pydantic field serializers
# when rendering observational datetimes. Endpoints set it with
# `set_active_project_timezone(tz)` once they know the project. Reads
# "UTC" when nothing set it, so a forgotten call gives a predictable,
# valid ISO string rather than a crash.
#
# The ContextVar holds an object rather than the name itself. FastAPI
# runs a plain `def` endpoint in a worker thread with a copy of the
# request's context, and serializes the response on the event loop. A
# name set in the thread stays in the copy; a field written on the shared
# object is seen by the serializer. `start_request_timezone` gives every
# request its own object, so nothing carries over between requests.
_active_project_timezone: ContextVar[_ActiveTimezone | None] = ContextVar(
    "active_project_timezone", default=None
)


def start_request_timezone() -> None:
    """Give the current request its own timezone, "UTC" until set.

    Called by a middleware before every request, on the event loop, so
    the endpoint's thread and the serializer share the object.
    """
    _active_project_timezone.set(_ActiveTimezone())


def set_active_project_timezone(tz_name: str) -> None:
    """Set the active project timezone for the current request.

    Outside a request (a worker, a test) there is no object yet, and this
    makes one for the current context.
    """
    active = _active_project_timezone.get()
    if active is None:
        active = _ActiveTimezone()
        _active_project_timezone.set(active)
    active.name = tz_name


def get_active_project_timezone() -> str:
    """Return the active project timezone for the current request."""
    active = _active_project_timezone.get()
    return active.name if active is not None else "UTC"


def to_local_iso_with_offset(dt: datetime, tz_name: str | None = None) -> str:
    """
    Serialize a naive wall-clock datetime as ISO 8601 with the UTC offset
    that applies to `tz_name` on `dt`'s calendar date.

    Example: `to_local_iso_with_offset(datetime(2026, 6, 15, 7, 30), "Europe/Amsterdam")`
    returns `"2026-06-15T07:30:00+02:00"` (CEST, DST on), whereas
    `to_local_iso_with_offset(datetime(2026, 1, 15, 7, 30), "Europe/Amsterdam")`
    returns `"2026-01-15T07:30:00+01:00"` (CET, DST off).

    Args:
        dt: Naive datetime (no tzinfo) representing wall-clock time at
            the camera. Raises ValueError if dt is already tz-aware.
        tz_name: IANA timezone name (e.g. "Europe/Amsterdam") or a fixed
            offset zone (e.g. "Etc/GMT-3", "UTC"). Defaults to the
            active project timezone from the request context.

    Returns:
        ISO 8601 string with seconds precision and a `±hh:mm` offset.
    """
    if dt.tzinfo is not None:
        raise ValueError(
            f"to_local_iso_with_offset expects a naive datetime, got tzinfo={dt.tzinfo!r}"
        )
    tz = tz_name if tz_name is not None else get_active_project_timezone()
    aware = dt.replace(tzinfo=ZoneInfo(tz))
    return aware.isoformat(timespec="seconds")


def serialize_local_datetime(dt: datetime | None) -> str | None:
    """
    Pydantic field_serializer helper for naive local datetimes.

    Reads the active project timezone from the request context. Returns
    None unchanged so that Optional fields stay Optional.
    """
    if dt is None:
        return None
    return to_local_iso_with_offset(dt)
