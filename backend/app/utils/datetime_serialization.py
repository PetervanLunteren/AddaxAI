"""
Datetime serialization helpers.

Observational datetimes (File.captured_at_local, Event.event_start_local,
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

# Per-request active project timezone used by Pydantic field serializers
# when rendering observational datetimes. Routers set this at the top of
# the endpoint via `set_active_project_timezone(tz)` after they know
# which project the response belongs to. Defaults to "UTC" so any code
# path that forgets to set it produces a predictable, valid ISO string
# rather than a crash.
_active_project_timezone: ContextVar[str] = ContextVar(
    "active_project_timezone", default="UTC"
)


def set_active_project_timezone(tz_name: str) -> None:
    """Set the active project timezone for the current request context."""
    _active_project_timezone.set(tz_name)


def get_active_project_timezone() -> str:
    """Return the active project timezone for the current request context."""
    return _active_project_timezone.get()


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
    tz = tz_name if tz_name is not None else _active_project_timezone.get()
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
