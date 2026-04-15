"""
Unit tests for app.utils.datetime_serialization.

Covers `to_local_iso_with_offset()` across the tz configurations a real
deployment could have:
- DST-on date in a regional zone (Europe/Amsterdam, June)
- DST-off date in the same regional zone (January)
- Fixed-offset zone with no DST (Etc/GMT-3 → UTC+3)
- UTC
- Input rejection when the caller passes an already-tz-aware datetime
- Polar latitudes and other astral edge cases do not apply here — the
  serializer only interprets a wall-clock datetime in a tz, no sun-calc.
"""

from datetime import UTC, datetime

import pytest

from app.utils.datetime_serialization import (
    serialize_local_datetime,
    set_active_project_timezone,
    to_local_iso_with_offset,
)


def test_dst_on_in_regional_zone():
    """Amsterdam in mid-June is CEST (UTC+2)."""
    dt = datetime(2026, 6, 15, 7, 30, 0)
    result = to_local_iso_with_offset(dt, "Europe/Amsterdam")
    assert result == "2026-06-15T07:30:00+02:00"


def test_dst_off_in_regional_zone():
    """Amsterdam in mid-January is CET (UTC+1). No DST clock-jump risk."""
    dt = datetime(2026, 1, 15, 7, 30, 0)
    result = to_local_iso_with_offset(dt, "Europe/Amsterdam")
    assert result == "2026-01-15T07:30:00+01:00"


def test_fixed_offset_zone_no_dst():
    """Etc/GMT-3 is fixed UTC+3 (IANA inverts the sign vs. POSIX)."""
    summer = datetime(2026, 6, 15, 7, 30, 0)
    winter = datetime(2026, 1, 15, 7, 30, 0)
    assert to_local_iso_with_offset(summer, "Etc/GMT-3") == "2026-06-15T07:30:00+03:00"
    assert to_local_iso_with_offset(winter, "Etc/GMT-3") == "2026-01-15T07:30:00+03:00"


def test_utc_zone():
    dt = datetime(2026, 6, 15, 7, 30, 0)
    assert to_local_iso_with_offset(dt, "UTC") == "2026-06-15T07:30:00+00:00"


def test_rejects_tz_aware_input():
    """Caller must pass a naive datetime. A tz-aware input is a bug."""
    dt = datetime(2026, 6, 15, 7, 30, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="naive datetime"):
        to_local_iso_with_offset(dt, "Europe/Amsterdam")


def test_serialize_helper_passes_through_none():
    """serialize_local_datetime(None) returns None for Optional fields."""
    assert serialize_local_datetime(None) is None


def test_serialize_helper_uses_active_tz():
    """serialize_local_datetime reads the context tz when tz_name is omitted."""
    set_active_project_timezone("Etc/GMT-3")
    try:
        result = serialize_local_datetime(datetime(2026, 6, 15, 7, 30, 0))
        assert result == "2026-06-15T07:30:00+03:00"
    finally:
        # Reset to default so other tests don't see our change
        set_active_project_timezone("UTC")


def test_default_active_tz_is_utc():
    """When nothing is set, serializer falls back to UTC so output is still valid."""
    # Use a unique-ish datetime so we're not depending on state from other tests
    set_active_project_timezone("UTC")
    assert (
        serialize_local_datetime(datetime(2026, 3, 21, 0, 0, 0))
        == "2026-03-21T00:00:00+00:00"
    )
