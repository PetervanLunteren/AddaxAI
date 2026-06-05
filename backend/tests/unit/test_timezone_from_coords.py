"""Tests for deriving an IANA timezone from coordinates.

Uses timezonefinder's lightweight TimezoneFinderL, which can return a
same-offset neighbour zone (a Dutch point resolves to Europe/Berlin, not
Europe/Amsterdam). The tests assert on the resolved zone's behaviour, not
a brittle exact label, except for points where the lite label is stable.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from app.utils.timezone_from_coords import tz_from_coords


def test_serengeti_resolves_to_east_africa():
    assert tz_from_coords(-2.33, 34.83) == "Africa/Dar_es_Salaam"


def test_yellowstone_resolves_to_mountain_time():
    assert tz_from_coords(44.6, -110.5) == "America/Denver"


def test_netherlands_resolves_to_a_zone_with_the_right_offset():
    # TimezoneFinderL labels a Dutch coordinate as Europe/Berlin; that zone
    # shares the Netherlands' offset and DST rules, which is all the sun
    # math needs. Assert offset-equivalence rather than the exact label.
    tz = tz_from_coords(52.37, 4.90)
    assert tz is not None
    summer = datetime(2026, 7, 1, 12, 0)
    winter = datetime(2026, 1, 1, 12, 0)
    nl = ZoneInfo("Europe/Amsterdam")
    got = ZoneInfo(tz)
    assert got.utcoffset(summer) == nl.utcoffset(summer)
    assert got.utcoffset(winter) == nl.utcoffset(winter)
