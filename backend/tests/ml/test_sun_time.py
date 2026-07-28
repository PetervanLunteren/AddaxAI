"""Tests for app.ml.sun_time.

Covers the Vazquez 2019 double-anchored sun-time transform:
- per_date_sun_phases: smoke test against astral for a known location.
- compute_anchors / compute_anchor_bands: aggregate means ignoring polar
  entries.
- transform_to_sun_time: identity, day-stretch, midnight-wraparound,
  polar drop.

All tests are pure Python: no DB, no FastAPI.
"""

from datetime import date

import pytest

from app.ml.sun_time import (
    compute_anchor_bands,
    compute_anchors,
    per_date_sun_phases,
    transform_to_sun_time,
)

# ---------------------------------------------------------------------------
# per_date_sun_phases
# ---------------------------------------------------------------------------


def test_per_date_sun_phases_nairobi_smoke():
    """Equator in late March: sunrise near 06:30, sunset near 18:40 local."""
    result = per_date_sun_phases(
        [date(2024, 3, 20)],
        lat=-1.286389,   # Nairobi
        lon=36.817223,
        tz_name="Africa/Nairobi",
    )
    assert date(2024, 3, 20) in result
    phases = result[date(2024, 3, 20)]
    assert phases is not None
    dawn, sunrise, sunset, dusk = phases
    # Order is dawn < sunrise < sunset < dusk.
    assert dawn < sunrise < sunset < dusk
    # Equinox + equator => day length near 12 hours.
    assert 11.5 < (sunset - sunrise) < 12.5
    # Sunrise around 06:30 local.
    assert 5.5 < sunrise < 7.0


def test_per_date_sun_phases_deduplicates():
    """Duplicate dates in the input should produce one astral call."""
    d = date(2024, 6, 15)
    result = per_date_sun_phases(
        [d, d, d],
        lat=52.37,
        lon=4.90,
        tz_name="Europe/Amsterdam",
    )
    assert len(result) == 1
    assert result[d] is not None


def test_per_date_sun_phases_polar_returns_none():
    """Mid-winter at 85N: astral refuses; function returns None for the date."""
    d = date(2024, 12, 21)
    result = per_date_sun_phases(
        [d],
        lat=85.0,
        lon=0.0,
        tz_name="UTC",
    )
    assert result[d] is None


def test_per_date_sun_phases_unwraps_dusk_past_midnight():
    """Near summer solstice at ~60N, astral reports dusk just past
    midnight on the same calendar date. The helper must keep phases
    monotonic so downstream averages stay sane; it does that by adding
    24 h to dusk when it lands before sunset.
    """
    d = date(2024, 6, 21)
    result = per_date_sun_phases(
        [d], lat=59.91, lon=10.75, tz_name="Europe/Oslo"
    )
    phases = result[d]
    assert phases is not None
    dawn, sunrise, sunset, dusk = phases
    # Order is monotonic after unwrap.
    assert dawn < sunrise < sunset < dusk
    # Dusk should be pushed past 24 h since astral returned ~00:30.
    assert dusk > 24.0


def test_compute_anchor_bands_wraps_mean_back_into_24h():
    """When per-date phases extend past 24 h (for solstice unwrap), the
    mean anchor values must still land in [0, 24) so the chart can plot
    them on a 0..24 axis."""
    phases = {
        date(2024, 6, 21): (2.0, 3.9, 22.7, 24.5),   # summer, unwrapped dusk
        date(2024, 11, 15): (7.5, 8.25, 15.8, 16.5), # winter
    }
    bands = compute_anchor_bands(phases)
    assert bands is not None
    dawn, sunrise, sunset, dusk = bands
    assert 0 <= dawn < 24
    assert 0 <= sunrise < 24
    assert 0 <= sunset < 24
    assert 0 <= dusk < 24
    # Mean dusk = (24.5 + 16.5) / 2 = 20.5. No wrap needed since it
    # stays under 24.
    assert dusk == 20.5


def test_per_date_sun_phases_dst_differs():
    """CEST summer vs CET winter have different sunrise hours for
    Amsterdam.  The function is supposed to respect the project
    timezone so DST is baked in automatically.
    """
    summer_d = date(2024, 6, 21)
    winter_d = date(2024, 12, 21)
    res = per_date_sun_phases(
        [summer_d, winter_d],
        lat=52.37,
        lon=4.90,
        tz_name="Europe/Amsterdam",
    )
    summer_sunrise = res[summer_d][1]
    winter_sunrise = res[winter_d][1]
    # Winter sunrise in wall-clock Amsterdam (CET) is much later than
    # summer sunrise in wall-clock Amsterdam (CEST).
    assert winter_sunrise - summer_sunrise > 2.5


# ---------------------------------------------------------------------------
# compute_anchors / compute_anchor_bands
# ---------------------------------------------------------------------------


def test_compute_anchors_empty_returns_none():
    assert compute_anchors({}) is None


def test_compute_anchors_all_polar_returns_none():
    phases = {
        date(2024, 12, 21): None,
        date(2024, 12, 22): None,
    }
    assert compute_anchors(phases) is None


def test_compute_anchors_mean_ignores_polar():
    phases = {
        date(2024, 6, 1): (4.0, 6.0, 18.0, 20.0),
        date(2024, 6, 2): (5.0, 7.0, 19.0, 21.0),
        date(2024, 12, 21): None,  # should be ignored
    }
    anchors = compute_anchors(phases)
    assert anchors is not None
    mean_sunrise, mean_sunset = anchors
    assert mean_sunrise == pytest.approx(6.5)
    assert mean_sunset == pytest.approx(18.5)


def test_compute_anchor_bands_mean_ignores_polar():
    phases = {
        date(2024, 6, 1): (4.0, 6.0, 18.0, 20.0),
        date(2024, 6, 2): (6.0, 8.0, 20.0, 22.0),
        date(2024, 12, 21): None,
    }
    bands = compute_anchor_bands(phases)
    assert bands is not None
    dawn, sunrise, sunset, dusk = bands
    assert dawn == pytest.approx(5.0)
    assert sunrise == pytest.approx(7.0)
    assert sunset == pytest.approx(19.0)
    assert dusk == pytest.approx(21.0)


# ---------------------------------------------------------------------------
# transform_to_sun_time
# ---------------------------------------------------------------------------


def test_transform_identity_when_anchors_match_day():
    """Per-day sunrise/sunset equal the anchors => output equals input."""
    d = date(2024, 6, 15)
    phases = {d: (4.5, 6.0, 18.0, 19.5)}
    obs = [(8.0, d), (12.0, d), (17.0, d)]
    result, dropped = transform_to_sun_time(
        obs, phases, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert dropped == 0
    assert result == pytest.approx([8.0, 12.0, 17.0])


def test_transform_identity_when_anchors_match_night():
    """Night-branch identity: 23:00 on a day with sunset 18:00 and
    anchors at 06:00/18:00 should stay at 23:00.
    """
    d = date(2024, 6, 15)
    phases = {d: (4.5, 6.0, 18.0, 19.5)}
    obs = [(23.0, d), (2.0, d)]
    result, dropped = transform_to_sun_time(
        obs, phases, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert dropped == 0
    assert result == pytest.approx([23.0, 2.0])


def test_transform_day_stretch():
    """Short day (8h) stretched to long anchor day (12h).

    A detection at solar noon on the short day should land at solar
    noon on the anchored frame. Short day: sunrise 08:00, sunset 16:00
    (noon 12:00). Anchor: sunrise 06:00, sunset 18:00 (noon 12:00).
    """
    d = date(2024, 12, 21)
    phases = {d: (7.0, 8.0, 16.0, 17.0)}
    obs = [(12.0, d)]
    result, dropped = transform_to_sun_time(
        obs, phases, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert dropped == 0
    assert result[0] == pytest.approx(12.0)
    # And sunrise on the short day must land exactly on the anchor sunrise.
    result2, _ = transform_to_sun_time(
        [(8.0, d)], phases, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert result2[0] == pytest.approx(6.0)
    # And sunset on the short day must land on anchor sunset.
    result3, _ = transform_to_sun_time(
        [(16.0, d)], phases, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert result3[0] == pytest.approx(18.0)


def test_transform_night_wraparound():
    """Post-midnight observation uses the same day's sunset as anchor.

    Day with sunrise 06:00 and sunset 18:00 => night length 12h.
    Anchor: sunrise 06:00, sunset 18:00 => same night length => identity.
    Use a day with different day length to verify wraparound stretching.
    """
    d = date(2024, 12, 21)
    # Short day (10h): 07:00 sunrise, 17:00 sunset; night = 14h.
    phases = {d: (6.0, 7.0, 17.0, 18.0)}
    # Anchor: 12h day / 12h night. Night stretches from 14h -> 12h.
    # elapsed_night at 23:00 = (23 - 17) mod 24 = 6. Night factor = 12/14.
    # t_sun = 18 + 6 * (12/14) mod 24 = 18 + 5.1428... = ~23.14.
    result, _ = transform_to_sun_time(
        [(23.0, d)], phases, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert result[0] == pytest.approx(18 + 6 * (12 / 14))
    # Observation just past midnight: 02:00 on the same day.
    # elapsed_night = (2 - 17) mod 24 = 9. t_sun = 18 + 9 * 12/14 mod 24
    #               = 18 + 7.714 mod 24 = 25.714 mod 24 = 1.714.
    result2, _ = transform_to_sun_time(
        [(2.0, d)], phases, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert result2[0] == pytest.approx((18 + 9 * (12 / 14)) % 24)


def test_transform_drops_polar_dates():
    polar_day = date(2024, 12, 21)
    normal_day = date(2024, 6, 21)
    phases = {polar_day: None, normal_day: (4.5, 6.0, 18.0, 19.5)}
    obs = [(10.0, polar_day), (12.0, normal_day), (15.0, polar_day)]
    result, dropped = transform_to_sun_time(
        obs, phases, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert dropped == 2
    assert len(result) == 1
    assert result[0] == pytest.approx(12.0)


def test_transform_empty_input():
    result, dropped = transform_to_sun_time(
        [], {}, anchor_sunrise=6.0, anchor_sunset=18.0
    )
    assert result == []
    assert dropped == 0
