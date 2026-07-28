"""
Sun-time transformation for the Activity overlap plot.

Implements the double-anchored mapping from Vazquez, C., Rowcliffe, J. M.,
Spoelstra, K., & Jansen, P. A. (2019). Comparing diel activity patterns of
wildlife across latitudes and seasons: time transformations using day
length. Methods in Ecology and Evolution, 10(12), 2057-2066.

Each observation's clock hour is transformed so that the day's sunrise /
sunset map to the dataset's mean sunrise / sunset, with daylight and
nighttime portions each linearly stretched or compressed. Detections
collected across seasons or latitudes can then be pooled honestly on a
single x-axis.

The module is pure Python: no database, no FastAPI, no numpy. Fed from
`app.api.crud.statistics.get_activity_overlap`. Unit-tested in
`backend/tests/ml/test_sun_time.py`.
"""

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun

# Four fractional-hour values per date: (dawn, sunrise, sunset, dusk).
# All four come from a single astral.sun() call; civil twilight (dawn / dusk)
# is included so the caller can paint anchor bands on the chart.
DayPhases = tuple[float, float, float, float]


def _hour(dt: datetime) -> float:
    return dt.hour + dt.minute / 60 + dt.second / 3600


def per_date_sun_phases(
    dates: Iterable[date],
    *,
    lat: float,
    lon: float,
    tz_name: str,
) -> dict[date, DayPhases | None]:
    """Look up civil twilight + sun times for every unique date in `dates`.

    Returns a dict keyed by date. Each value is a tuple
    `(dawn, sunrise, sunset, dusk)` of fractional hours in the project's
    local wall-clock, or `None` when astral refuses to compute (polar
    night / day, or any other input the library rejects).

    Duplicates in `dates` are deduped internally so the caller does not
    need to pre-filter. Each unique date costs one astral call.
    """
    location = LocationInfo("project", "project", tz_name, lat, lon)
    tz = ZoneInfo(tz_name)

    out: dict[date, DayPhases | None] = {}
    for d in set(dates):
        try:
            s = sun(location.observer, date=d, tzinfo=tz)
        except ValueError:
            out[d] = None
            continue
        dawn_h = _hour(s["dawn"])
        sunrise_h = _hour(s["sunrise"])
        sunset_h = _hour(s["sunset"])
        dusk_h = _hour(s["dusk"])
        # Near the solstices at high latitudes astral returns dusk at
        # the *start* of the calendar date (i.e. just past midnight of
        # the previous night) rather than the natural "after sunset".
        # Unwrap so phases stay monotonic within the day that owns this
        # sunrise. Symmetric for dawn in case astral reports it on the
        # previous evening. Callers that average these (anchor_sun_bands)
        # must mod 24 the mean to get it back into [0, 24).
        if dusk_h < sunset_h:
            dusk_h += 24
        if dawn_h > sunrise_h:
            dawn_h -= 24
        out[d] = (dawn_h, sunrise_h, sunset_h, dusk_h)
    return out


def compute_anchors(
    phases: Mapping[date, DayPhases | None],
) -> tuple[float, float] | None:
    """Arithmetic mean `(sunrise, sunset)` across non-`None` entries.

    Returns `None` when every entry is polar. Used as the reference
    frame for the sun-time transform.
    """
    valid = [p for p in phases.values() if p is not None]
    if not valid:
        return None
    mean_sunrise = sum(p[1] for p in valid) / len(valid)
    mean_sunset = sum(p[2] for p in valid) / len(valid)
    return mean_sunrise, mean_sunset


def compute_anchor_bands(
    phases: Mapping[date, DayPhases | None],
) -> DayPhases | None:
    """Mean `(dawn, sunrise, sunset, dusk)` across non-`None` entries.

    Returns `None` when every entry is polar. Used to paint twilight
    bands at the anchor positions on the chart. Dawn and dusk may be
    stored outside `[0, 24)` by `per_date_sun_phases` to keep phases
    monotonic across solstice wrap; this function wraps the final mean
    back into `[0, 24)` for display.
    """
    valid = [p for p in phases.values() if p is not None]
    if not valid:
        return None
    return (
        (sum(p[0] for p in valid) / len(valid)) % 24,
        (sum(p[1] for p in valid) / len(valid)) % 24,
        (sum(p[2] for p in valid) / len(valid)) % 24,
        (sum(p[3] for p in valid) / len(valid)) % 24,
    )


def transform_to_sun_time(
    observations: list[tuple[float, date]],
    phases: Mapping[date, DayPhases | None],
    *,
    anchor_sunrise: float,
    anchor_sunset: float,
) -> tuple[list[float], int]:
    """Apply the Vazquez double-anchor transform.

    For each `(clock_hour, date)` in `observations`, look up that date's
    sunrise / sunset from `phases`. Observations on polar dates (phase
    value `None`) are dropped. The rest are transformed so that:

    - the daytime portion `[sunrise_d, sunset_d)` maps to
      `[anchor_sunrise, anchor_sunset)` with linear stretch / compress,
    - the nighttime portion (wrapping across midnight) maps to the
      complementary `[anchor_sunset, anchor_sunrise + 24)` window with
      linear stretch / compress.

    Returns `(sun_hours, dropped_polar_count)`.
    """
    anchor_day = (anchor_sunset - anchor_sunrise) % 24
    anchor_night = 24 - anchor_day

    out: list[float] = []
    dropped = 0
    for t_obs, d in observations:
        phase = phases.get(d)
        if phase is None:
            dropped += 1
            continue
        _, sunrise_d, sunset_d, _ = phase
        day_length = (sunset_d - sunrise_d) % 24
        night_length = 24 - day_length
        # Degenerate day (no sunset after sunrise, or vice versa). Astral
        # should already have returned None for these, but keep a guard.
        if day_length == 0 or night_length == 0:
            dropped += 1
            continue
        if sunrise_d <= t_obs < sunset_d:
            t_sun = (
                anchor_sunrise
                + (t_obs - sunrise_d) * (anchor_day / day_length)
            )
        else:
            elapsed_night = (t_obs - sunset_d) % 24
            t_sun = (
                anchor_sunset + elapsed_night * (anchor_night / night_length)
            ) % 24
        out.append(t_sun % 24)
    return out, dropped
