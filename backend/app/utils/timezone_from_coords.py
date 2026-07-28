"""Derive an IANA timezone from coordinates, offline.

Used to set a project's camera timezone from its first site's GPS, so the
sun-based insights (sun-band overlay, the Vazquez sun-time transform, diel
classification) use the camera's actual zone instead of the browser's.

Uses timezonefinder's lightweight ``TimezoneFinderL``: it resolves to a zone
with the correct UTC offset and DST rules, though the IANA *label* can be a
same-offset neighbour (a Dutch coordinate resolves to ``"Europe/Berlin"``, not
``"Europe/Amsterdam"``). That's fine for sun math, and users can override the
exact label in Project settings. Choosing the lite class lets the build prune
the 62 MB full-polygon data file (see backend.spec).
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _finder():
    """Lazily build the TimezoneFinderL singleton (loads ~1.5 MB once)."""
    from timezonefinder import TimezoneFinderL

    return TimezoneFinderL()


def tz_from_coords(lat: float, lon: float) -> str | None:
    """Return the IANA timezone name for a coordinate, or None if unresolved.

    TimezoneFinderL returns a nautical ``Etc/GMT±N`` zone over open ocean and
    None only for inputs it cannot place, so a real camera site effectively
    always resolves.
    """
    return _finder().timezone_at(lat=lat, lng=lon)
