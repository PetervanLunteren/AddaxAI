"""The track post-filter's two values per data domain.

After tracking, a track that lasted under a second of sampled frames or
whose centre barely moved is dropped, unless its best box was confident
enough to be trusted anyway. The rule is SharkTrack's (Varini et al.
2024, supplement S3); the two numbers it turns on are the domain's.

Why two values and not one. The rule rests on "a real animal moves".
Under water that is nearly always true, and a static "shark" in turbid
seagrass is almost always a false box, so SharkTrack could drop a track
that moved under 8% of the frame unless it scored 0.7, removing 40% of
false boxes at under 0.08% true-positive loss. On a camera trap a still
animal is ordinary. The tracking benchmark of 2026-09-14
(``TRACKING_BENCHMARK_RESULTS.md``) measured that SharkTrack's values
lift camera-trap track counts from 49% to 70% exact at 3 fps but empty
9% of clips of a resting animal (the track is deleted, the clip is
filed blank); 0.06 / 0.5 keeps 65% and loses 4%, against 2% with no
filter. The one-second life rule is the same in both domains and stays
a constant in the tracking script. Nothing marine was measured; the
underwater values are SharkTrack's published ones.

The domain comes from the detector's catalog entry (``ModelManifest.
domain``), the same values the setup forms' Data type toggle is keyed
on, so a run can never carry a filter its footage did not ask for.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.schemas.model_manifest import ModelManifest


@dataclass(frozen=True)
class TrackFilter:
    # A track that moved less than this fraction of the frame, on the
    # axis it moved most, is nearly static.
    min_motion: float
    # A track whose best box reaches this confidence survives regardless.
    exempt_conf: float


TRACK_FILTER_BY_DOMAIN: dict[str, TrackFilter] = {
    "camera_trap": TrackFilter(min_motion=0.06, exempt_conf=0.5),
    "underwater": TrackFilter(min_motion=0.08, exempt_conf=0.7),
}


def track_filter_for(manifest: ModelManifest) -> TrackFilter:
    """The filter for a detector, from its catalog domain. A detector
    without one, or with a value this app does not know, is a catalog
    error, reported by name rather than run with somebody else's
    numbers."""
    if manifest.domain not in TRACK_FILTER_BY_DOMAIN:
        raise ValueError(
            f"detector {manifest.model_id} declares domain {manifest.domain!r} in the "
            f"catalog; expected one of {sorted(TRACK_FILTER_BY_DOMAIN)}"
        )
    return TRACK_FILTER_BY_DOMAIN[manifest.domain]
