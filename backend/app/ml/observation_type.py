"""Single source of truth for a file's ``observation_type``.

``observation_type`` is a denormalised summary of a file's *trusted*
content: the highest-priority detector category (animal > human > vehicle)
among the file's passing detections, else ``"blank"``. A detection passes
when it clears the project detection threshold OR a human has verified it,
the same rule applied everywhere detections are shown (see DEVELOPERS.md
"Detection threshold and verified override"). A file whose every detection
sits below the threshold has no trusted content and reads as ``"blank"``,
exactly as the verify grid hides those detections.

Because it depends on the mutable project threshold and per-detection
verified state, it must be recomputed whenever a file's detections change,
a detection is verified, or the project threshold changes. Keeping the
derivation here means ingestion, reprocessing, the detection endpoints, and
the threshold-change hook all agree.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

# Detector category -> observation type. Priority order below decides which
# category wins on a mixed file (an animal + a person -> "animal").
_CATEGORY_TO_OBS: dict[str, str] = {
    "animal": "animal",
    "person": "human",
    "vehicle": "vehicle",
}
_OBS_PRIORITY: dict[str, int] = {"animal": 3, "human": 2, "vehicle": 1}


class _DetectionLike(Protocol):
    category: str
    confidence: float
    verified: bool


def derive_observation_type(
    detections: Iterable[_DetectionLike], threshold: float
) -> str:
    """The file's ``observation_type`` from its detections at ``threshold``.

    ``detections`` is any iterable of objects exposing ``category``,
    ``confidence``, and ``verified``. Only detections that clear the
    threshold or are verified count; a category the map doesn't know is
    ignored. Returns ``"blank"`` when nothing passes.
    """
    best = "blank"
    best_priority = 0
    for det in detections:
        if not (det.confidence >= threshold or det.verified):
            continue
        obs = _CATEGORY_TO_OBS.get(det.category)
        if obs is None:
            continue
        priority = _OBS_PRIORITY[obs]
        if priority > best_priority:
            best_priority = priority
            best = obs
    return best
