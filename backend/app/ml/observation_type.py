"""Single source of truth for a file's ``observation_type``.

``observation_type`` is a denormalised summary of a file's *trusted*
content: **the raw detector category of its single strongest passing
detection**, else ``"blank"``. A detection passes when it clears the
project detection threshold OR a human has verified it, the same rule
applied everywhere detections are shown (see DEVELOPERS.md "Detection
threshold and verified override"). A file whose every detection sits
below the threshold has no trusted content and reads as ``"blank"``,
exactly as the verify grid hides those detections.

Strongest means verified first, then detector confidence. Verification
outranks confidence because a human looked at that box, which is the
same ordering ``build_event_primary_labels`` uses to pick an event's
folder, and the same principle as the verified-override on the
threshold itself.

**The category is passed through, never translated.** Whatever the
detector called it is what lands here: ``animal`` / ``person`` /
``vehicle`` from MegaDetector, and ``shark`` / ``fish`` / ``turtle``
from a detector that emits those. This module knows no vocabulary, so a
new detector needs no change here. The one place the value is
translated is the Camtrap DP export, whose ``observationType`` field has
a fixed controlled vocabulary (``crud/export.py``).

Until 2026-07-31 this ranked categories instead: a fixed priority
(animal > human > vehicle) decided a mixed file, so one animal box at
0.21 beat thirty person boxes at 0.95. That produced a clip of a person
in camouflage being filed as a chimpanzee, off a single false-positive
animal box the classifier guessed at 29%, while the picture beside it
was correctly labelled Person. Ranking by category cannot be right when
the thing being ranked is the detector's own guess about the category.

Because it depends on the mutable project threshold and per-detection
verified state, it must be recomputed whenever a file's detections
change, a detection is verified, or the project threshold changes.
Keeping the derivation here means ingestion, reprocessing, the detection
endpoints, and the threshold-change hook all agree.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

# A file with no trusted content. Not a detector category, so it can
# never collide with one.
BLANK = "blank"


class _DetectionLike(Protocol):
    category: str
    confidence: float
    verified: bool


def derive_observation_type(
    detections: Iterable[_DetectionLike], threshold: float
) -> str:
    """The file's ``observation_type`` from its detections at ``threshold``.

    ``detections`` is any iterable of objects exposing ``category``,
    ``confidence``, and ``verified``. Returns the raw category of the
    strongest passing detection, or ``"blank"`` when nothing passes.

    The sort key is ``(verified, confidence, category)``. The category
    is in there only to make the result deterministic when two
    detections tie on both of the first two: callers pass an unordered
    ORM collection, so without it the same file could derive differently
    between two runs.
    """
    best: tuple[bool, float, str] | None = None
    for det in detections:
        if not (det.confidence >= threshold or det.verified):
            continue
        key = (bool(det.verified), float(det.confidence), det.category)
        if best is None or key > best:
            best = key
    return best[2] if best is not None else BLANK
