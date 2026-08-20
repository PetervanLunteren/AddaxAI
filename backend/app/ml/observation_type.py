"""Single source of truth for which detection a file is about.

Two functions, one rule. ``strongest_passing_detection`` picks the
detection; ``derive_observation_type`` reads that detection's category.
Anything that needs another attribute of the deciding box (the Files
export needs its species) calls the first one, so the ordering rule
still has exactly one implementation.

**The iterable you pass must already be the file's visible surface.**
This module is frame-blind on purpose, so for a video the caller filters
first, with ``on_visible_frame()`` / ``on_visible_frame_of()`` in a query
or ``visible_detections(file, dets)`` on a list already in memory (both in
``ml/detection_visibility.py``). Pass a video's raw detections here and
you will summarise it by a box on a frame that was never written to disk,
which is a label the user can neither see nor correct.

``observation_type`` is a denormalised summary of a file's *trusted*
content: **the raw detector category of its single strongest passing
detection**, else ``"blank"``. A detection passes when it clears the
project detection threshold OR a human has verified it, the same rule
applied everywhere detections are shown (see DEVELOPERS.md "Detection
threshold and verified override"). A file whose every detection sits
below the threshold has no trusted content and reads as ``"blank"``,
exactly as the verify grid hides those detections.

Strongest means verified first, then detector confidence. Verification
outranks confidence because a human looked at that box, the same
principle as the verified-override on the threshold itself. Note that it
does not *count*: ten verified foxes and one verified deer on one file
resolve to whichever single box scored highest, not to the fox.
``build_event_primary_labels`` does count, deliberately, because an event
is many noisy looks at one animal where a file is a single look. See
DEVELOPERS.md "What a file is about" before unifying the two.

**A detection labelled as a non-label class cannot be the subject.**
``bait``, ``blank``, ``empty``, ``false detection``, ``none``, ``vide``:
these say "there is nothing here", so a file holding only those is
``blank``. The AI's own such calls never reach the database, dropped by
the ingest skip (see DEVELOPERS.md "Non-label detection skip"). This is
the same rule applied at read time, for the case the ingest skip cannot
reach: a human pressing X on the Labels page reaches the identical
verdict later, and until this existed the file stayed ``animal``, because
"Mark false" writes the label and deliberately leaves the detector's
category alone. Measured: such a file exported ``observation_type=animal``
with ``classification_label=false detection`` beside it.

The row is kept on purpose rather than deleted: a human looked at that
box and judged it, which is worth recording, keeps Ctrl+Z working, and
keeps ``detections.csv`` an honest record of what the detector found and
what was rejected.

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
from typing import Protocol, TypeVar

from app.ml.label_exclusion import is_non_label

# A file with no trusted content. Not a detector category, so it can
# never collide with one.
BLANK = "blank"


class _DetectionLike(Protocol):
    category: str
    confidence: float
    verified: bool
    label: str | None


# Generic in, generic out: the caller gets back one of the objects it
# passed in, not the bare Protocol, so it can read attributes the
# Protocol does not declare (``label``, ``common_name``, ...).
_D = TypeVar("_D", bound=_DetectionLike)


def strongest_passing_detection(
    detections: Iterable[_D], threshold: float
) -> _D | None:
    """The single detection a file is about, or ``None`` when none passes.

    ``detections`` is any iterable of objects exposing ``category``,
    ``confidence``, and ``verified``. A detection passes when it clears
    ``threshold`` or a human has verified it.

    The sort key is ``(verified, confidence, category)``. The category is
    in there only to keep the result stable when two detections tie on
    both of the first two. It is not enough to make the *object* unique:
    two equally strong boxes of the same category with different species
    still tie, and the first one in iteration order wins. A caller that
    needs a stable pick therefore has to pass a stably ordered sequence.
    ``build_files_rows`` does, via its query's ``ORDER BY ... Detection.id``.

    The detection id is deliberately not in the key. Callers pass
    SQLAlchemy Rows selected without it (``output_preview``), so adding it
    to the Protocol would break them for a tie-break nobody can observe in
    the category.
    """
    best: _D | None = None
    best_key: tuple[bool, float, str] | None = None
    for det in detections:
        if not (det.confidence >= threshold or det.verified):
            continue
        if is_non_label(det.label):
            continue
        key = (bool(det.verified), float(det.confidence), det.category)
        if best_key is None or key > best_key:
            best = det
            best_key = key
    return best


def derive_observation_type(
    detections: Iterable[_DetectionLike], threshold: float
) -> str:
    """The file's ``observation_type`` from its detections at ``threshold``.

    The raw category of the strongest passing detection, or ``"blank"``
    when nothing passes.
    """
    best = strongest_passing_detection(detections, threshold)
    return best.category if best is not None else BLANK
