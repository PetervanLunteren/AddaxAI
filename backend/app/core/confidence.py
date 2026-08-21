"""Single source of truth for the app's confidence defaults.

The design follows Dan Morris's three-threshold scheme (beta feedback
2026-07): MegaDetector runs effectively untresholded, the expensive per-crop model
passes are gated explicitly, and counting / visualization has its own
read-time filter. One constant per concept; every default in models,
schemas, and workers references these. The frontend mirrors them in
``frontend/src/lib/confidence.ts`` — keep the two files in sync.

MD_OUTPUT_CONFIDENCE_THRESHOLD — what MegaDetector writes to
results.json. Everything at or above it is preserved end to end: raw
JSON, database, and the folder-run data exports.

This is our cap, not MegaDetector's. MD's own floor is 0.005 and its
documentation advises never going below that, so 0.01 sits safely
above it. The reason to cap higher is that 0.005 stored a tail no
part of the app could ever address: every confidence slider bottoms
out at 0.01 (``CONFIDENCE_SCALE_MIN`` in the shared slider), the
classification gate starts at 0.1, counting at 0.2, and best-frame
scoring at 0.3. Measured on a real database, 19% of all detection
rows sat between 0.005 and 0.01, none of them ever verified, ever
classified, ever counted or ever visible. They only cost disk, delete
time and query time.

Raising it changes nothing downstream, because every consumer already
sits far above 0.01. It applies to new analyses only; rows already
stored below it stay, harmless and unreachable, until a re-analysis
replaces them.

DEFAULT_CLASSIFICATION_GATE — default detection confidence above which
an animal detection is classified AND embedded. Configurable per
project via ``Project.classification_gate``.

DEFAULT_COUNTING_THRESHOLD — default detection confidence for what
gets counted and visualized: ``Project.counting_threshold``, the
save step's media confidence, and the labels grid's seeded filter all
default to it. Below this value most detections are false positives,
which is also why the grid shows its noise advisory there.

A future classification-confidence counting threshold (Dan's third
knob, ~0.6 for SpeciesNet) gets its constant here when that feature
is designed; it interacts with taxonomic rollup and is deliberately
not implemented yet.
"""

MD_OUTPUT_CONFIDENCE_THRESHOLD = 0.01
DEFAULT_CLASSIFICATION_GATE = 0.1
DEFAULT_COUNTING_THRESHOLD = 0.2

# ROLLUP_THRESHOLD — the confidence below which taxonomic rollup rolls a
# species call up the tree (species -> genus -> family -> ...). Fixed
# policy, not a preference: it is never user-facing or per-project, and
# most users should not think about it. The rollup functions in
# app/ml/taxonomic_rollup.py default to this; it lives here so all
# confidence policy sits in one file.
ROLLUP_THRESHOLD = 0.65


def effective_floor(
    counting_threshold: float, min_confidence: float | None
) -> float:
    """The floor the Labels page is currently showing at.

    Never above ``counting_threshold``: the confidence slider digs *down*
    into the low-confidence tail, it does not raise the bar. A user
    narrowing to a range above the threshold is filtering what they see,
    which callers apply separately and literally; it does not change what
    counts as passing.

    One helper because two surfaces depend on agreeing exactly. The crop
    grid shows the detections above this floor; the empties grid shows
    the files with none. If the two computed it differently, a photo
    could appear in both tabs or in neither.
    """
    if min_confidence is None:
        return counting_threshold
    return min(counting_threshold, min_confidence)


def format_confidence_pct(value: float) -> str:
    """Render a 0-1 confidence as a whole-percent string for humans
    (sliders, captions, the run README, drawn boxes). Data files keep
    the raw 0-1 value."""
    return f"{round(value * 100)}%"
