"""Single source of truth for the app's confidence defaults.

The design follows Dan Morris's three-threshold scheme (beta feedback
2026-07): MegaDetector runs untresholded, the expensive per-crop model
passes are gated explicitly, and counting / visualization has its own
read-time filter. One constant per concept; every default in models,
schemas, and workers references these. The frontend mirrors them in
``frontend/src/lib/confidence.ts`` — keep the two files in sync.

MD_OUTPUT_CONFIDENCE_THRESHOLD — what MegaDetector writes to
results.json. MD's own internal default (its documentation advises
never going below it). Everything above it is preserved end to end:
raw JSON, database, and the folder-run data exports.

DEFAULT_CLASSIFICATION_GATE — default detection confidence above which
an animal detection is classified AND embedded. Configurable per
project via ``Project.classification_gate``.

DEFAULT_COUNTING_THRESHOLD — default detection confidence for what
gets counted and visualized: ``Project.detection_threshold``, the
save step's media confidence, and the labels grid's seeded filter all
default to it. Below this value most detections are false positives,
which is also why the grid shows its noise advisory there.

A future classification-confidence counting threshold (Dan's third
knob, ~0.6 for SpeciesNet) gets its constant here when that feature
is designed; it interacts with taxonomic rollup and is deliberately
not implemented yet.
"""

MD_OUTPUT_CONFIDENCE_THRESHOLD = 0.005
DEFAULT_CLASSIFICATION_GATE = 0.1
DEFAULT_COUNTING_THRESHOLD = 0.2
