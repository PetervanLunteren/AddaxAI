"""Detection module for running ML detection models."""

from .megadetector_runner import MegaDetectorRunner

# Single source of truth for the two detection-confidence constants.
# Design follows Dan Morris's recommendation (beta feedback 2026-07):
# run MegaDetector untresholded and gate the expensive per-detection
# work explicitly, instead of one number silently playing every role.
#
# MD_OUTPUT_CONFIDENCE_THRESHOLD — what MegaDetector writes to
# results.json. This is MD's own internal default (its documentation
# advises never going below it). Everything above it is preserved end
# to end: raw JSON, database, and the folder-run data exports, which
# are the complete record of a run.
#
# DEFAULT_CLASSIFICATION_GATE — default detection confidence above
# which an animal detection is classified AND embedded (the two
# per-crop model passes). Configurable per project via
# `Project.classification_gate`; this constant is only the default.
# Without this gate, running MD at 0.005 would multiply classifier and
# embedding compute by the near-noise tail.
#
# The user-visible "Detection threshold" in project settings remains a
# *display* filter for counting / visualization; it replaces neither.
MD_OUTPUT_CONFIDENCE_THRESHOLD = 0.005
DEFAULT_CLASSIFICATION_GATE = 0.1

__all__ = [
    "DEFAULT_CLASSIFICATION_GATE",
    "MD_OUTPUT_CONFIDENCE_THRESHOLD",
    "MegaDetectorRunner",
]
