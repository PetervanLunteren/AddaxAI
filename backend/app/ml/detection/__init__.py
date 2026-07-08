"""Detection module for running ML detection models."""

from app.core.confidence import (
    DEFAULT_CLASSIFICATION_GATE,
    MD_OUTPUT_CONFIDENCE_THRESHOLD,
)

from .megadetector_runner import MegaDetectorRunner

# The confidence constants live in app/core/confidence.py (single
# source of truth, mirrored by frontend/src/lib/confidence.ts); they
# are re-exported here because the detection worker imports them
# alongside MegaDetectorRunner.
__all__ = [
    "DEFAULT_CLASSIFICATION_GATE",
    "MD_OUTPUT_CONFIDENCE_THRESHOLD",
    "MegaDetectorRunner",
]
