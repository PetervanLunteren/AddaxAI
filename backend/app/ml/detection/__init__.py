"""Detection module for running ML detection models."""

from app.core.confidence import (
    DEFAULT_CLASSIFICATION_GATE,
    MD_OUTPUT_CONFIDENCE_THRESHOLD,
)

# The confidence constants live in app/core/confidence.py (single
# source of truth, mirrored by frontend/src/lib/confidence.ts); they
# are re-exported here because the detection worker imports them from
# this package. The live MegaDetector runners are MegaDetectorV1000
# (images) and VideoDetectionModel (videos) in app/ml/inference/.
__all__ = [
    "DEFAULT_CLASSIFICATION_GATE",
    "MD_OUTPUT_CONFIDENCE_THRESHOLD",
]
