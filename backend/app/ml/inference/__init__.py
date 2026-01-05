"""
ML Inference Module

Clean, type-safe implementations of detection and classification models.
Following DEVELOPERS.md principles: crash early, explicit config, type hints everywhere.

Architecture:
- base.py: Abstract base classes and data structures
- megadetector.py: MegaDetector v1000 implementation
- yolov8_classifier.py: YOLOv8 classification models
- Other classifier types can be added following the same pattern

Created by Claude Code on 2026-01-04
"""

from app.ml.inference.base import (
    BoundingBox,
    ClassificationResult,
    DetectionModel,
    DetectionResult,
    ClassificationModel,
    PipelineResult,
)

__all__ = [
    "BoundingBox",
    "ClassificationResult",
    "DetectionModel",
    "DetectionResult",
    "ClassificationModel",
    "PipelineResult",
]
