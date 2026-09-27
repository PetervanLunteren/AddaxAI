"""Compatibility import surface for manifest-driven detector backends."""

from app.ml.inference.detector_backend import (
    DetectorFactory,
    IsolatedSubprocessRunner,
    MegaDetectorAdapter,
    RFDETRDetectorAdapter,
    YoloDetectorAdapter,
    class_category_map,
    create_detector,
    normalize_detections,
    normalize_image_results,
    normalize_video_results,
)
from app.ml.schemas.model_manifest import DetectorBackend, ModelManifest

__all__ = [
    "DetectorFactory",
    "DetectorBackend",
    "IsolatedSubprocessRunner",
    "MegaDetectorAdapter",
    "RFDETRDetectorAdapter",
    "YoloDetectorAdapter",
    "class_category_map",
    "create_detector",
    "normalize_detections",
    "normalize_image_results",
    "normalize_video_results",
    "ModelManifest",
]
