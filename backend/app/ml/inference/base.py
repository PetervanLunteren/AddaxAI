"""
Base classes and data structures for ML inference.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Crash early if configuration invalid
- Explicit, not implicit
- Clean abstractions for extensibility

Created by Claude Code on 2026-01-04
"""

from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    """
    Classification result for a single detection.

    Contains both top prediction and all class probabilities
    for uncertainty analysis.
    """

    label: str  # Top prediction class name (e.g., "giraffe")
    confidence: float  # Top prediction confidence (0.0 - 1.0)
    all_probabilities: dict[str, float]  # All classes with confidences

    def __post_init__(self) -> None:
        """Validate classification data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if self.label not in self.all_probabilities:
            raise ValueError(f"Top label '{self.label}' not in probabilities")
        if abs(self.all_probabilities[self.label] - self.confidence) > 0.0001:
            raise ValueError("Top confidence mismatch with all_probabilities")


@dataclass
class PipelineResult:
    """
    Result summary from complete ML pipeline run.

    Used for reporting statistics after processing.
    """

    total_files: int
    total_detections: int
    animal_detections: int
    classified_detections: int
    person_detections: int = 0
    vehicle_detections: int = 0
    exclusion_rollup_entries: list[dict] = field(default_factory=list)


class DetectionModel:
    """
    Abstract base class for detection models.

    Detection models locate objects in images and return bounding boxes.
    Examples: MegaDetector, custom object detectors.
    """

    pass


class ClassificationModel:
    """
    Base class for classification models.

    Classification models identify labels in cropped detections.
    Examples: YOLOv8 classifiers, SpeciesNet, DeepFaune, MEWC.
    """

    pass
