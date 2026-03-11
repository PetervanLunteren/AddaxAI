"""
Base classes and data structures for ML inference.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Crash early if configuration invalid
- Explicit, not implicit
- Clean abstractions for extensibility

Created by Claude Code on 2026-01-04
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox:
    """
    Normalized bounding box coordinates (0-1).

    Format matches MegaDetector output: [x, y, width, height]
    where x, y is top-left corner.
    """

    x: float  # Top-left X (0-1)
    y: float  # Top-left Y (0-1)
    width: float  # Width (0-1)
    height: float  # Height (0-1)

    def __post_init__(self) -> None:
        """Validate bbox coordinates."""
        if not (0 <= self.x <= 1 and 0 <= self.y <= 1):
            raise ValueError(f"Invalid bbox position: x={self.x}, y={self.y}")
        if not (0 < self.width <= 1 and 0 < self.height <= 1):
            raise ValueError(f"Invalid bbox size: width={self.width}, height={self.height}")
        if (
            self.x + self.width > 1.0001 or self.y + self.height > 1.0001
        ):  # Small tolerance for float precision
            raise ValueError(f"Bbox extends beyond image bounds: {self}")


@dataclass
class DetectionResult:
    """
    Single detection result from detection model.

    Matches MegaDetector output format with category mapping:
    - "1" -> "animal"
    - "2" -> "person"
    - "3" -> "vehicle"
    """

    file_path: Path  # Absolute path to image file
    category: str  # "animal", "person", or "vehicle"
    confidence: float  # 0.0 - 1.0
    bbox: BoundingBox  # Normalized coordinates

    def __post_init__(self) -> None:
        """Validate detection data."""
        if self.category not in ("animal", "person", "vehicle"):
            raise ValueError(f"Invalid category: {self.category}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")


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


class DetectionModel(ABC):
    """
    Abstract base class for detection models.

    Detection models locate objects in images and return bounding boxes.
    Examples: MegaDetector, custom object detectors.
    """

    @abstractmethod
    def detect(
        self,
        image_paths: list[Path],
        confidence_threshold: float,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> list[DetectionResult]:
        """
        Run detection on a list of images.

        Args:
            image_paths: Absolute paths to image files
            confidence_threshold: Minimum confidence for detections (0.0 - 1.0)
            progress_callback: Optional callback(message, progress) for updates
                               progress is 0.0 - 1.0

        Returns:
            List of DetectionResult objects, one per detection found

        Raises:
            RuntimeError: If detection fails
            FileNotFoundError: If image file doesn't exist
        """
        pass


class ClassificationModel(ABC):
    """
    Abstract base class for classification models.

    Classification models identify labels in cropped detections.
    Examples: YOLOv8 classifiers, SpeciesNet, DeepFaune, MEWC.
    """

    @abstractmethod
    def classify(
        self,
        image_path: Path,
        bbox: BoundingBox,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ClassificationResult:
        """
        Classify a detection crop from an image.

        Args:
            image_path: Path to original image file (worker will load and crop it)
            bbox: Bounding box in normalized coordinates
            progress_callback: Optional callback(message, progress) for updates

        Returns:
            ClassificationResult with label and all probabilities

        Raises:
            RuntimeError: If classification fails
            ValueError: If bbox is invalid
            FileNotFoundError: If image_path does not exist
        """
        pass
