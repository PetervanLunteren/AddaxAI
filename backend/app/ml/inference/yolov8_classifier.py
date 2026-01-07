"""
YOLOv8 classification model implementation (addax-yolov8 type).

Following DEVELOPERS.MD principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere

CRITICAL: Uses EXACT cropping algorithm from streamlit-AddaxAI classify_detections.py
to guarantee identical results.

Created by Claude Code on 2026-01-04
"""

import csv
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageFile, ImageOps
from ultralytics import YOLO

from app.core.logging_config import get_logger
from app.ml.inference.base import BoundingBox, ClassificationModel, ClassificationResult

# Allow loading truncated images (common in camera trap data)
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__)


class YOLOv8Classifier(ClassificationModel):
    """
    YOLOv8 classification model for species identification.

    Used by models with type="addax-yolov8" in their manifest.
    Examples: NAM-ADS-v1 (Namibian Desert), NZI-ADS-v1 (New Zealand).

    Cropping algorithm matches streamlit-AddaxAI exactly (classify_detections.py:80-114)
    to ensure identical results.
    """

    def __init__(
        self,
        model_path: Path,
        taxonomy_path: Path | None = None,
    ):
        """
        Initialize YOLOv8 classifier.

        Args:
            model_path: Path to .pt model file
            taxonomy_path: Optional path to taxonomy.csv for class metadata

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading YOLOv8 model from: {model_path}")

        try:
            # Load YOLO model
            self.model = YOLO(str(model_path))
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {e}") from e

        # Check GPU availability
        self.gpu_available = self._check_gpu()
        logger.info(f"GPU available: {self.gpu_available}")

        # Load taxonomy if provided
        self.taxonomy = {}
        if taxonomy_path and taxonomy_path.exists():
            self.taxonomy = self._load_taxonomy(taxonomy_path)
            logger.info(f"Loaded taxonomy with {len(self.taxonomy)} classes")

    def _check_gpu(self) -> bool:
        """
        Check GPU availability (MPS for Mac, CUDA for others).

        Returns:
            True if GPU is available and working
        """
        try:
            # Check for Apple Silicon MPS
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                logger.info("Using Apple Silicon MPS GPU")
                return True
        except Exception:
            pass

        # Check for CUDA
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU")
            return True

        logger.info("No GPU available, using CPU")
        return False

    def _load_taxonomy(self, taxonomy_path: Path) -> dict[str, dict]:
        """
        Load taxonomy CSV with class metadata.

        Format: model_class,class,order,family,genus,species

        Args:
            taxonomy_path: Path to taxonomy.csv

        Returns:
            Dict mapping class name -> taxonomy dict
        """
        taxonomy = {}

        try:
            with open(taxonomy_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class_name = row["model_class"].strip()
                    taxonomy[class_name] = {
                        "class": row.get("class", "").strip(),
                        "order": row.get("order", "").strip(),
                        "family": row.get("family", "").strip(),
                        "genus": row.get("genus", "").strip(),
                        "species": row.get("species", "").strip(),
                    }
        except Exception as e:
            logger.warning(f"Failed to load taxonomy: {e}")

        return taxonomy

    def classify(
        self,
        image_path: Path,
        bbox: BoundingBox,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ClassificationResult:
        """
        Classify animal detection using YOLOv8.

        Uses EXACT cropping algorithm from streamlit-AddaxAI to guarantee
        identical results.

        Args:
            image_path: Path to image file
            bbox: Bounding box in normalized coordinates
            progress_callback: Optional progress callback (unused for single inference)

        Returns:
            ClassificationResult with species and all probabilities

        Raises:
            ValueError: If crop fails (invalid bbox)
            RuntimeError: If classification fails
            FileNotFoundError: If image_path does not exist
        """
        try:
            # Load image from path
            image = Image.open(image_path)

            # Apply exact cropping algorithm from streamlit
            crop = self._get_crop(image, bbox)

            if crop is None:
                raise ValueError("Failed to create crop from bbox")

            # Run YOLOv8 classification inference
            results = self.model(crop, verbose=False)

            # Extract class names and probabilities
            names_dict = results[0].names  # {0: "aardwolf", 1: "african wild cat", ...}
            probs = results[0].probs.data.tolist()  # [0.001, 0.002, ..., 0.997, ...]

            # Build complete probability dictionary
            all_probabilities = {}
            for idx, class_name in names_dict.items():
                # Round to 5 decimal places to match streamlit output
                all_probabilities[class_name] = round(float(probs[idx]), 5)

            # Get top prediction
            top_species = max(all_probabilities, key=all_probabilities.get)
            top_confidence = all_probabilities[top_species]

            logger.debug(
                f"Classified as '{top_species}' with confidence {top_confidence:.5f}"
            )

            return ClassificationResult(
                species=top_species,
                confidence=top_confidence,
                all_probabilities=all_probabilities,
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            raise RuntimeError(f"YOLOv8 classification failed: {e}") from e

    def _get_crop(self, img: Image.Image, bbox_norm: BoundingBox) -> Image.Image | None:
        """
        EXACT cropping algorithm from streamlit-AddaxAI classify_detections.py:80-114

        Critical for matching results - DO NOT MODIFY!

        This function:
        1. Converts normalized bbox to pixel coordinates
        2. Makes crop square by taking max(width, height)
        3. Applies smart padding (prevents over-enlargement of small animals)
        4. Centers the square crop on the original bbox
        5. Pads with zeros to create perfect square

        Args:
            img: PIL Image object
            bbox_norm: Normalized bounding box (0-1)

        Returns:
            Cropped and padded PIL Image, or None if invalid
        """
        img_w, img_h = img.size

        # Convert normalized to pixel coordinates
        xmin = int(bbox_norm.x * img_w)
        ymin = int(bbox_norm.y * img_h)
        box_w = int(bbox_norm.width * img_w)
        box_h = int(bbox_norm.height * img_h)

        # Make square crop by taking max dimension
        box_size = max(box_w, box_h)

        # Apply smart padding algorithm
        box_size = self._pad_crop(box_size)

        # Center the square crop on the original bbox
        xmin = max(
            0, min(xmin - int((box_size - box_w) / 2), img_w - box_w)
        )
        ymin = max(
            0, min(ymin - int((box_size - box_h) / 2), img_h - box_h)
        )

        # Ensure crop doesn't exceed image bounds
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

        # Invalid bbox check
        if box_w == 0 or box_h == 0:
            logger.warning("Invalid bbox dimensions (zero width or height)")
            return None

        # Crop the image
        crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

        # Pad to perfect square with zero (black) padding
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

        return crop

    def _pad_crop(self, box_size: int) -> int:
        """
        EXACT padding algorithm from streamlit-AddaxAI classify_detections.py:102-114

        Prevents small animals from being over-enlarged during cropping.

        Logic:
        - If box >= 224px: Add 30px padding
        - If box < 224px and diff < 30px: Add 30px padding
        - If box < 224px and diff >= 30px: Use 224px (network input size)

        Args:
            box_size: Original box size in pixels

        Returns:
            Padded box size
        """
        input_size_network = 224  # YOLOv8 input size
        default_padding = 30

        diff_size = input_size_network - box_size

        if box_size >= input_size_network:
            return box_size + default_padding
        elif diff_size < default_padding:
            return box_size + default_padding
        else:
            return input_size_network
