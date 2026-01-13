"""
Template for Custom Classification Model Inference

This template provides a framework-agnostic interface for implementing custom
classification models in AddaxAI. Works with any ML framework (PyTorch, Keras,
TensorFlow, JAX, etc.) as long as it follows the class-based API below.

USAGE:
1. Copy this file to your model's HuggingFace repo as 'inference.py'
2. Implement the ModelInference class with your model-specific logic
3. AddaxAI will automatically discover and use your implementation

REQUIREMENTS:
- Python 3.10+
- Your model's dependencies (PyTorch, Keras, etc.) in your conda environment
- Model files in the same directory as this script

Created for AddaxAI-WebUI
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class ModelInference:
    """
    Custom inference implementation for your classification model.

    This class is instantiated once per worker process. The model is loaded
    in load_model() and reused for all subsequent classification requests,
    providing efficient batch processing.

    Lifecycle:
    1. __init__() - Store paths
    2. load_model() - Load model into memory (called once)
    3. get_classification() - Run inference (called many times)
    """

    def __init__(self, model_dir: Path, model_path: Path):
        """
        Initialize with model paths. Model loading happens in load_model().

        Args:
            model_dir: Directory containing model files (taxonomy.csv, class_list.yaml, etc.)
            model_path: Path to main model file (.pt, .keras, .h5, etc.)

        Example:
            model_dir = Path("/path/to/NAM-ADS-v1/")
            model_path = Path("/path/to/NAM-ADS-v1/namib_desert_v1.pt")
        """
        self.model_dir = model_dir
        self.model_path = model_path
        self.model = None  # Loaded in load_model()

        # Add any other initialization (load config files, etc.)

    def check_gpu(self) -> bool:
        """
        Check if GPU is available for your framework.

        Returns:
            True if GPU available, False otherwise

        Example (PyTorch):
            import torch
            return torch.cuda.is_available() or (
                torch.backends.mps.is_built() and torch.backends.mps.is_available()
            )

        Example (TensorFlow/Keras):
            import tensorflow as tf
            gpus = tf.config.list_logical_devices('GPU')
            return len(gpus) > 0
        """
        raise NotImplementedError("Implement GPU check for your framework")

    def load_model(self) -> None:
        """
        Load your model into memory. Called once at worker startup.

        This is where expensive operations happen (model file loading,
        weight initialization, etc.). The loaded model is stored in
        self.model and reused for all subsequent classifications.

        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If model loading fails

        Example (YOLOv8):
            from ultralytics import YOLO
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            self.model = YOLO(str(self.model_path))

        Example (Keras):
            from keras import saving
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            self.model = saving.load_model(str(self.model_path), compile=False)
        """
        raise NotImplementedError("Implement model loading for your framework")

    def get_crop(
        self, image: Image.Image, bbox: tuple[float, float, float, float]
    ) -> Image.Image:
        """
        Crop and preprocess image for your model's input requirements.

        Each model may have specific preprocessing needs: squaring, padding,
        resizing, normalization, etc. Implement your model's exact preprocessing
        here to ensure correct predictions.

        Args:
            image: Full-resolution PIL Image
            bbox: Normalized bounding box (x, y, width, height) in range [0.0, 1.0]
                  Format: (x_min, y_min, box_width, box_height)

        Returns:
            Cropped and preprocessed PIL Image ready for classification

        Raises:
            ValueError: If bbox is invalid (zero size, out of bounds, etc.)

        Example (Square crop with padding):
            from PIL import ImageOps

            img_w, img_h = image.size
            x = int(bbox[0] * img_w)
            y = int(bbox[1] * img_h)
            w = int(bbox[2] * img_w)
            h = int(bbox[3] * img_h)

            # Square the crop
            size = max(w, h)

            # Crop and pad to square
            crop = image.crop([x, y, x + w, y + h])
            crop = ImageOps.pad(crop, size=(size, size), color=0)

            return crop
        """
        raise NotImplementedError("Implement cropping for your model")

    def get_classification(self, crop: Image.Image) -> list[tuple[str, float]]:
        """
        Run classification inference on cropped image.

        This method is called for every animal detection. It should return
        probabilities for ALL classes, not just the top prediction.

        Args:
            crop: Preprocessed image from get_crop()

        Returns:
            List of (class_name, confidence) tuples for ALL classes.
            Must include all classes with their probabilities.
            Order doesn't matter (will be sorted by framework).

            Example: [
                ("giraffe", 0.99985),
                ("cattle", 0.00003),
                ("elephant", 0.00002),
                ...
            ]

        Raises:
            RuntimeError: If inference fails

        Example (YOLOv8):
            results = self.model(crop, verbose=False)
            names_dict = results[0].names  # {0: "aardwolf", 1: "elephant", ...}
            probs = results[0].probs.data.tolist()

            classifications = []
            for idx, class_name in names_dict.items():
                confidence = float(probs[idx])
                classifications.append((class_name, confidence))

            return classifications

        Example (Keras):
            import numpy as np
            import cv2

            # Preprocess for Keras
            img_array = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            img_array = cv2.resize(img_array, (384, 384))
            img_array = np.expand_dims(img_array, axis=0)

            # Run prediction
            predictions = self.model.predict(img_array, verbose=0)[0]

            # Map to class names (load from file or model)
            class_names = self._load_class_names()

            classifications = []
            for i, class_name in enumerate(class_names):
                confidence = float(predictions[i])
                classifications.append((class_name, confidence))

            return classifications
        """
        raise NotImplementedError("Implement inference for your model")

    def get_class_names(self) -> dict[str, str]:
        """
        Get mapping of class IDs to class names.

        This mapping is saved to the output JSON and used by the database
        to store species information. IDs must be 1-indexed strings.

        Returns:
            Dictionary mapping class ID (1-indexed string) to class name.

            Example: {
                "1": "aardwolf",
                "2": "african wild cat",
                "3": "baboon",
                ...
            }

        Raises:
            RuntimeError: If class names cannot be extracted

        Example (YOLOv8 - names embedded in model):
            yolo_names = self.model.names  # {0: "aardwolf", 1: "elephant", ...}

            # Convert to 1-indexed strings
            class_names = {}
            for idx, name in yolo_names.items():
                class_id = str(idx + 1)  # 1-indexed
                class_names[class_id] = name

            return class_names

        Example (Keras - load from class_list.yaml):
            import yaml

            class_list_path = self.model_dir / "class_list.yaml"
            if not class_list_path.exists():
                raise RuntimeError(f"class_list.yaml not found: {class_list_path}")

            with open(class_list_path) as f:
                class_list = yaml.safe_load(f)

            # Convert to 1-indexed strings
            class_names = {}
            for idx, name in enumerate(class_list, start=1):
                class_names[str(idx)] = name

            return class_names
        """
        raise NotImplementedError("Implement get_class_names for your model")
