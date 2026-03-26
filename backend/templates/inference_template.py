"""
Template for custom classification model inference

This template provides a framework-agnostic interface for implementing custom
classification models in AddaxAI. Works with any ML framework (PyTorch, Keras,
TensorFlow, JAX, etc.) as long as it follows the class-based API below.

USAGE:
See for all these steps examples in ~/AddaxAI/models/cls/
1. Copy this file to your model's directory as 'inference.py'
2. Implement the ModelInference class with your model-specific logic
3. Add your model files (.pt, .h5, .yaml, etc.) to the same directory
4. Add a valid manifest.json for your model
5. Add taxonomy.csv with model class names and their taxonomic info
6. AddaxAI will automatically discover and use your implementation

Created for AddaxAI (https://github.com/PetervanLunteren/AddaxAI)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class ModelInference:
    """
    Custom inference implementation for your classification model.

    This class is instantiated once per worker process. The model is loaded
    in load_model() and reused for all subsequent classification requests.

    Lifecycle:
    1. __init__() - Store paths
    2. load_model() - Load model into memory (called once)
    3. get_classification() - Run inference (called many times)
    """

    def __init__(self, model_dir: Path, model_path: Path):
        """
        Initialize with model paths. Model loading happens in load_model().
        You can leave this as-is unless you need extra initialization.

        Args:
            model_dir: Directory containing model files
            model_path: Path to model file
        """

        # Leave this as-is
        self.model_dir = model_dir
        self.model_path = model_path
        self.model = None

        # Optionally add any other initialization here (load config files, etc.)

    def check_gpu(self) -> bool:
        """
        Check if GPU or MPS is available.

        Returns:
            True if GPU or MPS is available, False otherwise
        """

        raise NotImplementedError("Implement GPU check for your framework")

    def load_model(self) -> None:
        """
        Load your model into memory. Called once at worker startup.

        This is where expensive operations happen (model file loading,
        weight initialization, etc.). The loaded model is stored in
        self.model and reused for all subsequent classifications.
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
            bbox: Normalized MD bounding box (x, y, width, height) in range [0.0, 1.0]
                  Format: (x_min, y_min, box_width, box_height)

        Returns:
            Cropped and preprocessed PIL Image ready for classification
        """

        raise NotImplementedError("Implement cropping for your model")

    def get_classification(self, crop: Image.Image) -> list[list[str, float]]:
        """
        Run classification inference on cropped image.

        This method is called for every animal detection. It should return
        probabilities for ALL classes, not just the top prediction.

        Args:
            crop: Preprocessed image from get_crop()

        Returns:
            List of [class_name, confidence] lists for all classes. No sorting needed.
            Example: [["aardwolf", 0.01351], ["giraffe", 0.89985], ...]
        """

        raise NotImplementedError("Implement inference for your model")

    def get_class_names(self) -> dict[str, str]:
        """
        Get mapping of class IDs to class names.

        This mapping is saved to the output JSON and used by the database
        to store species information. IDs must be 1-indexed strings.

        NOTE: taxonomy.csv should not be used here. It's only for UI taxonomy tree display.

        Returns:
            Dictionary mapping class ID (1-indexed string) to class name.

            Example: {
                "1": "aardwolf",
                "2": "african wild cat",
                "3": "baboon",
                ...
            }
        """

        raise NotImplementedError("Implement get_class_names for your model")

    # ------------------------------------------------------------------
    # Optional batch inference methods
    #
    # Implementing get_tensor() + classify_batch() enables batch inference:
    # multiple crops processed in one GPU forward pass instead of one at a
    # time. This can give 5-15x speedup on GPU. If not implemented, the
    # worker falls back to calling get_classification() per crop.
    # ------------------------------------------------------------------

    def get_tensor(self, crop: Image.Image):
        """
        Optional: preprocess a crop into a model-ready numpy array.

        Called once per crop. The worker collects arrays from multiple
        crops, stacks them with numpy.stack(), and passes the batch
        to classify_batch().

        Args:
            crop: Cropped PIL Image from get_crop()

        Returns:
            numpy.ndarray: preprocessed array ready for model input.
            Shape depends on model (e.g., [H, W, C] or [C, H, W]).

        Example (PyTorch model with 224x224 input):
            import numpy as np
            from torchvision import transforms
            tensor = self.preprocess(crop)  # your transforms.Compose
            return tensor.numpy()
        """
        raise NotImplementedError

    def classify_batch(self, batch):
        """
        Optional: run inference on a batch of preprocessed arrays.

        Args:
            batch: numpy.ndarray of stacked arrays from get_tensor().
                   Shape: [batch_size, ...] where ... matches get_tensor().

        Returns:
            List of classification results, one per crop in the batch.
            Each result has the same format as get_classification():
            [[class_name, confidence], ...]

        Example (PyTorch):
            import torch
            tensor = torch.from_numpy(batch).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return [
                [[self.names[j], float(p[j])] for j in range(len(p))]
                for p in probs
            ]
        """
        raise NotImplementedError
