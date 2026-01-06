"""
Custom Inference Script Loader

Dynamically loads and executes model-specific inference.py scripts from HuggingFace repos.

Following DEVELOPERS.md principles:
- Crash early if inference.py missing or invalid
- Explicit error handling
- Type hints everywhere

Created by Claude Code on 2026-01-05
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from app.core.logging_config import get_logger
from app.ml.inference.base import BoundingBox

logger = get_logger(__name__)


class CustomInferenceLoader:
    """
    Loads and manages custom inference scripts for classification models.

    Each model should provide an inference.py script in its HuggingFace repo
    with the following contract:

    Required module variables (injected by AddaxAI):
        MODEL_DIR: Path to model directory
        MODEL_PATH: Path to main model file

    Required functions:
        check_gpu() -> bool
            Check GPU availability for this model's framework

        load_model() -> None
            Load model and auxiliary files into module-level variables
            Called once during initialization

        get_crop(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image
            Crop image using model-specific preprocessing
            bbox format: (x_norm, y_norm, width_norm, height_norm)

        get_classification(crop: Image.Image) -> list[tuple[str, float]]
            Run inference on cropped image
            Returns [(class_name, confidence), ...] for ALL classes
    """

    def __init__(self, model_dir: Path, model_path: Path):
        """
        Initialize custom inference loader.

        Args:
            model_dir: Path to model directory
            model_path: Path to main model file

        Raises:
            FileNotFoundError: If inference.py not found in model_dir
            ValueError: If inference.py missing required functions
        """
        self.model_dir = model_dir
        self.model_path = model_path
        self.inference_script_path = model_dir / "inference.py"

        # Validate inference.py exists
        if not self.inference_script_path.exists():
            raise FileNotFoundError(
                f"Custom inference script not found: {self.inference_script_path}\n"
                f"Model developers must provide inference.py in their HuggingFace repo."
            )

        logger.info(f"Loading custom inference script: {self.inference_script_path}")

        # Load the module
        self.module = self._load_module()

        # Inject required variables
        self.module.MODEL_DIR = model_dir
        self.module.MODEL_PATH = model_path

        # Validate required functions exist
        self._validate_interface()

        # Call model loading function
        logger.info("Calling load_model() from custom inference script")
        try:
            self.module.load_model()
        except Exception as e:
            logger.error(f"load_model() failed: {e}", exc_info=True)
            raise RuntimeError(f"Custom inference load_model() failed: {e}") from e

        logger.info("Custom inference script loaded successfully")

    def _load_module(self) -> Any:
        """
        Dynamically load inference.py as a Python module.

        Returns:
            Loaded module object

        Raises:
            ImportError: If module loading fails
        """
        try:
            # Create unique module name to avoid collisions
            module_name = f"custom_inference_{self.model_dir.name}"

            # Load module from file
            spec = importlib.util.spec_from_file_location(
                module_name, self.inference_script_path
            )

            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to create module spec for {self.inference_script_path}")

            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules so imports within inference.py work
            sys.modules[module_name] = module

            # Execute module
            spec.loader.exec_module(module)

            return module

        except Exception as e:
            logger.error(f"Failed to load custom inference script: {e}", exc_info=True)
            raise ImportError(f"Failed to load {self.inference_script_path}: {e}") from e

    def _validate_interface(self) -> None:
        """
        Validate that the loaded module provides all required functions.

        Raises:
            ValueError: If required functions are missing
        """
        required_functions = ["check_gpu", "load_model", "get_crop", "get_classification"]

        missing_functions = []
        for func_name in required_functions:
            if not hasattr(self.module, func_name):
                missing_functions.append(func_name)

        if missing_functions:
            raise ValueError(
                f"Custom inference script missing required functions: {', '.join(missing_functions)}\n"
                f"Required: {', '.join(required_functions)}"
            )

        # Validate functions are callable
        for func_name in required_functions:
            if not callable(getattr(self.module, func_name)):
                raise ValueError(f"Required attribute '{func_name}' is not callable")

        logger.info(f"Validated custom inference interface: all required functions present")

    def check_gpu(self) -> bool:
        """
        Check GPU availability using model's custom check_gpu() function.

        Returns:
            True if GPU available, False otherwise
        """
        try:
            return self.module.check_gpu()
        except Exception as e:
            logger.error(f"check_gpu() failed: {e}", exc_info=True)
            # Default to False if check fails
            return False

    def get_crop(self, image: Image.Image, bbox: BoundingBox) -> Image.Image:
        """
        Crop image using model's custom get_crop() function.

        Args:
            image: PIL Image
            bbox: BoundingBox with normalized coordinates (0.0-1.0)

        Returns:
            Cropped PIL Image ready for classification

        Raises:
            RuntimeError: If get_crop() fails
        """
        try:
            # Convert BoundingBox to tuple format expected by inference.py
            bbox_tuple = (bbox.x, bbox.y, bbox.width, bbox.height)
            return self.module.get_crop(image, bbox_tuple)
        except Exception as e:
            logger.error(f"get_crop() failed: {e}", exc_info=True)
            raise RuntimeError(f"Custom inference get_crop() failed: {e}") from e

    def get_classification(self, crop: Image.Image) -> list[tuple[str, float]]:
        """
        Run classification using model's custom get_classification() function.

        Args:
            crop: Cropped PIL Image

        Returns:
            List of (class_name, confidence) tuples for ALL classes

        Raises:
            RuntimeError: If get_classification() fails
        """
        try:
            return self.module.get_classification(crop)
        except Exception as e:
            logger.error(f"get_classification() failed: {e}", exc_info=True)
            raise RuntimeError(f"Custom inference get_classification() failed: {e}") from e
