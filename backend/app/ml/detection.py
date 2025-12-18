"""
MegaDetector runner for object detection in camera trap images.

Following DEVELOPERS.md principles:
- Crash early if setup fails
- Explicit error messages
- Type hints everywhere
"""

import json
import subprocess
from pathlib import Path
from typing import Callable

from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.ml.model_storage import ModelStorage
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)


class MegaDetectorRunner:
    """
    Runs MegaDetector inference on images using isolated conda environments.

    Uses subprocess to run detection in the model's environment to avoid
    dependency conflicts between different models.
    """

    def __init__(self, env_manager: EnvironmentManager):
        """
        Initialize MegaDetector runner.

        Args:
            env_manager: Environment manager for accessing model environments
        """
        self.env_manager = env_manager
        self.model_storage = ModelStorage()

    def run_detection(
        self,
        manifest: ModelManifest,
        image_paths: list[Path],
        confidence_threshold: float = 0.1,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict:
        """
        Run MegaDetector on a list of images.

        Args:
            manifest: Model manifest
            image_paths: List of paths to image files
            confidence_threshold: Confidence threshold for detections
            progress_callback: Optional callback(message, progress) for updates

        Returns:
            Detection results in MegaDetector JSON format

        Raises:
            RuntimeError: If detection fails
        """
        try:
            if progress_callback:
                progress_callback("Preparing MegaDetector...", 0.15)

            # Get model file path
            model_file = self.model_storage.get_model_file(manifest)
            logger.info(f"Using model: {model_file}")

            # Get Python executable from environment
            env_name = f"env-{manifest.env}"
            python_path = self.env_manager.get_python(env_name)
            logger.info(f"Using Python from: {python_path}")

            if progress_callback:
                progress_callback(f"Running detection on {len(image_paths)} images...", 0.20)

            # Create temporary file list
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                for img_path in image_paths:
                    f.write(str(img_path) + "\n")
                image_list_file = Path(f.name)

            # Create output file
            output_file = image_list_file.parent / "detection_results.json"

            try:
                # Run MegaDetector using the environment's Python
                # The megadetector package provides run_detector module
                cmd = [
                    str(python_path),
                    "-m",
                    "megadetector.detection.run_detector",
                    str(model_file),
                    str(image_list_file),
                    str(output_file),
                    f"--threshold",
                    str(confidence_threshold),
                    "--quiet",
                ]

                logger.info(f"Running command: {' '.join(cmd)}")

                # Run detection
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Monitor progress (basic - just wait for completion)
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    logger.error(f"Detection stdout: {stdout}")
                    logger.error(f"Detection stderr: {stderr}")
                    raise RuntimeError(
                        f"MegaDetector failed with return code {process.returncode}:\n{stderr}"
                    )

                # Read results
                if not output_file.exists():
                    raise RuntimeError(f"Detection output file not found: {output_file}")

                with open(output_file) as f:
                    results = json.load(f)

                logger.info(f"Detection complete: {len(results.get('images', []))} images processed")

                if progress_callback:
                    progress_callback("Detection complete", 0.90)

                return results

            finally:
                # Clean up temporary files
                if image_list_file.exists():
                    image_list_file.unlink()
                if output_file.exists():
                    output_file.unlink()

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to run detection: {e}") from e
