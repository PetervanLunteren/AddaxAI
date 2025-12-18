"""
MegaDetector runner using official CLI.

Following DEVELOPERS.md principles:
- Crash early if setup fails
- Explicit error messages
- Type hints everywhere

Uses the official megadetector package CLI for inference.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)


class MegaDetectorRunner:
    """
    Runs MegaDetector using official package CLI.

    Handles environment setup, model execution, and result parsing.
    Provides progress callbacks for real-time updates.
    """

    def __init__(self, env_manager: EnvironmentManager):
        """
        Initialize MegaDetector runner.

        Args:
            env_manager: EnvironmentManager for creating/managing micromamba envs
        """
        self.env_manager = env_manager

    def run_detection(
        self,
        manifest: ModelManifest,
        image_paths: list[Path],
        confidence_threshold: float = 0.1,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """
        Run MegaDetector on a list of images.

        Args:
            manifest: Model manifest with environment specs
            image_paths: List of absolute paths to images
            confidence_threshold: Minimum confidence threshold (default: 0.1)
            progress_callback: Optional callback(message, progress) for updates

        Returns:
            Detection results in MegaDetector JSON format:
            {
                "images": [
                    {
                        "file": "/path/to/image.jpg",
                        "detections": [
                            {
                                "category": "1",  # 1=animal, 2=person, 3=vehicle
                                "conf": 0.95,
                                "bbox": [x, y, width, height]  # normalized 0-1
                            }
                        ],
                        "max_detection_conf": 0.95
                    }
                ]
            }

        Raises:
            RuntimeError: If detection fails
        """
        if not image_paths:
            logger.warning("No images provided for detection")
            return {"images": []}

        # Setup environment
        if progress_callback:
            progress_callback("Setting up environment...", 0.05)

        env_path = self.env_manager.get_or_create_env(manifest)
        python_path = self.env_manager.get_python(env_path.name)

        logger.info(f"Using environment: {env_path.name}")

        # Get model file path
        from app.ml.model_storage import ModelStorage

        model_storage = ModelStorage()
        model_file = model_storage.get_model_file(manifest)
        logger.info(f"Using model file: {model_file}")

        # Create temp file list
        if progress_callback:
            progress_callback("Preparing image list...", 0.10)

        file_list_path = self._create_file_list(image_paths)

        try:
            # Run MegaDetector CLI
            if progress_callback:
                progress_callback("Running MegaDetector...", 0.15)

            results = self._run_megadetector_cli(
                python_path=python_path,
                file_list_path=file_list_path,
                confidence_threshold=confidence_threshold,
                model_file=model_file,
                progress_callback=progress_callback,
            )

            if progress_callback:
                progress_callback("Detection complete", 1.0)

            return results

        finally:
            # Clean up temp file list
            if file_list_path.exists():
                file_list_path.unlink()

    def _create_file_list(self, image_paths: list[Path]) -> Path:
        """
        Create temporary file list for MegaDetector CLI.

        Args:
            image_paths: List of image paths

        Returns:
            Path to temporary file list
        """
        # Create temp file in system temp directory
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".txt", prefix="md_files_")
        file_list_path = Path(path)

        # Write image paths (one per line)
        with open(fd, "w") as f:
            for image_path in image_paths:
                f.write(f"{image_path}\n")

        logger.debug(f"Created file list: {file_list_path} ({len(image_paths)} images)")
        return file_list_path

    def _run_megadetector_cli(
        self,
        python_path: Path,
        file_list_path: Path,
        confidence_threshold: float,
        model_file: Path,
        progress_callback: Callable[[str, float], None] | None,
    ) -> dict[str, Any]:
        """
        Run MegaDetector CLI and parse results.

        Args:
            python_path: Path to Python executable in environment
            file_list_path: Path to file list
            confidence_threshold: Minimum confidence threshold
            model_file: Path to model weights file (e.g., "md_v5a.0.0.pt")
            progress_callback: Optional progress callback

        Returns:
            Detection results dict

        Raises:
            RuntimeError: If CLI execution fails
        """
        # Create temp output file
        import tempfile

        fd, output_path = tempfile.mkstemp(suffix=".json", prefix="md_results_")
        output_file = Path(output_path)

        try:
            # Build command
            # Using MegaDetector's run_detector_batch module
            cmd = [
                str(python_path),
                "-m",
                "megadetector.detection.run_detector_batch",
                str(model_file),  # Path to model weights file
                str(file_list_path),
                str(output_file),
                "--threshold",
                str(confidence_threshold),
                "--quiet",  # Reduce output noise
            ]

            logger.info(f"Running MegaDetector: {' '.join(cmd)}")

            # Run command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Monitor progress from stderr
            total_images = sum(1 for _ in open(file_list_path))
            processed = 0

            for line in iter(process.stderr.readline, ""):
                if not line:
                    break

                # Parse progress from MegaDetector output
                # MegaDetector prints: "Processing image X/Y"
                if "Processing image" in line and progress_callback:
                    try:
                        # Extract progress
                        parts = line.split()
                        if "/" in parts[-1]:
                            current, total = parts[-1].split("/")
                            processed = int(current)
                            # Map to 15-95% range (setup was 0-15%, final is 95-100%)
                            progress = 0.15 + (processed / total_images) * 0.80
                            progress_callback(
                                f"Processing image {processed}/{total_images}...",
                                progress,
                            )
                    except (ValueError, IndexError):
                        pass  # Ignore parsing errors

            # Wait for completion
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(
                    f"MegaDetector CLI failed (exit code {process.returncode}):\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stdout: {stdout}\n"
                    f"Stderr: {stderr}"
                )

            # Read results
            with open(output_file) as f:
                results = json.load(f)

            logger.info(f"MegaDetector processed {len(results.get('images', []))} images")
            return results

        finally:
            # Clean up output file
            if output_file.exists():
                output_file.unlink()

    def validate_environment(self, manifest: ModelManifest) -> bool:
        """
        Validate that MegaDetector environment is set up correctly.

        Args:
            manifest: Model manifest

        Returns:
            True if valid, False otherwise
        """
        try:
            env_path = self.env_manager.get_or_create_env(manifest)
            python_path = self.env_manager.get_python(env_path.name)

            # Test that megadetector package is available
            cmd = [
                str(python_path),
                "-c",
                "import megadetector; print(megadetector.__version__)",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                logger.info(f"MegaDetector environment valid: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"MegaDetector validation failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Environment validation error: {e}")
            return False
