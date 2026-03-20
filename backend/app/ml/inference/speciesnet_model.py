"""
SpeciesNet Classification Model - Batch Processor

SpeciesNet uses a different architecture from other classification models:
- Processes entire detection JSON at once (batch processing)
- Calls external run_md_and_speciesnet script from megadetector package
- Uses country/state geofencing instead of species filtering
- Designed to work as ensemble with MegaDetector

This implementation matches the proven approach from streamlit-AddaxAI.

Created by Claude Code on 2026-01-07
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.base import BoundingBox, ClassificationModel, ClassificationResult

logger = get_logger(__name__)


class SpeciesNetClassificationModel(ClassificationModel):
    """
    SpeciesNet classification model with batch processing.

    Unlike per-detection models, SpeciesNet:
    1. Takes detection JSON as input
    2. Processes all detections in one subprocess call
    3. Outputs enhanced JSON with classifications
    4. Uses Google's run_md_and_speciesnet script
    """

    def __init__(
        self,
        model_dir: Path,
        model_path: Path,
        env_name: str,
        env_manager: EnvironmentManager,
    ):
        """
        Initialize SpeciesNet classification model.

        Args:
            model_dir: Path to SpeciesNet model directory
            model_path: Path to main model file (for compatibility)
            env_name: Environment name from manifest (e.g., "addaxai-base")
            env_manager: Environment manager for accessing Python executable

        Raises:
            RuntimeError: If environment setup fails
        """
        self.model_dir = model_dir
        self.model_path = model_path
        self.env_name = env_name
        self.env_manager = env_manager

        # Get Python path from environment
        try:
            env_full_name = f"env-{env_name}"
            self.python_path = env_manager.get_python(env_full_name)
            logger.info(
                f"SpeciesNet model initialized: {model_dir.name} "
                f"using Python: {self.python_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get Python environment '{env_full_name}': {e}"
            ) from e

        logger.info(f"SpeciesNet model ready: {model_dir.name}")

    def classify_batch(
        self,
        detection_json_path: Path,
        country_code: str,
        state_code: str | None,
        deployment_folder: Path,
        progress_callback: Callable[[str, float, str, float], None] | None = None,
    ) -> None:
        """
        Run SpeciesNet classification on entire detection JSON.

        This method calls megadetector.detection.run_md_and_speciesnet
        which processes all detections and outputs an enhanced JSON file.
        The original detection JSON is replaced with the SpeciesNet output.

        Args:
            detection_json_path: Path to detection_results.json (will be modified in-place)
            country_code: Country code for geofencing (e.g., "USA", "KEN")
            state_code: Optional state code for USA (e.g., "CA", "TX")
            deployment_folder: Base folder containing images
            progress_callback: Optional async callback
                (message, overall_progress, phase, phase_progress)

        Raises:
            RuntimeError: If SpeciesNet subprocess fails
            ValueError: If country_code is missing
        """
        if not country_code:
            raise ValueError(
                "SpeciesNet requires country_code to be set in project settings. "
                "Please configure the country in your project settings."
            )

        if not detection_json_path.exists():
            raise FileNotFoundError(
                f"Detection JSON not found: {detection_json_path}"
            )

        # Create output file path
        output_file = detection_json_path.parent / (
            detection_json_path.stem + "-speciesnet-output.json"
        )

        # Build command for run_md_and_speciesnet
        command = [
            str(self.python_path),
            "-m",
            "megadetector.detection.run_md_and_speciesnet",
            str(deployment_folder),  # source folder
            str(output_file),  # output file
            "--detections_file",
            str(detection_json_path),  # skip detection, use existing
            "--classification_model",
            str(self.model_dir),  # local model directory
            "--loader_workers",
            "1",  # Reduce workers to avoid multiprocessing issues
            "--classifier_batch_size",
            "1",  # Process one image at a time for granular progress
        ]

        # Add country code if specified (skip if None, empty, or "NONE")
        if country_code and country_code.upper() not in ("NONE", ""):
            command.extend(["--country", country_code])

            # Add state for USA if specified (skip if None, empty, or "NONE")
            if (
                country_code.upper() == "USA"
                and state_code
                and state_code.upper() not in ("NONE", "")
            ):
                command.extend(["--admin1_region", state_code])
                logger.info(f"SpeciesNet geofencing: USA/{state_code}")
            else:
                logger.info(f"SpeciesNet geofencing: {country_code}")
        else:
            logger.info("SpeciesNet geofencing: DISABLED (no country specified)")

        # Log command for debugging
        logger.info(f"Running SpeciesNet command: {' '.join(command)}")

        # Initial progress update
        if progress_callback:
            progress_callback(
                "Classification: Starting SpeciesNet...",
                0.0,
                "classification",
                0.0,
            )

        try:
            # SpeciesNet's run_md_and_speciesnet subprocess does not emit
            # incremental progress during inference. Its only tqdm bar
            # tracks enqueueing images (instant), while the real inference
            # happens silently between "Finished waiting for input queue"
            # and "Finished waiting for workers". We parse milestone lines
            # to show status, and report the image count so the frontend
            # can display context.
            import os
            import time

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            output_lines: list[str] = []
            device_detected = False
            image_count = 0
            inference_started = False

            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                output_lines.append(line)
                logger.debug(f"[SpeciesNet] {line}")

                # Parse device info (appears once during init)
                if not device_detected:
                    if "PTDetector using device" in line:
                        raw = line.split(
                            "PTDetector using device"
                        )[-1].strip()
                        device_name = self._format_device_name(raw)
                        device_detected = True
                        if progress_callback:
                            progress_callback(
                                "Initializing classifier...",
                                0.0,
                                "classification",
                                0.0,
                                {"compute_device": device_name},
                            )
                    elif "GPU available" in line:
                        import platform

                        has_gpu = "True" in line
                        if has_gpu:
                            device_name = (
                                "GPU (Apple Silicon)"
                                if platform.system() == "Darwin"
                                else "GPU (NVIDIA)"
                            )
                        else:
                            device_name = "CPU"
                        device_detected = True
                        if progress_callback:
                            progress_callback(
                                "Initializing classifier...",
                                0.0,
                                "classification",
                                0.0,
                                {"compute_device": device_name},
                            )

                # Parse image count from "loaded detection results for N images"
                count_match = re.search(
                    r"loaded detection results for (\d+) images", line
                )
                if count_match:
                    image_count = int(count_match.group(1))
                    if progress_callback:
                        progress_callback(
                            f"Classifying {image_count} images...",
                            0.0,
                            "classification",
                            0.05,
                            {
                                "raw_line": (
                                    f"Classifying {image_count} images..."
                                ),
                                "total": image_count,
                                "unit": "image",
                            },
                        )

                # "Finished waiting for input queue" = images enqueued,
                # inference is running in workers
                if "Finished waiting for input queue" in line:
                    inference_started = True
                    if progress_callback:
                        progress_callback(
                            f"Running inference on {image_count} images...",
                            0.0,
                            "classification",
                            0.5,
                            {
                                "raw_line": (
                                    f"Running inference on "
                                    f"{image_count} images..."
                                ),
                                "total": image_count,
                                "unit": "image",
                            },
                        )

                # "Finished waiting for workers" = inference done
                if "Finished waiting for workers" in line:
                    if progress_callback:
                        progress_callback(
                            f"Classified {image_count} images",
                            0.0,
                            "classification",
                            0.9,
                            {
                                "raw_line": (
                                    f"Classified {image_count} images"
                                ),
                                "current": image_count,
                                "total": image_count,
                                "unit": "image",
                            },
                        )

                # "Processing complete" = fully done
                elapsed_match = re.search(
                    r"Processing complete in ([\d.]+) seconds", line
                )
                if elapsed_match:
                    elapsed = float(elapsed_match.group(1))
                    elapsed_str = (
                        f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                    )
                    rate = (
                        image_count / elapsed if elapsed > 0 else 0
                    )
                    if progress_callback:
                        progress_callback(
                            f"Classified {image_count} images",
                            0.0,
                            "classification",
                            1.0,
                            {
                                "raw_line": (
                                    f"Classified {image_count} images "
                                    f"in {elapsed_str}"
                                ),
                                "current": image_count,
                                "total": image_count,
                                "elapsed": elapsed_str,
                                "rate": rate,
                                "unit": "image",
                            },
                        )

            process.stdout.close()
            return_code = process.wait()
            logger.info(
                f"[SpeciesNet] Subprocess done. return_code={return_code}"
            )

            if return_code != 0:
                logger.error("SpeciesNet subprocess failed. Full output:")
                for out_line in output_lines:
                    logger.error(f"  {out_line}")

                error_msg = (
                    "SpeciesNet classification failed "
                    f"with exit code {return_code}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Replace original JSON with SpeciesNet output
            if not output_file.exists():
                raise RuntimeError(
                    "SpeciesNet output file was not created"
                )

            detection_json_path.unlink()
            output_file.rename(detection_json_path)

            logger.info("SpeciesNet classification completed successfully")

            # Final progress update
            if progress_callback:
                progress_callback(
                    "Classification: SpeciesNet complete",
                    0.0,
                    "classification",
                    1.0,
                )

        except subprocess.SubprocessError as e:
            logger.error(
                f"SpeciesNet subprocess error: {e}", exc_info=True
            )
            raise RuntimeError(
                f"SpeciesNet subprocess failed: {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"SpeciesNet classification error: {e}", exc_info=True
            )
            raise RuntimeError(
                f"SpeciesNet classification failed: {e}"
            ) from e

    @staticmethod
    def _format_device_name(raw: str) -> str:
        """Convert raw device string to user-friendly name."""
        r = raw.lower()
        if "mps" in r:
            return "GPU (Apple Silicon)"
        if "cuda" in r:
            return "GPU (NVIDIA)"
        return "CPU"

    def classify(
        self,
        image_path: Path,
        bbox: BoundingBox,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ClassificationResult:
        """
        Per-detection classification (NOT SUPPORTED for SpeciesNet).

        SpeciesNet only supports batch processing via classify_batch().
        This method exists to satisfy the ClassificationModel interface
        but should never be called.

        Raises:
            NotImplementedError: Always - SpeciesNet doesn't support per-detection
        """
        raise NotImplementedError(
            "SpeciesNet only supports batch processing. "
            "Use classify_batch() instead of classify(). "
            "This should be handled automatically by the pipeline."
        )

    def get_class_names(self) -> dict[str, str]:
        """
        Get class name mapping (not applicable for SpeciesNet).

        SpeciesNet's class names are embedded in its model and output JSON.
        This method is required by the interface but not used for SpeciesNet.

        Returns:
            Empty dict - SpeciesNet handles class names internally
        """
        logger.debug(
            "get_class_names() called on SpeciesNet (not applicable)"
        )
        return {}

    def __enter__(self):
        """Context manager entry - no setup needed for batch processing."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - no cleanup needed for batch processing."""
        return False
