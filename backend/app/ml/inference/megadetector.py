"""
MegaDetector implementation using official megadetector package.

Following DEVELOPERS.md principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere

Uses exact same execution as streamlit-AddaxAI to guarantee matching results.

Created by Claude Code on 2026-01-04
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.base import BoundingBox, DetectionModel, DetectionResult

logger = get_logger(__name__)


class MegaDetectorV1000(DetectionModel):
    """
    MegaDetector v1000 implementation.

    Uses official megadetector Python package via subprocess in isolated environment.
    Command matches streamlit-AddaxAI exactly:

    python -m megadetector.detection.run_detector_batch \\
        --recursive \\
        --output_relative_filenames \\
        --include_image_size \\
        --include_image_timestamp \\
        --include_exif_data \\
        --threshold 0.1 \\
        model.pt folder/ output.json
    """

    # Category mapping from MegaDetector output to internal labels
    CATEGORY_MAP = {
        "1": "animal",
        "2": "person",
        "3": "vehicle",
    }

    def __init__(self, model_path: Path, env_manager: EnvironmentManager):
        """
        Initialize MegaDetector.

        Args:
            model_path: Path to .pt model file
            env_manager: Environment manager for accessing conda environments

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If environment not found
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.env_manager = env_manager

        # Verify environment exists
        try:
            self.python_path = env_manager.get_python("env-addaxai-base")
            logger.info(f"MegaDetector using Python: {self.python_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to get Python environment: {e}") from e

    def detect(
        self,
        image_paths: list[Path],
        confidence_threshold: float,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> list[DetectionResult]:
        """
        Run MegaDetector on images.

        Calls official megadetector package to guarantee exact same results
        as streamlit-AddaxAI.

        Args:
            image_paths: List of absolute paths to image files
            confidence_threshold: Minimum confidence for detections (typically 0.1)
            progress_callback: Optional callback(message, progress)

        Returns:
            List of DetectionResult objects

        Raises:
            RuntimeError: If detection fails
            FileNotFoundError: If image files don't exist
        """
        if not image_paths:
            raise ValueError("No image paths provided")

        # Verify all images exist
        for img_path in image_paths:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

        logger.info(
            f"Running MegaDetector on {len(image_paths)} images "
            f"with threshold {confidence_threshold}"
        )

        if progress_callback:
            progress_callback("Preparing MegaDetector...", 0.0)

        try:
            # Create temporary directory for working files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Determine folder to process (common parent of all images)
                # This is needed because MegaDetector processes folders, not file lists
                folder_path = self._get_common_folder(image_paths)

                output_file = temp_path / "detection_results.json"

                # Build command exactly as streamlit-AddaxAI
                cmd = [
                    str(self.python_path),
                    "-m",
                    "megadetector.detection.run_detector_batch",
                    "--recursive",
                    "--output_relative_filenames",
                    "--include_image_size",
                    "--include_image_timestamp",
                    "--include_exif_data",
                    "--threshold",
                    str(confidence_threshold),
                    str(self.model_path),
                    str(folder_path),
                    str(output_file),
                ]

                logger.info(f"Running command: {' '.join(cmd)}")

                if progress_callback:
                    progress_callback(f"Running detection on {len(image_paths)} images...", 0.1)

                # Execute MegaDetector with streaming output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Monitor progress from stdout
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        # Print TQDM progress directly to console for debugging
                        print(f"[MEGADETECTOR] {line}", flush=True)
                        logger.info(f"MegaDetector: {line}")

                        # Parse device from PTDetector output (appears once during init)
                        if "PTDetector using device" in line and progress_callback:
                            raw = line.split("PTDetector using device")[-1].strip()
                            device_name = self._format_device_name(raw)
                            try:
                                progress_callback("Initializing detector...", 0.0, {"compute_device": device_name})
                            except TypeError:
                                pass

                        # Parse tqdm progress and metrics if callback provided
                        if progress_callback and ("Processing image" in line or "%" in line):
                            progress = self._parse_progress_line(line)
                            metrics = self._parse_tqdm_metrics(line)
                            if progress is not None:
                                # Try to send metrics if callback accepts 3 params, else fallback to 2
                                try:
                                    progress_callback(line if metrics else line[:80], 0.1 + progress * 0.8, metrics)
                                except TypeError:
                                    # Callback only accepts 2 params (backward compatibility)
                                    progress_callback(line[:80], 0.1 + progress * 0.8)

                process.stdout.close()
                process.wait()

                if process.returncode != 0:
                    raise RuntimeError(
                        f"MegaDetector failed with return code {process.returncode}"
                    )

                # Read and parse results
                if not output_file.exists():
                    raise RuntimeError(f"Detection output file not found: {output_file}")

                with open(output_file) as f:
                    raw_results = json.load(f)

                logger.info(
                    f"Detection complete: {len(raw_results.get('images', []))} images processed"
                )

                if progress_callback:
                    progress_callback("Parsing detection results...", 0.95)

                # Convert to DetectionResult objects
                detections = self._parse_results(raw_results, folder_path, image_paths)

                if progress_callback:
                    progress_callback("Detection complete", 1.0)

                logger.info(f"Found {len(detections)} total detections")

                return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise RuntimeError(f"MegaDetector execution failed: {e}") from e

    def _get_common_folder(self, image_paths: list[Path]) -> Path:
        """
        Get common parent folder containing all images.

        Args:
            image_paths: List of image paths

        Returns:
            Common parent directory
        """
        if len(image_paths) == 1:
            return image_paths[0].parent

        # Find common parent
        common = image_paths[0].parent
        for path in image_paths[1:]:
            while not str(path).startswith(str(common)):
                common = common.parent
                if common == common.parent:  # Reached filesystem root
                    raise ValueError("Images have no common parent directory")

        return common

    @staticmethod
    def _format_device_name(raw: str) -> str:
        """Convert raw device string to user-friendly name."""
        r = raw.lower()
        if "mps" in r:
            return "MPS (Apple Silicon)"
        if "cuda" in r:
            return "CUDA (NVIDIA)"
        return "CPU"

    def _parse_progress_line(self, line: str) -> float | None:
        """
        Parse progress percentage from tqdm output line.

        Args:
            line: stdout line from MegaDetector

        Returns:
            Progress as 0.0 - 1.0, or None if couldn't parse
        """
        try:
            # Look for percentage pattern like "45%" or "100%"
            if "%" in line:
                parts = line.split("%")
                if len(parts) >= 1:
                    # Extract number before %
                    num_str = parts[0].split()[-1]
                    percent = float(num_str)
                    return percent / 100.0
        except (ValueError, IndexError):
            pass

        return None

    def _parse_tqdm_metrics(self, line: str) -> dict | None:
        """
        Parse full tqdm metrics from output line.

        Tqdm format examples:
        - "Processing: 45/100 [=====>...] 45% 2.3it/s 00:24<00:30"
        - "100%|██████████| 120/120 [00:52<00:00,  2.29it/s]"
        - "Processing video:  50%|█████     | 10/20 [00:48<00:42,  4.25s/it]"

        Args:
            line: Raw tqdm output line

        Returns:
            Dict with metrics: {current, total, elapsed, remaining, rate, unit, raw_line}
            or None if couldn't parse
        """
        import re

        try:
            metrics = {"raw_line": line}

            # Extract current/total: "45/100" or "120/120"
            progress_match = re.search(r"(\d+)/(\d+)", line)
            if progress_match:
                metrics["current"] = int(progress_match.group(1))
                metrics["total"] = int(progress_match.group(2))

            # Extract rate and unit
            # Handle both formats: "2.3it/s" (rate) and "5.67s/it" (time per item)
            rate_match = re.search(r"(\d+\.?\d*)([\w]+)/s", line)
            if rate_match:
                metrics["rate"] = float(rate_match.group(1))
                metrics["unit"] = rate_match.group(2)  # "it", "images", "video", etc.
            else:
                # Try inverse format: "5.67s/it" -> convert to rate
                inverse_match = re.search(r"(\d+\.?\d*)s/([\w]+)", line)
                if inverse_match:
                    time_per_item = float(inverse_match.group(1))
                    if time_per_item > 0:
                        metrics["rate"] = 1.0 / time_per_item  # Convert to items/s
                        metrics["unit"] = inverse_match.group(2)

            # Extract elapsed time: "00:52" or "01:23:45"
            # Look for pattern before "<" (elapsed comes first in tqdm)
            time_match = re.search(r"\[(\d{2}:\d{2}(?::\d{2})?)<", line)
            if time_match:
                metrics["elapsed"] = time_match.group(1)

            # Extract remaining time: after "<"
            remaining_match = re.search(r"<(\d{2}:\d{2}(?::\d{2})?)", line)
            if remaining_match:
                metrics["remaining"] = remaining_match.group(1)

            # Only return if we got meaningful data
            if "current" in metrics and "total" in metrics:
                return metrics

        except (ValueError, IndexError, AttributeError):
            pass

        return None

    def _parse_results(
        self, raw_results: dict, folder_path: Path, expected_images: list[Path]
    ) -> list[DetectionResult]:
        """
        Parse MegaDetector JSON output into DetectionResult objects.

        Args:
            raw_results: Raw JSON dict from MegaDetector
            folder_path: Base folder path
            expected_images: Images we expect to find in results

        Returns:
            List of DetectionResult objects
        """
        detections: list[DetectionResult] = []

        # Build set of expected image paths for validation
        expected_set = {str(p) for p in expected_images}

        for image_result in raw_results.get("images", []):
            # Construct absolute path from relative filename
            relative_file = image_result["file"]
            absolute_path = (folder_path / relative_file).resolve()

            # Verify this is an image we expected
            if str(absolute_path) not in expected_set:
                logger.warning(f"Unexpected image in results: {absolute_path}")
                continue

            # Parse detections for this image
            for det in image_result.get("detections", []):
                category_num = str(det["category"])
                category = self.CATEGORY_MAP.get(category_num, "animal")
                confidence = float(det["conf"])
                bbox_list = det["bbox"]  # [x, y, width, height]

                # Create BoundingBox (validation happens in __post_init__)
                try:
                    bbox = BoundingBox(
                        x=float(bbox_list[0]),
                        y=float(bbox_list[1]),
                        width=float(bbox_list[2]),
                        height=float(bbox_list[3]),
                    )

                    detection = DetectionResult(
                        file_path=absolute_path,
                        category=category,
                        confidence=confidence,
                        bbox=bbox,
                    )

                    detections.append(detection)

                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Skipping invalid detection in {absolute_path}: {e}"
                    )
                    continue

        return detections

    def detect_to_json(
        self,
        image_paths: list[Path],
        deployment_folder: Path,
        confidence_threshold: float,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Path:
        """
        Run MegaDetector and save results directly to JSON file (for JSON-based pipeline).

        This method saves the MegaDetector output JSON to the deployment artifacts folder
        instead of parsing it into DetectionResult objects.

        Args:
            image_paths: List of absolute paths to image files
            deployment_folder: Path to deployment folder (will create .addaxai subfolder)
            confidence_threshold: Minimum confidence for detections (typically 0.1)
            progress_callback: Optional callback(message, progress)

        Returns:
            Path to saved detection_results.json file

        Raises:
            RuntimeError: If detection fails
            FileNotFoundError: If image files don't exist
        """
        if not image_paths:
            raise ValueError("No image paths provided")

        # Verify all images exist
        for img_path in image_paths:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

        logger.info(
            f"Running MegaDetector on {len(image_paths)} images "
            f"with threshold {confidence_threshold}"
        )

        if progress_callback:
            progress_callback("Preparing MegaDetector...", 0.0)

        try:
            # Create artifacts folder
            artifacts_folder = deployment_folder / ".addaxai"
            artifacts_folder.mkdir(parents=True, exist_ok=True)
            output_file = artifacts_folder / "detection_results.json"

            # Create temporary directory for working files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Determine folder to process (common parent of all images)
                folder_path = self._get_common_folder(image_paths)

                temp_output = temp_path / "temp_detection_results.json"

                # Build command exactly as before
                cmd = [
                    str(self.python_path),
                    "-m",
                    "megadetector.detection.run_detector_batch",
                    "--recursive",
                    "--output_relative_filenames",
                    "--include_image_size",
                    "--include_image_timestamp",
                    "--include_exif_data",
                    "--threshold",
                    str(confidence_threshold),
                    str(self.model_path),
                    str(folder_path),
                    str(temp_output),
                ]

                logger.info(f"Running command: {' '.join(cmd)}")

                if progress_callback:
                    progress_callback(f"Running detection on {len(image_paths)} images...", 0.1)

                # Execute MegaDetector with streaming output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Monitor progress from stdout
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        # Print TQDM progress directly to console for debugging
                        print(f"[MEGADETECTOR] {line}", flush=True)
                        logger.info(f"MegaDetector: {line}")

                        # Parse device from PTDetector output (appears once during init)
                        if "PTDetector using device" in line and progress_callback:
                            raw = line.split("PTDetector using device")[-1].strip()
                            device_name = self._format_device_name(raw)
                            try:
                                progress_callback("Initializing detector...", 0.0, {"compute_device": device_name})
                            except TypeError:
                                pass

                        # Parse tqdm progress and metrics if callback provided
                        if progress_callback and ("Processing image" in line or "%" in line):
                            progress = self._parse_progress_line(line)
                            metrics = self._parse_tqdm_metrics(line)
                            if progress is not None:
                                # Try to send metrics if callback accepts 3 params, else fallback to 2
                                try:
                                    progress_callback(line if metrics else line[:80], 0.1 + progress * 0.8, metrics)
                                except TypeError:
                                    # Callback only accepts 2 params (backward compatibility)
                                    progress_callback(line[:80], 0.1 + progress * 0.8)

                process.stdout.close()
                process.wait()

                if process.returncode != 0:
                    raise RuntimeError(
                        f"MegaDetector failed with return code {process.returncode}"
                    )

                # Verify temp output exists
                if not temp_output.exists():
                    raise RuntimeError(f"Detection output file not found: {temp_output}")

                # Copy to permanent artifacts location
                import shutil
                shutil.copy2(temp_output, output_file)

                logger.info(
                    f"Detection complete: Results saved to {output_file}"
                )

                if progress_callback:
                    progress_callback("Detection complete", 1.0)

                return output_file

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise RuntimeError(f"MegaDetector execution failed: {e}") from e
