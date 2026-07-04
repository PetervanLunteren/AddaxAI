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
from collections.abc import Callable
from pathlib import Path

from app.core.job_cancellation import (
    JobCancelledError,
    is_cancel_requested,
    track_subprocess,
)
from app.core.logging_config import get_logger
from app.core.subprocess_group import popen_group
from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.base import DetectionModel
from app.utils.fs_hidden import mkdir_hidden_addaxai
from app.utils.subprocess_env import clean_python_env

logger = get_logger(__name__)


def _tqdm_time_to_seconds(value: str) -> int | None:
    """Parse a tqdm time field ("MM:SS" or "HH:MM:SS") to whole seconds."""
    parts = value.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def _seconds_to_tqdm_time(seconds: float) -> str:
    """Format seconds back into tqdm's style: "MM:SS", or "H:MM:SS" past an hour."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class MegaDetectorV1000(DetectionModel):
    """
    MegaDetector v1000 implementation.

    Uses official megadetector Python package via subprocess in isolated environment.
    Command matches streamlit-AddaxAI exactly:

    python -m megadetector.detection.run_detector_batch \\
        --recursive \\
        --output_relative_filenames \\
        --include_image_size \\
        --include_exif_tags "datetimeoriginal,gpsinfo" \\
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

    @staticmethod
    def _format_device_name(raw: str) -> str:
        """Convert raw device string to user-friendly name."""
        r = raw.lower()
        if "mps" in r:
            return "GPU (Apple Silicon)"
        if "cuda" in r:
            return "GPU (NVIDIA)"
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

            # Extract elapsed time: "00:52", "01:23:45", or "1:02:49" (single-digit hours)
            # Look for pattern before "<" (elapsed comes first in tqdm)
            time_match = re.search(r"\[(\d{1,2}:\d{2}(?::\d{2})?)<", line)
            if time_match:
                metrics["elapsed"] = time_match.group(1)

            # Extract remaining time: after "<"
            remaining_match = re.search(r"<(\d{1,2}:\d{2}(?::\d{2})?)", line)
            if remaining_match:
                metrics["remaining"] = remaining_match.group(1)

            # Replace tqdm's remaining (a short rolling-window estimate that
            # swings 6h<->8h on heterogeneous batches) with a smooth ETA from
            # the overall average rate so far: it drifts gradually instead of
            # bouncing. Stateless; uses the current/total/elapsed already parsed.
            current = metrics.get("current")
            total = metrics.get("total")
            elapsed = metrics.get("elapsed")
            if current and total and elapsed and 0 < current <= total:
                elapsed_s = _tqdm_time_to_seconds(elapsed)
                if elapsed_s:
                    remaining_s = elapsed_s * (total - current) / current
                    metrics["remaining"] = _seconds_to_tqdm_time(remaining_s)

            # Only return if we got meaningful data
            if "current" in metrics and "total" in metrics:
                return metrics

        except (ValueError, IndexError, AttributeError):
            pass

        return None


    def detect_to_json(
        self,
        image_paths: list[Path],
        deployment_folder: Path,
        confidence_threshold: float,
        batch_size: int | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
        output_path: Path | None = None,
        job_id: str | None = None,
    ) -> Path:
        """
        Run MegaDetector and save results directly to JSON file (for JSON-based pipeline).

        This method saves the MegaDetector output JSON to the deployment artifacts folder
        instead of parsing it into DetectionResult objects.

        Args:
            image_paths: List of absolute paths to image files
            deployment_folder: Path to deployment folder (will create .addaxai subfolder)
            confidence_threshold: Minimum confidence for detections (typically 0.1)
            batch_size: Number of images processed in parallel. None means let
                MegaDetector use its own default (1). A non-None integer is
                the user's Custom override from the project settings.
            progress_callback: Optional callback(message, progress)
            output_path: Optional explicit output path. If provided, results are written
                here instead of the default .addaxai/detection_results.json.

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
            f"with threshold {confidence_threshold}, batch_size {batch_size}"
        )

        if progress_callback:
            progress_callback("Preparing MegaDetector...", 0.0)

        try:
            # Determine output file location
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_file = output_path
            else:
                artifacts_folder = deployment_folder / ".addaxai"
                mkdir_hidden_addaxai(artifacts_folder)
                output_file = artifacts_folder / "detection_results.json"

            # Create temporary directory for working files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                temp_output = temp_path / "temp_detection_results.json"

                # Write file list JSON so MegaDetector processes only these
                # files (avoids its own recursive scan picking up .addaxai frames)
                file_list_json = temp_path / "image_file_list.json"
                with open(file_list_json, "w") as f:
                    json.dump([str(p) for p in image_paths], f)

                # Build command — pass file list instead of folder
                cmd = [
                    str(self.python_path),
                    "-m",
                    "megadetector.detection.run_detector_batch",
                    "--include_image_size",
                    "--include_exif_tags",
                    "datetimeoriginal,gpsinfo",
                    "--threshold",
                    str(confidence_threshold),
                    str(self.model_path),
                    str(file_list_json),
                    str(temp_output),
                ]

                # Only override batch size when the user explicitly set a
                # Custom value. None = let MegaDetector use its own default.
                if batch_size is not None:
                    cmd.insert(-3, "--batch_size")
                    cmd.insert(-3, str(batch_size))

                logger.info(f"Running command: {' '.join(cmd)}")

                if progress_callback:
                    progress_callback(f"Running detection on {len(image_paths)} images...", 0.1)

                # Execute MegaDetector with streaming output in its own
                # process group so cancel can take the whole tree down.
                process = popen_group(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=clean_python_env(),
                )

                # Monitor progress from stdout
                with track_subprocess(job_id, process):
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            # Print TQDM progress directly to console for debugging
                            print(f"[MEGADETECTOR] {line}", flush=True)
                            logger.info(f"MegaDetector: {line}")

                            # Parse device from PTDetector output (appears once during init)
                            if "PTDetector using device" in line:
                                raw = line.split("PTDetector using device")[-1].strip()
                                device_name = self._format_device_name(raw)
                                logger.info(
                                    f"MegaDetector device: {device_name} (raw: {raw})"
                                )
                                if progress_callback:
                                    try:
                                        progress_callback(
                                            "Initializing detector...",
                                            0.0,
                                            {"compute_device": device_name},
                                        )
                                    except TypeError:
                                        pass

                            # Parse tqdm progress and metrics if callback provided
                            if progress_callback and ("Processing image" in line or "%" in line):
                                progress = self._parse_progress_line(line)
                                metrics = self._parse_tqdm_metrics(line)

                                # When batch_size > 1, MegaDetector's tqdm
                                # iterates over batches, not images. Remap
                                # to image-level counts so the UI shows the
                                # correct total and throughput.
                                if (
                                    batch_size is not None
                                    and batch_size > 1
                                    and metrics
                                ):
                                    total_batches = metrics.get("total", 0)
                                    if total_batches > 0:
                                        fraction = (
                                            metrics.get("current", 0)
                                            / total_batches
                                        )
                                        metrics["total"] = len(image_paths)
                                        metrics["current"] = round(
                                            fraction * len(image_paths)
                                        )
                                    if "rate" in metrics:
                                        metrics["rate"] *= batch_size

                                if progress is not None:
                                    # Try to send metrics if callback accepts
                                    # 3 params, else fallback to 2
                                    try:
                                        progress_callback(
                                            line if metrics else line[:80],
                                            0.1 + progress * 0.8,
                                            metrics,
                                        )
                                    except TypeError:
                                        # Callback only accepts 2 params (backward compatibility)
                                        progress_callback(line[:80], 0.1 + progress * 0.8)

                    process.stdout.close()
                    process.wait()

                if job_id and is_cancel_requested(job_id):
                    raise JobCancelledError()

                if process.returncode != 0:
                    raise RuntimeError(f"MegaDetector failed with return code {process.returncode}")

                # Verify temp output exists
                if not temp_output.exists():
                    raise RuntimeError(f"Detection output file not found: {temp_output}")

                # Post-process: convert absolute paths in output to relative paths
                # MegaDetector outputs absolute paths when given a file list,
                # but downstream consumers expect relative paths
                with open(temp_output) as f:
                    md_results = json.load(f)

                for img in md_results.get("images", []):
                    abs_path = Path(img["file"])
                    try:
                        img["file"] = str(abs_path.relative_to(deployment_folder))
                    except ValueError:
                        # Path not relative to deployment folder — keep as-is
                        pass

                with open(output_file, "w") as f:
                    json.dump(md_results, f, indent=2)

                logger.info(f"Detection complete: Results saved to {output_file}")

                if progress_callback:
                    progress_callback("Detection complete", 1.0)

                return output_file

        except JobCancelledError:
            raise
        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise RuntimeError(f"MegaDetector execution failed: {e}") from e
