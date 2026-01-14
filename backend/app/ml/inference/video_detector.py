"""
Video detection model using MegaDetector's process_video module.

Following DEVELOPERS.md principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere

Uses MegaDetector's built-in process_video module (matches streamlit-AddaxAI exactly).

Created by Claude Code on 2026-01-07
"""

import asyncio
import re
import subprocess
from pathlib import Path
from typing import Callable

from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager

logger = get_logger(__name__)


class VideoDetectionModel:
    """
    MegaDetector video detection wrapper.

    Uses megadetector.detection.process_video module which:
    - Extracts frames automatically (time-based sampling)
    - Runs detection on frames
    - Outputs JSON with frame_rate and frames_processed
    - Handles everything internally (no manual frame extraction needed)
    """

    def __init__(self, model_path: Path, env_manager: EnvironmentManager):
        """
        Initialize video detection model.

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
            logger.info(f"VideoDetectionModel using Python: {self.python_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to get Python environment: {e}") from e

    async def detect_videos_to_json(
        self,
        video_folder: Path,
        output_json: Path,
        fps: float,
        confidence_threshold: float,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Path:
        """
        Run MegaDetector on videos using process_video module.

        Calls megadetector.detection.process_video which handles frame extraction
        and detection internally. Outputs JSON in correct format with frame_rate
        and frames_processed fields.

        Args:
            video_folder: Folder containing video files
            output_json: Path to output JSON file
            fps: Frames per second to extract (converted to time_sample)
            confidence_threshold: Minimum confidence for detections
            progress_callback: Optional callback(message, progress)

        Returns:
            Path to output JSON file

        Raises:
            RuntimeError: If video detection fails
        """
        # Convert FPS to time_sample parameter
        # fps=2.0 → extract every 0.5 seconds → time_sample=0.5
        time_sample = 1.0 / fps

        logger.info(
            f"Running video detection on {video_folder} at {fps} FPS "
            f"(time_sample={time_sample})"
        )

        command = [
            str(self.python_path),
            "-m",
            "megadetector.detection.process_video",
            str(self.model_path),
            str(video_folder),
            "--output_json_file",
            str(output_json),
            "--recursive",
            "--time_sample",
            str(time_sample),
            "--json_confidence_threshold",
            str(confidence_threshold),
        ]

        logger.info(f"Running command: {' '.join(command)}")

        if progress_callback:
            await progress_callback("Starting video detection...", 0.0)

        try:
            # Run subprocess with progress streaming
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Stream output and parse progress
            last_progress = 0.0
            for line in process.stdout:
                line = line.strip()

                # Log output
                logger.debug(f"[VideoDetector] {line}")

                # Parse progress from tqdm output
                # Look for patterns like: "45/100" or "Processing video 5/10"
                progress_match = re.search(r"(\d+)/(\d+)", line)
                if progress_match and progress_callback:
                    current, total = map(int, progress_match.groups())
                    phase_progress = current / total

                    # Only update if progress changed significantly
                    if phase_progress - last_progress >= 0.01:
                        # Parse full tqdm metrics from line
                        metrics = self._parse_tqdm_metrics(line)

                        # Debug: Log what we parsed
                        if metrics:
                            logger.info(f"[VideoDetector] Parsed metrics: {metrics}")
                        else:
                            logger.info(f"[VideoDetector] No metrics parsed from: {line}")

                        # Send raw line and metrics
                        try:
                            await progress_callback(
                                line if metrics else f"Processing video {current}/{total}",
                                phase_progress,
                                metrics
                            )
                        except TypeError:
                            # Fallback for callbacks that don't accept metrics
                            await progress_callback(
                                f"Processing video {current}/{total}",
                                phase_progress,
                            )
                        last_progress = phase_progress

            process.stdout.close()
            return_code = process.wait()

            # Send final 100% update
            if progress_callback and last_progress < 1.0:
                await progress_callback("Video detection complete", 1.0)

            if return_code != 0:
                error_msg = f"Video detection failed with exit code {return_code}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if not output_json.exists():
                raise RuntimeError("Output JSON was not created")

            logger.info(f"Video detection complete: {output_json}")

            return output_json

        except subprocess.SubprocessError as e:
            logger.error(f"Video detection subprocess error: {e}", exc_info=True)
            raise RuntimeError(f"Video detection failed: {e}") from e
        except Exception as e:
            logger.error(f"Video detection error: {e}", exc_info=True)
            raise RuntimeError(f"Video detection failed: {e}") from e

    def _parse_tqdm_metrics(self, line: str) -> dict | None:
        """
        Parse full tqdm metrics from output line.

        Similar to MegaDetector._parse_tqdm_metrics.
        """
        try:
            metrics = {"raw_line": line}

            # Extract current/total
            progress_match = re.search(r"(\d+)/(\d+)", line)
            if progress_match:
                metrics["current"] = int(progress_match.group(1))
                metrics["total"] = int(progress_match.group(2))

            # Extract rate and unit
            # Handle both formats: "2.3it/s" (rate) and "5.67s/it" (time per item)
            rate_match = re.search(r"(\d+\.?\d*)([\w]+)/s", line)
            if rate_match:
                metrics["rate"] = float(rate_match.group(1))
                metrics["unit"] = rate_match.group(2)
            else:
                # Try inverse format: "5.67s/it" -> convert to rate
                inverse_match = re.search(r"(\d+\.?\d*)s/([\w]+)", line)
                if inverse_match:
                    time_per_item = float(inverse_match.group(1))
                    if time_per_item > 0:
                        metrics["rate"] = 1.0 / time_per_item  # Convert to items/s
                        metrics["unit"] = inverse_match.group(2)

            # Extract elapsed time
            time_match = re.search(r"\[(\d{2}:\d{2}(?::\d{2})?)<", line)
            if time_match:
                metrics["elapsed"] = time_match.group(1)

            # Extract remaining time
            remaining_match = re.search(r"<(\d{2}:\d{2}(?::\d{2})?)", line)
            if remaining_match:
                metrics["remaining"] = remaining_match.group(1)

            if "current" in metrics and "total" in metrics:
                return metrics

        except (ValueError, IndexError, AttributeError):
            pass

        return None
