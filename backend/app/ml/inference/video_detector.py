"""
Video detection model using MegaDetector's process_video module.

Following DEVELOPERS.md principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere

Uses MegaDetector's built-in process_video module (matches streamlit-AddaxAI exactly).

Created by Claude Code on 2026-01-07
"""

import re
import subprocess
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
from app.ml.gpu_guard import cuda_guard_overrides
from app.utils.subprocess_env import clean_python_env

logger = get_logger(__name__)

# Windows exit code 0xC0000005 (access violation). OpenCV's FFmpeg
# backend dies with it on videos whose pixel format changes mid-stream
# (Bushnell MJPEG AVIs: frame 0 is yuvj422p, the rest yuvj420p). See
# "Mixed pixel format videos" in DEVELOPERS.md.
_WINDOWS_ACCESS_VIOLATION = 3221225477


def _build_process_video_cmd(
    *,
    python_path: Path,
    model_path: Path,
    video_folder: Path,
    output_json: Path,
    time_sample: float,
    confidence_threshold: float,
    image_size: int | None,
    augment: bool,
) -> list[str]:
    """Assemble the ``process_video`` command line.

    Pure and side-effect free so the flag logic is unit-testable without
    spawning the subprocess. Optional inference flags (image size, augment)
    are appended only when set, mirroring the image detector; process_video
    accepts them as ``--image_size N`` and ``--augment`` (store_true).
    """
    command = [
        str(python_path),
        "-m",
        "megadetector.detection.process_video",
        str(model_path),
        str(video_folder),
        "--output_json_file",
        str(output_json),
        "--recursive",
        "--time_sample",
        str(time_sample),
        "--json_confidence_threshold",
        str(confidence_threshold),
    ]
    if image_size is not None:
        command += ["--image_size", str(image_size)]
    if augment:
        command.append("--augment")
    return command


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

    def detect_videos_to_json(
        self,
        video_folder: Path,
        output_json: Path,
        fps: float,
        confidence_threshold: float,
        image_size: int | None = None,
        augment: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
        job_id: str | None = None,
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
            image_size: Override the detector's long-edge resize size. None
                means use MegaDetector's model-native default.
            augment: Run detection with image augmentation (slower, may add
                false positives). From the project's detection_augment setting.
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

        command = _build_process_video_cmd(
            python_path=self.python_path,
            model_path=self.model_path,
            video_folder=video_folder,
            output_json=output_json,
            time_sample=time_sample,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            augment=augment,
        )

        logger.info(f"Running command: {' '.join(command)}")

        if progress_callback:
            progress_callback("Starting video detection...", 0.0)

        try:
            base_env = clean_python_env(**cuda_guard_overrides(self.env_manager))
            return_code = self._stream_process(
                command, base_env, progress_callback, job_id
            )

            cancelled = is_cancel_requested(job_id) if job_id else False
            if return_code == _WINDOWS_ACCESS_VIOLATION and not cancelled:
                # OpenCV's FFmpeg backend takes the whole subprocess down
                # on a mixed-pixel-format video. Deprioritising FFmpeg
                # makes cv2 pick MSMF, which decodes those files (with a
                # slight colour-range shift on the detector's input, the
                # least sensitive consumer). One retry covers the whole
                # folder; deployments are single-camera, so a folder that
                # trips this is all such files anyway.
                logger.warning(
                    "Video detection died with an access violation, likely "
                    "OpenCV's FFmpeg backend on a mixed-pixel-format video "
                    "(see 'Mixed pixel format videos' in DEVELOPERS.md). "
                    "Retrying once with OPENCV_VIDEOIO_PRIORITY_FFMPEG=0."
                )
                retry_env = dict(base_env)
                retry_env["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"
                return_code = self._stream_process(
                    command, retry_env, progress_callback, job_id
                )
                cancelled = is_cancel_requested(job_id) if job_id else False

            # If we were cancelled mid-stream, the process was killed and
            # returned non-zero; surface that as a cancel rather than an
            # opaque RuntimeError.
            if cancelled:
                raise JobCancelledError()

            # Send final 100% update
            if progress_callback:
                progress_callback("Video detection complete", 1.0)

            if return_code != 0:
                error_msg = f"Video detection failed with exit code {return_code}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if not output_json.exists():
                raise RuntimeError("Output JSON was not created")

            logger.info(f"Video detection complete: {output_json}")

            return output_json

        except JobCancelledError:
            raise
        except subprocess.SubprocessError as e:
            logger.error(f"Video detection subprocess error: {e}", exc_info=True)
            raise RuntimeError(f"Video detection failed: {e}") from e
        except Exception as e:
            logger.error(f"Video detection error: {e}", exc_info=True)
            raise RuntimeError(f"Video detection failed: {e}") from e

    def _stream_process(
        self,
        command: list[str],
        env: dict[str, str],
        progress_callback: Callable[[str, float], None] | None,
        job_id: str | None,
    ) -> int:
        """
        Spawn `command`, stream its output into the log and the progress
        callback, and return its exit code.
        """
        # Run subprocess with progress streaming in its own process
        # group so cancel can take the whole tree down.
        process = popen_group(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        # Stream output and parse progress
        last_progress = 0.0
        with track_subprocess(job_id, process):
                for line in process.stdout:
                    line = line.strip()

                    # Log output
                    logger.debug(f"[VideoDetector] {line}")

                    # Parse device from PTDetector output (appears once during init)
                    if "PTDetector using device" in line:
                        raw = line.split("PTDetector using device")[-1].strip()
                        device_name = self._format_device_name(raw)
                        # Log at INFO so backend.log preserves the device for
                        # post-mortem checks (e.g. "did the GPU actually run
                        # this analysis?"). Without this the device only
                        # surfaces on the live progress modal, which the
                        # diagnostic ZIP can't reconstruct after the fact.
                        logger.info(
                            f"VideoDetector device: {device_name} (raw: {raw})"
                        )
                        if progress_callback:
                            try:
                                progress_callback(
                                    "Initializing detector...", 0.0,
                                    {"compute_device": device_name},
                                )
                            except TypeError:
                                pass

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
                                progress_callback(
                                    line if metrics else f"Processing video {current}/{total}",
                                    phase_progress,
                                    metrics,
                                )
                            except TypeError:
                                # Fallback for callbacks that don't accept metrics
                                progress_callback(
                                    f"Processing video {current}/{total}",
                                    phase_progress,
                                )
                            last_progress = phase_progress

                process.stdout.close()
                return process.wait()

    @staticmethod
    def _format_device_name(raw: str) -> str:
        """Convert raw device string to user-friendly name."""
        r = raw.lower()
        if "mps" in r:
            return "GPU (Apple Silicon)"
        if "cuda" in r:
            return "GPU (NVIDIA)"
        return "CPU"

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

            # Extract elapsed time (supports single-digit hours like "1:02:49")
            time_match = re.search(r"\[(\d{1,2}:\d{2}(?::\d{2})?)<", line)
            if time_match:
                metrics["elapsed"] = time_match.group(1)

            # Extract remaining time
            remaining_match = re.search(r"<(\d{1,2}:\d{2}(?::\d{2})?)", line)
            if remaining_match:
                metrics["remaining"] = remaining_match.group(1)

            if "current" in metrics and "total" in metrics:
                return metrics

        except (ValueError, IndexError, AttributeError):
            pass

        return None
