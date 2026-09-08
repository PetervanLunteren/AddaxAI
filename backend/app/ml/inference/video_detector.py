"""
Video detection: every video goes through the tracking script.

`tracking_script.py` runs in the detector's own environment: the same
frame sampling MegaDetector's `process_video` used, the detector named by
the catalog's `detector_runtime`, BoT-SORT over the sampled frames, one
crop per track, and MegaDetector-shaped JSON with a `track_id` on every
box. This class builds its command line, streams its progress and
device into the job, and hands the JSON back. The confidence floors on
that command line come from `app.core.confidence`, the one source.
"""

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from app.core.confidence import DEFAULT_COUNTING_THRESHOLD, MD_OUTPUT_CONFIDENCE_THRESHOLD
from app.core.job_cancellation import (
    JobCancelledError,
    is_cancel_requested,
    track_subprocess,
)
from app.core.logging_config import get_logger
from app.core.subprocess_group import popen_group
from app.ml.environment_manager import EnvironmentManager
from app.ml.gpu_guard import cuda_guard_overrides
from app.utils.ffmpeg_bin import resolve_ffmpeg
from app.utils.subprocess_env import clean_python_env

logger = get_logger(__name__)


def _build_tracking_cmd(
    *,
    python_path: Path,
    model_path: Path,
    video_folder: Path,
    file_list_json: Path,
    output_json: Path,
    crops_dir: Path,
    fps: float,
    detector_runtime: str,
    ffmpeg_path: str,
    track_filter: bool,
    image_size: int | None,
    augment: bool,
) -> list[str]:
    """Assemble the ``tracking_script`` command line. The script takes an
    explicit file list rather than walking the folder, so the videos it
    reads are exactly the ones the worker's scan admitted (the media
    filter, and never a previous run's output folders).

    The tracker's floors travel here from ``app.core.confidence``: a
    track starts at the default counting threshold and keeps boxes down
    to the storage floor, the same for every detector and the same
    numbers the rest of the app hides and stores by.

    ``-P`` keeps the script's own directory off ``sys.path``: it holds
    ``megadetector.py``, the app's wrapper, which Python would otherwise
    find before the megadetector package."""
    command = [
        str(python_path),
        "-P",
        str(Path(__file__).parent / "tracking_script.py"),
        str(model_path),
        str(video_folder),
        str(file_list_json),
        str(output_json),
        "--fps",
        str(fps),
        "--detector_runtime",
        detector_runtime,
        "--ffmpeg",
        ffmpeg_path,
        "--crops_dir",
        str(crops_dir),
        "--track_high_thresh",
        str(DEFAULT_COUNTING_THRESHOLD),
        "--track_low_thresh",
        str(MD_OUTPUT_CONFIDENCE_THRESHOLD),
    ]
    if track_filter:
        command.append("--track_filter")
    if image_size is not None:
        command += ["--image_size", str(image_size)]
    if augment:
        command.append("--augment")
    return command


class VideoDetectionModel:
    """Runs the tracking script over a deployment's videos."""

    def __init__(
        self, model_path: Path, env_manager: EnvironmentManager, *, env_name: str
    ):
        """
        Initialize video detection model.

        Args:
            model_path: Path to .pt model file
            env_manager: Environment manager for accessing conda environments
            env_name: The catalog's `env` for this detector, as for
                `MegaDetectorV1000`.

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If environment not found
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.env_manager = env_manager
        self.env_name = env_name

        # Verify environment exists
        try:
            self.python_path = env_manager.get_python(f"env-{env_name}")
            logger.info(f"VideoDetectionModel using Python: {self.python_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to get Python environment: {e}") from e

    def detect_videos_to_json(
        self,
        *,
        video_folder: Path,
        video_files: list[Path],
        output_json: Path,
        crops_dir: Path,
        fps: float,
        detector_runtime: str,
        track_filter: bool,
        image_size: int | None = None,
        augment: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
        job_id: str | None = None,
    ) -> Path:
        """
        Detect and track through ``video_files`` (the worker's scan) and
        write MegaDetector-shaped JSON with a ``track_id`` on every box,
        plus one crop per track under ``crops_dir``.

        Args:
            video_folder: The deployment folder; JSON paths are relative to it.
            video_files: The videos to read, absolute.
            output_json: Where the results JSON goes.
            crops_dir: Root for the track crops (``<crops_dir>/<relative
                video>/track000007.jpg``), the folder the cover frames use.
            fps: Sampling rate in frames per second.
            detector_runtime: ``megadetector`` or ``ultralytics``, from the catalog.
            track_filter: Run SharkTrack's false-positive filter (catalog flag).
            image_size: Override the detector's long-edge resize size.
            augment: Run detection with augmentation.
            progress_callback: Optional callback(message, progress[, metrics]).
            job_id: For cancellation.

        Raises:
            ValueError: No videos to read.
            RuntimeError: The script failed.
        """
        if not video_files:
            raise ValueError("tracking needs the list of videos to read")
        file_list_json = output_json.with_name(output_json.stem + "_files.json")
        file_list_json.parent.mkdir(parents=True, exist_ok=True)
        with open(file_list_json, "w") as f:
            json.dump([str(p) for p in video_files], f)
        logger.info(
            f"Running video detection and tracking on {len(video_files)} "
            f"videos at {fps} FPS ({detector_runtime})"
        )
        command = _build_tracking_cmd(
            python_path=self.python_path,
            model_path=self.model_path,
            video_folder=video_folder,
            file_list_json=file_list_json,
            output_json=output_json,
            crops_dir=crops_dir,
            fps=fps,
            detector_runtime=detector_runtime,
            ffmpeg_path=resolve_ffmpeg(self.env_name),
            track_filter=track_filter,
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

                    # Parse progress from tqdm output: the script's bar over
                    # every sampled frame of the run ("45/100").
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
                                    line if metrics else f"Tracking frame {current}/{total}",
                                    phase_progress,
                                    metrics,
                                )
                            except TypeError:
                                # Fallback for callbacks that don't accept metrics
                                progress_callback(
                                    f"Tracking frame {current}/{total}",
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
