"""
Frame extraction service for video files.

Extracts ALL analyzed frames from videos to disk using MegaDetector's
extract_frames_from_video CLI tool. Frames are saved as JPEGs and later
ingested as individual File rows in the database.

Created by Claude Code on 2026-02-20
"""

import subprocess
from pathlib import Path

from app.core.job_cancellation import (
    JobCancelledError,
    is_cancel_requested,
    track_subprocess,
)
from app.core.logging_config import get_logger
from app.core.subprocess_group import popen_group
from app.ml.environment_manager import EnvironmentManager

logger = get_logger(__name__)

# Output directory name within .addaxai
VIDEO_FRAMES_DIR = "video_frames"


def extract_all_video_frames(
    deployment_folder: Path,
    video_fps: float,
    env_manager: EnvironmentManager,
    output_dir: Path,
    job_id: str | None = None,
) -> Path:
    """
    Extract all analyzed frames from videos in a deployment folder.

    Uses MegaDetector's extract_frames_from_video CLI tool which recursively
    finds videos and extracts frames at the configured FPS. Output goes to
    output_dir/{video_stem}/ with frame naming like frame000000.jpg.

    Args:
        deployment_folder: Path to the deployment folder containing videos
        video_fps: Frames per second used during detection (e.g., 1.0)
        env_manager: Environment manager for accessing the conda env
        output_dir: Path to write extracted frames to

    Returns:
        Path to the output directory

    Raises:
        RuntimeError: If frame extraction fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    python_path = env_manager.get_python("env-addaxai-base")

    # Convert FPS to frame_sample parameter
    # --frame_sample -1.0 means "extract 1 frame per second"
    # --frame_sample -0.5 means "extract 1 frame per 2 seconds" (0.5 FPS)
    # The negative value signals time-based sampling (seconds between frames)
    frame_sample = -1.0 / video_fps

    command = [
        str(python_path),
        "-m",
        "megadetector.utils.extract_frames_from_video",
        str(deployment_folder),
        str(output_dir),
        "--frame_sample",
        str(frame_sample),
        "--quality",
        "90",
        "--n_workers",
        "4",
    ]

    logger.info(f"Extracting video frames: {' '.join(command)}")

    try:
        process = popen_group(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        with track_subprocess(job_id, process):
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.debug(f"[FrameExtract] {line}")

            process.stdout.close()
            return_code = process.wait()

        if job_id and is_cancel_requested(job_id):
            raise JobCancelledError()

        if return_code != 0:
            raise RuntimeError(
                f"Frame extraction failed with exit code {return_code}"
            )

        logger.info(f"Frame extraction complete: {output_dir}")
        return output_dir

    except JobCancelledError:
        raise
    except subprocess.SubprocessError as e:
        logger.error(f"Frame extraction subprocess error: {e}", exc_info=True)
        raise RuntimeError(f"Frame extraction failed: {e}") from e
