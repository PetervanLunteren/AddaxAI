"""
Frame extraction service for video files.

Extracts ALL analyzed frames from videos to disk using MegaDetector's
extract_frames_from_video CLI tool. Frames are saved as JPEGs and later
ingested as individual File rows in the database.

Created by Claude Code on 2026-02-20
"""

import json
import re
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
        "80",
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


_FRAME_NAME_RE = re.compile(r"^frame(\d+)\.jpg$")


def cleanup_unused_frames(video_json_path: Path, frames_base_dir: Path) -> int:
    """
    Delete extracted frame JPEGs that no detection or best-frame pointer
    refers to.

    For each video in the detection JSON, the keep-set is
    {frame_numbers in detections} ∪ {best_frame_number}. Everything else
    under that video's frames subdirectory is removed. Empty subdirectories
    are left in place because downstream code already tolerates missing
    dirs and re-creating one on retry is free.

    Best-effort: per-file unlink errors are logged and skipped, the
    overall sweep keeps going. The JSON must have been updated with
    best_frame_number first (run after select_best_frames).

    Args:
        video_json_path: Path to detection_video.json. Returns 0 if missing.
        frames_base_dir: Path to the video_frames directory.

    Returns:
        Number of JPEGs deleted across all videos.
    """
    if not video_json_path.exists():
        return 0

    with open(video_json_path) as f:
        data = json.load(f)

    deleted_count = 0

    for img_entry in data.get("images") or []:
        # Skip process_video failure entries (corrupt videos): no
        # detections recorded and no frames extracted.
        if img_entry.get("failure"):
            continue

        relative_file = img_entry["file"]
        frames_dir = frames_base_dir / relative_file
        if not frames_dir.exists():
            continue

        keep: set[int] = set()
        for det in img_entry.get("detections") or []:
            fn = det.get("frame_number")
            if fn is not None:
                keep.add(int(fn))
        best = img_entry.get("best_frame_number")
        if best is not None:
            keep.add(int(best))

        for frame_jpg in frames_dir.glob("frame*.jpg"):
            m = _FRAME_NAME_RE.match(frame_jpg.name)
            if m is None:
                continue
            if int(m.group(1)) in keep:
                continue
            try:
                frame_jpg.unlink()
                deleted_count += 1
            except OSError as e:
                logger.warning(f"Could not delete {frame_jpg}: {e}")

    logger.info(f"Frame cleanup: deleted {deleted_count} unused JPEGs")
    return deleted_count
