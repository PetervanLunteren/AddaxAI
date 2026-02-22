"""
Best frame selection for video files.

After video detection (Phase 1) produces per-frame detections with confidence
scores, selects the "best" frame using detection confidence as the primary
signal and image sharpness (Laplacian variance) as a tiebreaker.

Blank videos (no detections) get the sharpest sampled frame.

Created by Claude Code on 2026-02-13
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.logging_config import get_logger
from app.ml.scoring import compute_sharpness, pick_best_candidate, score_detections

logger = get_logger(__name__)


def _pick_sharpest(frames_dir: Path, candidate_frames: list[int]) -> int:
    """
    Pick the sharpest frame from a list of candidates by reading JPEGs from disk.

    Args:
        frames_dir: Directory containing extracted frame JPEGs
        candidate_frames: List of frame numbers to evaluate

    Returns:
        Frame number of the sharpest frame
    """
    sharpness_scores: dict[int, float] = {}

    for frame_num in candidate_frames:
        frame_path = frames_dir / f"frame{frame_num:06d}.jpg"
        if not frame_path.exists():
            logger.debug(f"Frame {frame_path.name} not found, skipping")
            continue
        image_np = np.array(Image.open(frame_path))
        sharpness_scores[frame_num] = compute_sharpness(image_np)

    if not sharpness_scores:
        # Fall back to first candidate if no frames found on disk
        return candidate_frames[0] if candidate_frames else 0

    return max(sharpness_scores, key=sharpness_scores.get)  # type: ignore[arg-type]


def _blank_video_sample_frames(frames_dir: Path) -> list[int]:
    """
    Sample ~3 evenly-spaced frame numbers from available extracted frames.

    Args:
        frames_dir: Directory containing extracted frame JPEGs

    Returns:
        List of frame numbers to evaluate
    """
    import re
    frame_files = sorted(frames_dir.glob("frame*.jpg"))
    if not frame_files:
        return [0]

    # Extract frame numbers from filenames
    frame_numbers = []
    for f in frame_files:
        m = re.match(r"frame(\d+)\.jpg", f.name)
        if m:
            frame_numbers.append(int(m.group(1)))

    if not frame_numbers:
        return [0]

    num_samples = min(3, len(frame_numbers))
    step = max(1, len(frame_numbers) // num_samples)
    return [frame_numbers[i] for i in range(0, len(frame_numbers), step)][:num_samples]


def select_best_frames(video_json_path: Path, frames_base_dir: Path) -> None:
    """
    Select the best frame for each video in the detection JSON.

    Reads the detection JSON (output of Phase 1), computes the best frame
    number for each video, and updates the JSON in-place with
    best_frame_number. No frames are saved — the corresponding JPEG already
    exists in video_frames/.

    Args:
        video_json_path: Path to detection_video.json
        frames_base_dir: Path to video_frames directory (e.g. .addaxai/projects/{id}/video_frames)
    """
    with open(video_json_path) as f:
        data = json.load(f)

    for img_entry in data.get("images", []):
        relative_file = img_entry["file"]
        video_name = Path(relative_file).name
        video_stem = Path(relative_file).stem

        # Frames directory: {frames_base_dir}/{video_filename}/
        frames_dir = frames_base_dir / video_name

        if not frames_dir.exists():
            logger.warning(f"Frames directory not found for {video_name}, skipping best frame selection")
            continue

        detections = img_entry.get("detections", [])

        # Build detection tuples: (frame_number_str, confidence, bbox)
        det_tuples = [
            (str(det["frame_number"]), float(det.get("conf", 0)), tuple(det["bbox"]))
            for det in detections
            if det.get("frame_number") is not None
        ]

        frame_scores = score_detections(det_tuples)

        # Sharpness tiebreaker wrapper: converts str keys <-> int frame numbers
        def get_sharpest(keys: list[str], _fd: Path = frames_dir) -> str:
            frame_nums = [int(k) for k in keys]
            return str(_pick_sharpest(_fd, frame_nums))

        fallback_keys = [str(f) for f in _blank_video_sample_frames(frames_dir)]

        try:
            best_key = pick_best_candidate(
                frame_scores,
                get_sharpest=get_sharpest,
                fallback_keys=fallback_keys,
            )
            best_frame = int(best_key) if best_key is not None else 0

            # Update JSON entry (frame already exists in video_frames/)
            img_entry["best_frame_number"] = best_frame

            logger.info(f"Best frame for {video_stem}: frame {best_frame}")

        except Exception as e:
            logger.error(f"Best frame selection failed for {video_stem}: {e}", exc_info=True)
            # Continue with next video

    # Write updated JSON back
    with open(video_json_path, "w") as f:
        json.dump(data, f, indent=2)
