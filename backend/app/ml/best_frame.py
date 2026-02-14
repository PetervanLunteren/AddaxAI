"""
Best frame selection for video files.

After video detection (Phase 1) produces per-frame detections with confidence
scores, selects the "best" frame using detection confidence as the primary
signal and image sharpness (Laplacian variance) as a tiebreaker.

Blank videos (no animal detections) get the sharpest sampled frame.

Created by Claude Code on 2026-02-13
"""

import json
from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray
from PIL import Image

from app.core.logging_config import get_logger
from app.utils.video_utils import run_callback_on_frames

logger = get_logger(__name__)


def _score_frames(detections: list[dict]) -> dict[int, float]:
    """
    Score frames by summing animal detection confidences >= 0.3.

    Args:
        detections: List of detection dicts with 'frame_number', 'category', 'conf'

    Returns:
        Dict mapping frame_number -> summed confidence score
    """
    scores: dict[int, float] = {}
    for det in detections:
        frame_num = det.get("frame_number")
        if frame_num is None:
            continue
        # Only count animal detections (category "1") with confidence >= 0.3
        if str(det.get("category")) != "1":
            continue
        if float(det.get("conf", 0)) < 0.3:
            continue
        scores[frame_num] = scores.get(frame_num, 0.0) + float(det["conf"])
    return scores


def _compute_sharpness(image_np: ndarray) -> float:
    """
    Compute image sharpness using Laplacian variance.

    Args:
        image_np: Image as numpy array (RGB)

    Returns:
        Sharpness score (higher = sharper)
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pick_sharpest(video_path: str, candidate_frames: list[int]) -> int:
    """
    Pick the sharpest frame from a list of candidates.

    Args:
        video_path: Path to video file
        candidate_frames: List of frame numbers to evaluate

    Returns:
        Frame number of the sharpest frame
    """
    sharpness_scores: dict[int, float] = {}

    def sharpness_callback(image_np: ndarray, frame_filename: str) -> None:
        # Extract frame number from the synthetic filename
        from app.utils.video_utils import _filename_to_frame_number
        frame_num = _filename_to_frame_number(frame_filename)
        sharpness_scores[frame_num] = _compute_sharpness(image_np)

    run_callback_on_frames(
        video_path,
        sharpness_callback,
        frames_to_process=candidate_frames,
        verbose=False,
    )

    # Return the frame with the highest sharpness
    return max(sharpness_scores, key=sharpness_scores.get)  # type: ignore[arg-type]


def _extract_and_save_frame(video_path: str, frame_number: int, output_path: Path) -> None:
    """
    Extract a single frame from a video and save as JPEG.

    Args:
        video_path: Path to video file
        frame_number: 0-based frame index to extract
        output_path: Path to save the JPEG
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_callback(image_np: ndarray, frame_filename: str) -> None:
        img = Image.fromarray(image_np)
        img.save(str(output_path), "JPEG", quality=90)

    run_callback_on_frames(
        video_path,
        save_callback,
        frames_to_process=[frame_number],
        verbose=False,
    )


def select_best_frames(video_json_path: Path, deployment_folder: Path) -> None:
    """
    Select the best frame for each video in the detection JSON.

    Reads the detection JSON (output of Phase 1), computes the best frame
    number for each video, extracts and saves the frame as JPEG, and updates
    the JSON in-place with best_frame_number.

    Args:
        video_json_path: Path to detection_video.json
        deployment_folder: Path to deployment folder
    """
    with open(video_json_path) as f:
        data = json.load(f)

    frames_dir = deployment_folder / ".addaxai" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for img_entry in data.get("images", []):
        relative_file = img_entry["file"]
        video_path = str((deployment_folder / relative_file).resolve())
        video_stem = Path(relative_file).stem

        detections = img_entry.get("detections", [])
        frame_scores = _score_frames(detections)

        try:
            if not frame_scores:
                # Blank video: sample ~10 evenly-spaced frames, pick sharpest
                best_frame = _best_frame_for_blank_video(video_path)
            else:
                best_score = max(frame_scores.values())
                threshold = best_score * 0.9  # Within 10% of best
                candidates = [f for f, s in frame_scores.items() if s >= threshold]

                if len(candidates) == 1:
                    best_frame = candidates[0]
                else:
                    best_frame = _pick_sharpest(video_path, candidates)

            # Save frame as JPEG
            output_path = frames_dir / f"{video_stem}.jpg"
            _extract_and_save_frame(video_path, best_frame, output_path)

            # Update JSON entry
            img_entry["best_frame_number"] = best_frame

            logger.info(f"Best frame for {video_stem}: frame {best_frame}")

        except Exception as e:
            logger.error(f"Best frame selection failed for {video_stem}: {e}", exc_info=True)
            # Continue with next video

    # Write updated JSON back
    with open(video_json_path, "w") as f:
        json.dump(data, f, indent=2)


def _best_frame_for_blank_video(video_path: str) -> int:
    """
    For a blank video (no animal detections), sample ~10 evenly-spaced frames
    and return the sharpest one.

    Args:
        video_path: Path to video file

    Returns:
        Frame number of the sharpest sampled frame
    """
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if total_frames <= 0:
        return 0

    # Sample ~10 evenly-spaced frames
    num_samples = min(10, total_frames)
    step = max(1, total_frames // num_samples)
    sample_frames = list(range(0, total_frames, step))[:num_samples]

    return _pick_sharpest(video_path, sample_frames)
