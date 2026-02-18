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
from numpy import ndarray
from PIL import Image

from app.core.logging_config import get_logger
from app.ml.scoring import compute_sharpness, pick_best_candidate, score_detections
from app.utils.video_utils import run_callback_on_frames

logger = get_logger(__name__)


def _pick_sharpest(video_path: str, candidate_frames: list[int]) -> int:
    """
    Pick the sharpest frame from a list of candidates.

    Uses batch frame extraction via run_callback_on_frames for efficiency
    (single sequential pass through the video file).

    Args:
        video_path: Path to video file
        candidate_frames: List of frame numbers to evaluate

    Returns:
        Frame number of the sharpest frame
    """
    sharpness_scores: dict[int, float] = {}

    def sharpness_callback(image_np: ndarray, frame_filename: str) -> None:
        from app.utils.video_utils import _filename_to_frame_number
        frame_num = _filename_to_frame_number(frame_filename)
        sharpness_scores[frame_num] = compute_sharpness(image_np)

    run_callback_on_frames(
        video_path,
        sharpness_callback,
        frames_to_process=candidate_frames,
        verbose=False,
    )

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


def _blank_video_sample_frames(video_path: str) -> list[int]:
    """
    Sample ~10 evenly-spaced frame numbers from a video.

    Args:
        video_path: Path to video file

    Returns:
        List of frame numbers to evaluate
    """
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if total_frames <= 0:
        return [0]

    num_samples = min(10, total_frames)
    step = max(1, total_frames // num_samples)
    return list(range(0, total_frames, step))[:num_samples]


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

        # Build detection tuples: (frame_number_str, confidence)
        det_tuples = [
            (str(det["frame_number"]), float(det.get("conf", 0)))
            for det in detections
            if det.get("frame_number") is not None and str(det.get("category")) == "1"
        ]

        frame_scores = score_detections(det_tuples)

        # Sharpness tiebreaker wrapper: converts str keys <-> int frame numbers
        def get_sharpest(keys: list[str], _vp: str = video_path) -> str:
            frame_nums = [int(k) for k in keys]
            return str(_pick_sharpest(_vp, frame_nums))

        fallback_keys = [str(f) for f in _blank_video_sample_frames(video_path)]

        try:
            best_key = pick_best_candidate(
                frame_scores,
                get_sharpest=get_sharpest,
                fallback_keys=fallback_keys,
            )
            best_frame = int(best_key) if best_key is not None else 0

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
