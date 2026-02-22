"""
Video utilities for frame extraction and processing.

Adapted from MegaDetector's video_utils.py to support video classification.
Uses OpenCV for video reading with multiple backend fallbacks for cross-platform compatibility.

Key functions:
- open_video(): Opens video with backend fallbacks
- run_callback_on_frames(): Extracts specific frames from video
- Frame naming helpers for consistent frame identification

Created: 2026-01-07
"""

import os
import re
import cv2
from typing import Callable, Optional, Union

from app.core.logging_config import get_logger
from app.core.media_types import VIDEO_EXTENSIONS

logger = get_logger(__name__)

# OpenCV backend constants
DEFAULT_BACKEND = -1

# Backend IDs to try in order for maximum compatibility
backend_id_to_name = {
    DEFAULT_BACKEND: 'default',
    cv2.CAP_FFMPEG: 'CAP_FFMPEG',
    cv2.CAP_DSHOW: 'CAP_DSHOW',
    cv2.CAP_MSMF: 'CAP_MSMF',
    cv2.CAP_AVFOUNDATION: 'CAP_AVFOUNDATION',
    cv2.CAP_GSTREAMER: 'CAP_GSTREAMER'
}


def _frame_number_to_filename(frame_number: int) -> str:
    """
    Convert frame number to synthetic filename.

    Args:
        frame_number: Frame index (0-based)

    Returns:
        Synthetic filename like "frame000042.jpg"
    """
    return f'frame{frame_number:06d}.jpg'


def _filename_to_frame_number(filename: str) -> int:
    """
    Extract frame number from synthetic filename.

    Args:
        filename: Filename created by _frame_number_to_filename

    Returns:
        Frame number extracted from filename

    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    filename = os.path.basename(filename)
    match = re.search(r'frame(\d+)\.jpg', filename)
    if match is None:
        raise ValueError(f'{filename} does not appear to be a frame file')

    try:
        frame_number = int(match.group(1))
    except Exception:
        raise ValueError(f'Filename {filename} does not contain a valid frame number')

    return frame_number


def open_video(video_path: str, verbose: bool = False) -> tuple[Optional[cv2.VideoCapture], Optional[object]]:
    """
    Open video file with multiple OpenCV backend fallbacks.

    Tries multiple backends in order for cross-platform compatibility:
    - Default (platform-specific)
    - CAP_FFMPEG (Linux)
    - CAP_DSHOW (Windows)
    - CAP_MSMF (Windows)
    - CAP_AVFOUNDATION (macOS)
    - CAP_GSTREAMER

    Args:
        video_path: Path to video file
        verbose: Enable debug output

    Returns:
        Tuple of (VideoCapture object, first frame) or (None, None) if all backends fail
    """
    if not os.path.isfile(video_path):
        logger.error(f'Video file {video_path} not found')
        return None, None

    backend_ids = backend_id_to_name.keys()

    for backend_id in backend_ids:
        backend_name = backend_id_to_name[backend_id]

        if verbose:
            logger.debug(f'Trying backend {backend_name} for {video_path}')

        try:
            if backend_id == DEFAULT_BACKEND:
                vidcap = cv2.VideoCapture(video_path)
            else:
                vidcap = cv2.VideoCapture(video_path, backend_id)
        except Exception as e:
            if verbose:
                logger.warning(f'Error opening {video_path} with backend {backend_name}: {e}')
            continue

        if not vidcap.isOpened():
            if verbose:
                logger.warning(f'isOpened() is False for {video_path} with backend {backend_name}')
            try:
                vidcap.release()
            except Exception:
                pass
            continue

        success, image = vidcap.read()
        if success and (image is not None):
            if verbose:
                logger.info(f'Successfully opened {video_path} with backend: {backend_name}')
            return vidcap, image

        logger.warning(f'Failed to read first frame from {video_path} with backend {backend_name}')
        try:
            vidcap.release()
        except Exception:
            pass

    logger.error(f'Failed to open {video_path} with any backend')
    return None, None


def run_callback_on_frames(
    input_video_file: str,
    frame_callback: Callable[[object, str], object],
    frames_to_process: Optional[Union[list[int], int]] = None,
    verbose: bool = False
) -> dict:
    """
    Extract specific frames from video and run callback on each.

    This is the core function for video frame extraction. It sequentially reads
    through a video and calls the provided callback on specified frames.

    Args:
        input_video_file: Path to video file
        frame_callback: Function that takes (image_np, frame_filename) and returns result
                       image_np is numpy array in RGB format (PIL convention)
                       frame_filename is synthetic name like "frame000042.jpg"
        frames_to_process: List of frame numbers to process (0-indexed)
                          If None, processes all frames
                          Can also be a single int for a single frame
        verbose: Enable debug output

    Returns:
        Dict with keys:
        - 'frame_filenames': List of synthetic frame filenames processed
        - 'frame_rate': Video frame rate (fps)
        - 'results': List of callback return values (one per frame)

    Raises:
        Exception: If video cannot be opened or contains no frames
    """
    if not os.path.isfile(input_video_file):
        raise FileNotFoundError(f'Video file {input_video_file} not found')

    if isinstance(frames_to_process, int):
        frames_to_process = [frames_to_process]

    vidcap = None

    try:
        vidcap, image = open_video(input_video_file, verbose=verbose)

        if vidcap is None:
            raise RuntimeError(f'Failed to open video {input_video_file}')

        n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = vidcap.get(cv2.CAP_PROP_FPS)

        if verbose:
            logger.info(f'Video {input_video_file} contains {n_frames} frames at {frame_rate} Hz')

        frame_filenames = []
        results = []

        for frame_number in range(0, n_frames):
            # We've already read the first frame when opening the video
            if frame_number != 0:
                success, image = vidcap.read()
            else:
                success = True

            if not success:
                if verbose:
                    logger.debug(f'Read terminating at frame {frame_number} of {n_frames}')
                break

            # Skip frames not in frames_to_process list
            if frames_to_process is not None:
                # Stop early if we've passed the last frame we need
                if frame_number > max(frames_to_process):
                    break
                # Skip frames not in the list
                if frame_number not in frames_to_process:
                    continue

            frame_filename = _frame_number_to_filename(frame_number)
            frame_filenames.append(frame_filename)

            # Convert from OpenCV BGR to PIL RGB format
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run callback on this frame
            frame_results = frame_callback(image_np, frame_filename)
            results.append(frame_results)

        if len(frame_filenames) == 0:
            raise Exception(f'Error: found no frames in file {input_video_file}')

        if verbose:
            logger.info(f'Processed {len(frame_filenames)} of {n_frames} frames for {input_video_file}')

    finally:
        if vidcap is not None:
            try:
                vidcap.release()
            except Exception:
                pass

    return {
        'frame_filenames': frame_filenames,
        'frame_rate': frame_rate,
        'results': results
    }
