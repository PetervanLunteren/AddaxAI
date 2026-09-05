"""
Detect and track animals through videos in one pass, writing MegaDetector
JSON with a ``track_id`` on every box.

Runs as a subprocess in the detector's own environment (env-marine for
the underwater detectors) with no ``app.*`` imports, like
``classification_worker.py``. It is started with ``python -P`` so its
own directory stays off ``sys.path`` (that directory holds the app's
``megadetector.py``, which would shadow the megadetector package), and
it loads the sibling ``video_iter`` (its ``open_video``, for the header
read) by file path instead. Everything the app knows about the run comes
in on the command line; everything it learns goes out in the JSON.

Per video: decode the sampled frames through an ffmpeg pipe (``stride =
round(native_fps / fps)``, the same sampling ``process_video`` uses),
run the detector on each, hand the boxes to BoT-SORT, and keep the
tracker's output. ffmpeg rather than OpenCV because it gets the
platform's hardware decoder and scales the frame down in the same pass:
on a 4K HEVC BRUVS clip OpenCV's software decode cost 140 to 220 ms a
frame in the busy parts and its hardware path silently dropped to
software for whole stretches, where ffmpeg with VideoToolbox held 8 ms a
frame throughout (measured 2026-09-06). The frames arrive at most
``DECODE_WIDTH`` wide, which every detector here downsizes further
anyway, and every box is normalised, so the source size never matters. Only boxes
that belong to a surviving track are written: the tracker's own
thresholds are the storage floor, so a 90-minute reef video does not
write hundreds of thousands of noise rows nobody can address. Frame
numbers are absolute indices in the source video, as ``process_video``
writes them, so the rest of the app cannot tell the two apart.

Two loaders, chosen by the catalog's ``detector_runtime``:

- ``megadetector``: the megadetector package's ``load_detector``, which
  reads MegaDetector ``.pt`` files and RF-DETR ``.pth`` files (the
  Community Fish Detector) and carries their class names.
- ``ultralytics``: a plain ultralytics checkpoint such as SharkTrack,
  loaded directly, with ``model.names`` as the class map. The
  megadetector package forces its three classes onto any YOLO ``.pt``
  it does not know, which would turn a shark into "animal".

The tracker parameters and the false-positive filter are SharkTrack's
(Varini et al. 2024, supplement S3): BoT-SORT tuned on underwater
footage to MOTA 0.77, and a filter that drops a track shorter than one
second or nearly static when its best box is under 0.7, which removed
40% of false boxes at under 0.08% true positive loss. The buffer scales
with the sampling rate so "two seconds" stays two seconds at any fps.

Progress goes to stdout as tqdm lines over every sampled frame of the
run, which ``VideoDetectionModel._stream_process`` already parses, plus
the ``PTDetector using device`` line it reads the compute device from.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from tqdm import tqdm


def _load_video_iter():
    """Import the sibling ``video_iter`` module by path. Not through
    ``sys.path``: this directory also holds ``megadetector.py``, the
    app's wrapper, which would shadow the megadetector package."""
    path = Path(__file__).resolve().with_name("video_iter.py")
    spec = importlib.util.spec_from_file_location("video_iter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


open_video = _load_video_iter().open_video

# --- SharkTrack's BoT-SORT parameters (trackers/tracker_3fps.yaml) ---------
TRACK_HIGH_THRESH = 0.4
TRACK_LOW_THRESH = 0.2
NEW_TRACK_THRESH = 0.4
MATCH_THRESH = 0.97
TRACK_BUFFER_SECONDS = 2.0
# The detector's own floor: nothing below the tracker's second
# association threshold can ever join a track, so it is never asked for.
DETECT_CONF = TRACK_LOW_THRESH
DETECT_IOU = 0.5
ULTRALYTICS_DEFAULT_IMAGE_SIZE = 640
# Frames are decoded at most this wide. The 640 and 1024 px detectors
# downsize from here, and the tracker's motion step works on half of it.
DECODE_WIDTH = 1280

# --- SharkTrack's false-positive filter (supplement S3) ---------------------
FILTER_MIN_LIFE_SECONDS = 1.0
FILTER_MIN_MOTION = 0.08  # of the frame, whichever axis moved more
FILTER_MAX_CONF = 0.7  # tracks at or above this survive regardless


def _tracker_args(fps: float) -> SimpleNamespace:
    """BoT-SORT's config namespace, SharkTrack's values, buffer scaled to
    the sampling rate. ``fuse_score`` off: SharkTrack ran on ultralytics
    8.1.47, which did not fuse scores into the IoU distance. ReID off:
    the app carries no ReID model and SharkTrack found no need."""
    return SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=TRACK_HIGH_THRESH,
        track_low_thresh=TRACK_LOW_THRESH,
        new_track_thresh=NEW_TRACK_THRESH,
        track_buffer=max(1, round(TRACK_BUFFER_SECONDS * fps)),
        match_thresh=MATCH_THRESH,
        fuse_score=False,
        gmc_method="sparseOptFlow",
        proximity_thresh=1.0,
        appearance_thresh=0.25,
        with_reid=False,
        model="auto",
    )


# --- Detector loaders --------------------------------------------------------


def _device() -> str:
    """cuda, else Apple's MPS, else cpu. The app's GPU guard hides an
    unsupported CUDA device through CUDA_VISIBLE_DEVICES before we start."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _UltralyticsDetector:
    def __init__(self, model_path: Path, image_size: int | None, augment: bool) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.categories = {str(k): str(v) for k, v in self.model.names.items()}
        self.image_size = image_size or ULTRALYTICS_DEFAULT_IMAGE_SIZE
        self.augment = augment
        self.device = _device()
        print(f"PTDetector using device {self.device}", flush=True)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            frame_bgr,
            conf=DETECT_CONF,
            iou=DETECT_IOU,
            imgsz=self.image_size,
            device=self.device,
            augment=self.augment,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.concatenate(
            [
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy()[:, None],
                boxes.cls.cpu().numpy()[:, None],
            ],
            axis=1,
        ).astype(np.float32)


class _MegaDetectorDetector:
    def __init__(self, model_path: Path, image_size: int | None, augment: bool) -> None:
        from megadetector.detection.run_detector import load_detector

        # RF-DETR takes its resolution at load time and refuses it per
        # call; PTDetector takes it per call. Give both what they read.
        options = {"image_size": image_size} if image_size else None
        self.detector = load_detector(str(model_path), detector_options=options)
        self.categories = {str(k): str(v) for k, v in self.detector.detection_categories.items()}
        self.is_rfdetr = type(self.detector).__name__ == "RFDETRDetector"
        self.image_size = image_size
        self.augment = augment

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        # The package takes PIL in RGB, like the app's own image path.
        image = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        result = self.detector.generate_detections_one_image(
            image,
            image_id="frame",
            detection_threshold=DETECT_CONF,
            image_size=None if self.is_rfdetr else self.image_size,
            augment=self.augment,
        )
        rows = []
        for det in result.get("detections") or []:
            x, y, w, h = det["bbox"]
            rows.append(
                [x * width, y * height, (x + w) * width, (y + h) * height,
                 float(det["conf"]), int(det["category"])]
            )
        if not rows:
            return np.zeros((0, 6), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)


def _load_detector(runtime: str, model_path: Path, image_size: int | None, augment: bool):
    if runtime == "ultralytics":
        return _UltralyticsDetector(model_path, image_size, augment)
    if runtime == "megadetector":
        return _MegaDetectorDetector(model_path, image_size, augment)
    raise ValueError(f"unknown detector runtime {runtime!r}")


# --- Pure helpers (unit tested) ---------------------------------------------


def sampling_stride(native_fps: float, fps: float) -> int:
    """Every n-th source frame, ``round(native / fps)``, as ``process_video``
    samples; every frame when either rate is unknown."""
    return max(1, round(native_fps / fps)) if native_fps > 0 and fps > 0 else 1


def sampled_frames(frame_count: int, native_fps: float, fps: float) -> list[int]:
    """The frame indices ``process_video`` would sample: 0, stride, 2*stride..."""
    return list(range(0, max(0, frame_count), sampling_stride(native_fps, fps)))


def decode_size(width: int, height: int, max_width: int = DECODE_WIDTH) -> tuple[int, int]:
    """The frame size ffmpeg is asked for: at most ``max_width`` wide, the
    aspect kept, both sides even (what the encoder-side scaler needs and
    what makes the raw frame size exact)."""
    if width <= 0 or height <= 0:
        return (0, 0)
    out_w = min(width, max_width)
    out_h = round(height * out_w / width)
    return (out_w - out_w % 2, max(2, out_h - out_h % 2))


def ffmpeg_decode_cmd(
    ffmpeg: str, video: Path, stride: int, out_w: int, out_h: int
) -> list[str]:
    """The pipe: every ``stride``-th source frame, scaled, as raw BGR.

    ``select`` keeps frames by source index, so output frame ``i`` is
    source frame ``i * stride``, the same numbering ``process_video``
    writes; ``-fps_mode passthrough`` stops ffmpeg from duplicating
    frames to fill the timeline. ``-hwaccel auto`` takes the platform's
    hardware decoder and falls back to software without a word.
    """
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-hwaccel", "auto",
        "-i", str(video),
        "-vf", f"select=not(mod(n\\,{stride})),scale={out_w}:{out_h}",
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]


def ffmpeg_frames(ffmpeg: str, video: Path, stride: int, out_w: int, out_h: int):
    """Yield ``(frame_number, frame_bgr)`` for every sampled frame ffmpeg
    hands back, in order. Stops at the end of the stream; a decoder
    error after some frames ends the video there, with its message on
    stderr, rather than pretending the rest was empty."""
    frame_bytes = out_w * out_h * 3
    proc = subprocess.Popen(
        ffmpeg_decode_cmd(ffmpeg, video, stride, out_w, out_h),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    index = 0
    try:
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(out_h, out_w, 3)
            yield index * stride, frame
            index += 1
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
        code = proc.wait()
        if code != 0 or stderr:
            print(
                f"ffmpeg exit {code} on {video.name} after {index} frames: {stderr[-500:]}",
                file=sys.stderr, flush=True,
            )


def normalise_track_rows(
    rows: np.ndarray, width: int, height: int, frame_number: int
) -> list[dict]:
    """BoT-SORT output rows (x1, y1, x2, y2, track_id, conf, cls, idx) to
    JSON boxes with normalised ``[x, y, w, h]``, clamped to the frame."""
    boxes = []
    for x1, y1, x2, y2, track_id, conf, cls, _idx in rows.tolist():
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        boxes.append({
            "category": str(int(cls)),
            "conf": round(float(conf), 5),
            "bbox": [
                round(x1 / width, 5),
                round(y1 / height, 5),
                round((x2 - x1) / width, 5),
                round((y2 - y1) / height, 5),
            ],
            "frame_number": int(frame_number),
            "track_id": int(track_id),
        })
    return boxes


def filter_tracks(boxes: list[dict], fps: float) -> list[dict]:
    """SharkTrack's post-processing: drop every box of a track that is
    short (under one second of sampled frames) or nearly static (its
    centre moved under 8% of the frame on both axes) unless its best box
    scored 0.7 or more."""
    by_track: dict[int, list[dict]] = {}
    for box in boxes:
        by_track.setdefault(box["track_id"], []).append(box)
    min_life = max(1, round(FILTER_MIN_LIFE_SECONDS * fps))
    keep: set[int] = set()
    for track_id, members in by_track.items():
        max_conf = max(b["conf"] for b in members)
        if max_conf >= FILTER_MAX_CONF:
            keep.add(track_id)
            continue
        frames = {b["frame_number"] for b in members}
        cx = [b["bbox"][0] + b["bbox"][2] / 2 for b in members]
        cy = [b["bbox"][1] + b["bbox"][3] / 2 for b in members]
        motion = max(max(cx) - min(cx), max(cy) - min(cy))
        if len(frames) >= min_life and motion >= FILTER_MIN_MOTION:
            keep.add(track_id)
    return [b for b in boxes if b["track_id"] in keep]


def failure_entry(relative_path: str) -> dict:
    """MegaDetector's failure shape for a video that would not open."""
    return {
        "file": relative_path,
        "frame_rate": -1,
        "frames_processed": [],
        "detections": None,
        "failure": "Failure video access",
    }


def write_results(
    output_json: Path, images: list[dict], categories: dict[str, str], info: dict
) -> None:
    payload = {
        "images": images,
        "detection_categories": categories,
        "info": {
            "detection_completion_time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "format_version": "1.4",
            **info,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=1)


# --- The run -----------------------------------------------------------------


def gmc_downscale(frame_width: int) -> int:
    """How much BoT-SORT's camera-motion step shrinks the frame before
    looking for features: to about 960 px wide, never less than the
    tracker's own default of 2. On the full 4K frame the step cost 70 ms
    a sample; the transform it finds is scaled back by this factor, so
    the boxes it corrects are still in full-frame coordinates."""
    return max(2, round(frame_width / 960)) if frame_width > 0 else 2


def _video_frame_counts(videos: list[Path]) -> dict[Path, tuple[float, int, int, int]]:
    """``(native_fps, frame_count, width, height)`` per video, for the
    progress total, the sampling stride and the decode size. Read through
    OpenCV, which is cheap for the header. A video that will not open
    maps to ``(0, 0, 0, 0)``."""
    import cv2

    counts: dict[Path, tuple[float, int, int, int]] = {}
    for path in videos:
        cap = open_video(path)
        if cap is None:
            counts[path] = (0.0, 0, 0, 0)
            continue
        try:
            counts[path] = (
                float(cap.get(cv2.CAP_PROP_FPS)),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        finally:
            cap.release()
    return counts


def track_video(
    detector,
    tracker_args: SimpleNamespace,
    ffmpeg: str,
    path: Path,
    fps: float,
    native_fps: float,
    size: tuple[int, int],
    progress: tqdm,
) -> tuple[list[int], list[dict]]:
    """Run one video's sampled frames through the detector and a fresh
    tracker. Returns the frames actually decoded and the surviving boxes."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers import BOTSORT
    from ultralytics.trackers.utils.gmc import GMC

    stride = sampling_stride(native_fps, fps)
    out_w, out_h = decode_size(*size)
    tracker = BOTSORT(tracker_args)
    tracker.gmc = GMC(method=tracker_args.gmc_method, downscale=gmc_downscale(out_w))
    processed: list[int] = []
    boxes_out: list[dict] = []
    for frame_number, frame_bgr in ffmpeg_frames(ffmpeg, path, stride, out_w, out_h):
        processed.append(frame_number)
        detections = detector.detect(frame_bgr)
        # BoT-SORT wants the frame for its camera-motion compensation.
        tracked = tracker.update(Boxes(detections, (out_h, out_w)), frame_bgr)
        if len(tracked):
            boxes_out.extend(normalise_track_rows(tracked, out_w, out_h, frame_number))
        progress.update(1)
    return processed, filter_tracks(boxes_out, fps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "video_folder", type=Path,
        help="the deployment folder; paths in the JSON are relative to it",
    )
    parser.add_argument(
        "file_list", type=Path, help="JSON list of absolute video paths to process"
    )
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--fps", type=float, required=True, help="sampling rate in frames per second"
    )
    parser.add_argument(
        "--detector_runtime", choices=["megadetector", "ultralytics"], required=True
    )
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--ffmpeg", required=True, help="path to the ffmpeg binary")
    args = parser.parse_args()

    with open(args.file_list) as f:
        videos = [Path(p) for p in json.load(f)]
    detector = _load_detector(args.detector_runtime, args.model, args.image_size, args.augment)
    tracker_args = _tracker_args(args.fps)
    counts = _video_frame_counts(videos)
    total_frames = sum(
        len(sampled_frames(n, native, args.fps)) for native, n, _w, _h in counts.values()
    )

    images: list[dict] = []
    with tqdm(total=total_frames, unit="frame", desc="Tracking", file=sys.stdout,
              mininterval=1.0, dynamic_ncols=False) as progress:
        for path in videos:
            relative = str(path.relative_to(args.video_folder))
            native_fps, frame_count, width, height = counts[path]
            if frame_count <= 0 or width <= 0:
                images.append(failure_entry(relative))
                continue
            processed, boxes = track_video(
                detector, tracker_args, args.ffmpeg, path, args.fps, native_fps,
                (width, height), progress,
            )
            if not processed:
                images.append(failure_entry(relative))
                continue
            images.append({
                "file": relative,
                "frame_rate": native_fps,
                "frames_processed": processed,
                "detections": boxes,
            })
            # Progress in videos as well, for the log.
            # "of", not "/": the progress parser reads n/N off any line and
            # would count videos instead of frames.
            print(f"Tracked video {len(images)} of {len(videos)}: {relative}", flush=True)

    write_results(args.output_json, images, detector.categories, {
        "detector": args.model.name,
        "detector_runtime": args.detector_runtime,
        "tracker": "botsort",
        "fps": args.fps,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
