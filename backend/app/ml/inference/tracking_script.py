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
``DECODE_LONG_EDGE`` on the long edge (the cap the cover frame has, so a
track's card cut from them keeps the resolution a photo card has), which
every detector here downsizes further anyway, and every box is
normalised, so the source size never matters. Only boxes that belong to
a surviving track are written: the tracker's thresholds are the storage
floor, so a 90-minute reef video does not write hundreds of thousands of
noise rows nobody can address. Frame numbers are absolute indices in the
source video, the numbering MegaDetector's own ``process_video`` used.

The card of each track is cut here too, while the frame is in memory:
the crop of its highest-confidence box so far is kept per live track
and the survivors are written when the video is done, one
``track000007.jpg`` per track beside the cover frame. No full frame per
track is ever stored; any other frame is decoded on request.

Two loaders, chosen by the catalog's ``detector_runtime``:

- ``megadetector``: the megadetector package's ``load_detector``, which
  reads MegaDetector ``.pt`` files and RF-DETR ``.pth`` files (the
  Community Fish Detector) and carries their class names.
- ``ultralytics``: a plain ultralytics checkpoint such as SharkTrack,
  loaded directly, with ``model.names`` as the class map. The
  megadetector package forces its three classes onto any YOLO ``.pt``
  it does not know, which would turn a shark into "animal".

The confidence floors come in on the command line from the app's own
constants (``app/core/confidence.py``): a track starts at the default
counting threshold and keeps boxes down to the storage floor, the same
for every detector. The association parameters are SharkTrack's (Varini
et al. 2024, supplement S3): BoT-SORT tuned on underwater footage to
MOTA 0.77, the buffer scaled with the sampling rate so "two seconds"
stays two seconds at any fps. SharkTrack's false-positive filter, which
drops a track shorter than one second or nearly static when its best
box is under 0.7 (40% of false boxes removed at under 0.08% true
positive loss, on sharks), runs only with ``--track_filter``: it would
delete a resting animal on a camera trap.

Progress goes to stdout as tqdm lines over every sampled frame of the
run, which ``VideoDetectionModel._stream_process`` already parses, plus
the ``PTDetector using device`` line it reads the compute device from.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from tqdm import tqdm


def _load_sibling(name: str):
    """Import a sibling module by path. Not through ``sys.path``: this
    directory also holds ``megadetector.py``, the app's wrapper, which
    would shadow the megadetector package."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


open_video = _load_sibling("video_iter").open_video
track_crop_filename = _load_sibling("scoring").track_crop_filename
_crop_box = _load_sibling("crop_box")

# --- SharkTrack's BoT-SORT association parameters (tracker_3fps.yaml) ------
# The confidence floors are not here: they come in on the command line
# from ``app/core/confidence.py``, so the app and the script cannot drift.
MATCH_THRESH = 0.97
TRACK_BUFFER_SECONDS = 2.0
DETECT_IOU = 0.5
ULTRALYTICS_DEFAULT_IMAGE_SIZE = 640
# Frames are decoded at most this long on their long edge, the cap the
# cover frame gets (`video_iter.write_best_frame`), so a track's card
# cut from a decoded frame matches a card cut from the cover. The
# detectors downsize from here; the motion step works on a fraction.
DECODE_LONG_EDGE = 1920
# A track's card: the crop service's own numbers, so a track card and a
# photo card are cut and encoded alike. 512 is the largest size the crop
# endpoint ever serves.
TRACK_CROP_MAX_SIDE = 512
TRACK_CROP_QUALITY = 85

# --- SharkTrack's false-positive filter (supplement S3) ---------------------
FILTER_MIN_LIFE_SECONDS = 1.0
FILTER_MIN_MOTION = 0.08  # of the frame, whichever axis moved more
FILTER_MAX_CONF = 0.7  # tracks at or above this survive regardless


def _tracker_args(fps: float, high: float, low: float) -> SimpleNamespace:
    """BoT-SORT's config namespace: a track starts from a box at ``high``
    or above and keeps boxes down to ``low``; SharkTrack's association
    values; buffer scaled to the sampling rate. ``fuse_score`` off:
    SharkTrack ran on ultralytics 8.1.47, which did not fuse scores into
    the IoU distance. ReID off: the app carries no ReID model and
    SharkTrack found no need."""
    return SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=high,
        track_low_thresh=low,
        new_track_thresh=high,
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
    def __init__(
        self, model_path: Path, image_size: int | None, augment: bool, conf: float
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.categories = {str(k): str(v) for k, v in self.model.names.items()}
        self.image_size = image_size or ULTRALYTICS_DEFAULT_IMAGE_SIZE
        self.augment = augment
        self.conf = conf
        self.device = _device()
        print(f"PTDetector using device {self.device}", flush=True)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            frame_bgr,
            conf=self.conf,
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
    def __init__(
        self, model_path: Path, image_size: int | None, augment: bool, conf: float
    ) -> None:
        from megadetector.detection.run_detector import load_detector

        # RF-DETR takes its resolution at load time and refuses it per
        # call; PTDetector takes it per call. Give both what they read.
        options = {"image_size": image_size} if image_size else None
        self.detector = load_detector(str(model_path), detector_options=options)
        # RF-DETR carries its class names on the detector; a MegaDetector
        # .pt has the package's fixed three.
        from megadetector.detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP

        categories = (
            getattr(self.detector, "detection_categories", None) or DEFAULT_DETECTOR_LABEL_MAP
        )
        self.categories = {str(k): str(v) for k, v in categories.items()}
        self.is_rfdetr = type(self.detector).__name__ == "RFDETRDetector"
        self.image_size = image_size
        self.augment = augment
        self.conf = conf
        # The app reads its compute device off this line (the image
        # detector prints it itself; the package's loader does not).
        device = getattr(self.detector, "device", None) or _device()
        print(f"PTDetector using device {device}", flush=True)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        # The package takes PIL in RGB, like the app's own image path.
        image = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        result = self.detector.generate_detections_one_image(
            image,
            image_id="frame",
            detection_threshold=self.conf,
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


def _load_detector(
    runtime: str, model_path: Path, image_size: int | None, augment: bool, conf: float
):
    if runtime == "ultralytics":
        return _UltralyticsDetector(model_path, image_size, augment, conf)
    if runtime == "megadetector":
        return _MegaDetectorDetector(model_path, image_size, augment, conf)
    raise ValueError(f"unknown detector runtime {runtime!r}")


# --- Pure helpers (unit tested) ---------------------------------------------


def sampling_stride(native_fps: float, fps: float) -> int:
    """Every n-th source frame, ``round(native / fps)``, as ``process_video``
    samples; every frame when either rate is unknown."""
    return max(1, round(native_fps / fps)) if native_fps > 0 and fps > 0 else 1


def sampled_frames(frame_count: int, native_fps: float, fps: float) -> list[int]:
    """The frame indices ``process_video`` would sample: 0, stride, 2*stride..."""
    return list(range(0, max(0, frame_count), sampling_stride(native_fps, fps)))


def decode_size(
    width: int, height: int, max_edge: int = DECODE_LONG_EDGE
) -> tuple[int, int]:
    """The frame size ffmpeg is asked for: at most ``max_edge`` on the
    long edge (so a portrait clip is capped by its height), the aspect
    kept, both sides even (what the scaler needs and what makes the raw
    frame size exact)."""
    if width <= 0 or height <= 0:
        return (0, 0)
    scale = min(1.0, max_edge / max(width, height))
    out_w = round(width * scale)
    out_h = round(height * scale)
    return (max(2, out_w - out_w % 2), max(2, out_h - out_h % 2))


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


def cut_track_crop(frame: Image.Image, bbox: list[float]) -> bytes:
    """A track's card from a decoded frame: the crop service's geometry
    (a square around the box with context padding, blurred fill past the
    edge) on the normalised box the JSON carries, shrunk to at most
    ``TRACK_CROP_MAX_SIDE`` and encoded like a photo card."""
    x, y, w, h = bbox
    left, top, right, bottom = _crop_box.compute_expanded_crop_region(
        x, y, w, h, frame.width, frame.height
    )
    crop = _crop_box.crop_with_blur_fill(frame, left, top, right, bottom)
    crop.thumbnail((TRACK_CROP_MAX_SIDE, TRACK_CROP_MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=TRACK_CROP_QUALITY)
    return buf.getvalue()


def keep_best_crop(
    best: dict[int, tuple[float, int, bytes]],
    boxes: list[dict],
    frame_bgr: np.ndarray,
    frame_number: int,
) -> None:
    """Keep, per track, the crop of its highest-confidence box so far.

    Compared on the rounded confidence that goes into the JSON, strictly
    greater, so the crop lands on the frame `scoring.summarise_tracks`
    will name as the track's representative when it reads that JSON
    back (highest confidence, ties to the earliest frame). The frame is
    converted once, and only when some track improves on it.
    """
    frame: Image.Image | None = None
    for box in boxes:
        track_id = box["track_id"]
        prev = best.get(track_id)
        if prev is not None and box["conf"] <= prev[0]:
            continue
        if frame is None:
            frame = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        best[track_id] = (box["conf"], frame_number, cut_track_crop(frame, box["bbox"]))


def write_track_crops(
    crops_dir: Path, best: dict[int, tuple[float, int, bytes]], surviving: set[int]
) -> None:
    """Write the crops of the tracks that survived the filter, named by
    `track_crop_filename`. A crop that cannot be written is reported and
    skipped; the ingest then leaves that track without a picture."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    for track_id in sorted(surviving):
        entry = best.get(track_id)
        if entry is None:
            continue
        dest = crops_dir / track_crop_filename(track_id)
        try:
            dest.write_bytes(entry[2])
        except OSError as e:
            print(f"could not write {dest}: {e}", file=sys.stderr, flush=True)


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
    track_filter: bool,
) -> tuple[list[int], list[dict], dict[int, tuple[float, int, bytes]]]:
    """Run one video's sampled frames through the detector and a fresh
    tracker. Returns the frames actually decoded, the surviving boxes,
    and the best crop per track (surviving tracks included, the filtered
    ones too; the caller writes only the survivors)."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers import BOTSORT
    from ultralytics.trackers.utils.gmc import GMC

    stride = sampling_stride(native_fps, fps)
    out_w, out_h = decode_size(*size)
    tracker = BOTSORT(tracker_args)
    tracker.gmc = GMC(method=tracker_args.gmc_method, downscale=gmc_downscale(out_w))
    processed: list[int] = []
    boxes_out: list[dict] = []
    best_crops: dict[int, tuple[float, int, bytes]] = {}
    for frame_number, frame_bgr in ffmpeg_frames(ffmpeg, path, stride, out_w, out_h):
        processed.append(frame_number)
        detections = detector.detect(frame_bgr)
        # BoT-SORT wants the frame for its camera-motion compensation.
        tracked = tracker.update(Boxes(detections, (out_h, out_w)), frame_bgr)
        if len(tracked):
            boxes = normalise_track_rows(tracked, out_w, out_h, frame_number)
            boxes_out.extend(boxes)
            keep_best_crop(best_crops, boxes, frame_bgr, frame_number)
        progress.update(1)
    if track_filter:
        boxes_out = filter_tracks(boxes_out, fps)
    return processed, boxes_out, best_crops


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
    parser.add_argument(
        "--crops_dir", type=Path, required=True,
        help="where each track's card is written, under the video's relative path",
    )
    parser.add_argument(
        "--track_high_thresh", type=float, required=True,
        help="a track starts from a box at or above this confidence",
    )
    parser.add_argument(
        "--track_low_thresh", type=float, required=True,
        help="boxes down to this confidence extend a track; nothing below is asked for",
    )
    parser.add_argument(
        "--track_filter", action="store_true",
        help="drop short or static tracks under 0.7 (SharkTrack's false-positive filter)",
    )
    args = parser.parse_args()

    with open(args.file_list) as f:
        videos = [Path(p) for p in json.load(f)]
    detector = _load_detector(
        args.detector_runtime, args.model, args.image_size, args.augment,
        args.track_low_thresh,
    )
    tracker_args = _tracker_args(args.fps, args.track_high_thresh, args.track_low_thresh)
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
            processed, boxes, best_crops = track_video(
                detector, tracker_args, args.ffmpeg, path, args.fps, native_fps,
                (width, height), progress, args.track_filter,
            )
            if not processed:
                images.append(failure_entry(relative))
                continue
            write_track_crops(
                args.crops_dir / relative, best_crops, {b["track_id"] for b in boxes}
            )
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
        "track_high_thresh": args.track_high_thresh,
        "track_low_thresh": args.track_low_thresh,
        "track_filter": args.track_filter,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
