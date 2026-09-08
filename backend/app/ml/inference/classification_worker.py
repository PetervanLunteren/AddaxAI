"""
One-shot batch classification worker for subprocess execution.

Usage: python classification_worker.py <model_dir> <model_path> <input_json> <output_json>

Reads all detections from input_json, classifies them, writes results to output_json.
Progress and status are streamed via stderr as JSON lines.

Items come in two flavours via a `source` discriminator:

    {"source": "image", "image_path": "...jpg", "bbox": [x, y, w, h]}
    {"source": "video", "video_path": "...mp4", "frame_number": 42,
     "bbox": [...], "detection_conf": 0.83}

Video items don't have a pre-extracted frame JPEG on disk. The worker
opens the source video with cv2, iterates frames sequentially using
`video_iter.iter_wanted_frames`, and crops every item pinned to that
frame. In the same pass it keeps the one frame chosen as the video's
best, and writes its JPEG to a per-video output directory listed in
`best_frame_outputs`.

Best-frame scoring runs on a separate population from classification,
supplied as `scoring_detections`:

    {"scoring_detections": {"...mp4": [{"frame_number": 42, "conf": 0.83,
                                        "bbox": [x, y, w, h]}, ...]}}

That list holds every detection on the video, any category. `items` holds
only animals above the classification gate, which is the wrong population
to pick a thumbnail from: a clip containing only people scored nothing and
fell back to sharpness over three arbitrary samples. Detection confidence
is emitted by every detector, so scoring on it is one rule that behaves
the same with or without a classifier and needs no knowledge of what the
categories mean (`fish`, `shark`, `turtle` work unchanged).

Output JSON adds a `best_frames` map: `{video_path: best_frame_number}`
so the parent process can stamp `best_frame_number` onto the deployment's
detection JSON before DB load.

Supports two model modes:
- Batch mode: if the model implements get_tensor() + classify_batch(),
  crops are accumulated and processed in one GPU forward pass per batch.
- Per-crop mode (fallback): calls get_classification() one crop at a time.

Created by Claude Code on 2026-01-05
Updated on 2026-03-14 - Simplified from persistent worker to one-shot batch
Updated on 2026-03-26 - Added image caching and batch inference support
Updated on 2026-05-13 - Stream frames from source videos; fuse best-frame
                       scoring into the classification pass; stop relying
                       on bulk-extracted frame JPEGs.
Updated on 2026-07-31 - Score the best frame on every detection regardless
                       of category, via `scoring_detections`, instead of on
                       the animals-only classification input.
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import cv2  # frame count for the blank-video fallback
import numpy as np  # batch tensor stacking
from PIL import Image, ImageFile

# Decode truncated JPEGs instead of refusing them. Camera traps write
# them routinely (dead battery, pulled card, interrupted copy) and this
# script runs in its own subprocess, so it cannot inherit the setting
# from `app/__init__.py`. See the comment there for the full reasoning.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# `video_iter` and `scoring` live next to this script. Python adds the
# script's directory to sys.path automatically, so a flat import works
# in the subprocess (which has no app.* on its path).
from scoring import choose_frame_number  # noqa: E402
from video_iter import (  # noqa: E402
    iter_wanted_frames,
    open_video,
    read_frame_by_seek,
    write_best_frame,
)

# Keep only the top-N classifications per detection in the output JSON.
# Classifiers like SpeciesNet return the full ~2000+ class softmax, almost all
# near-zero; writing all of them produced multi-GB detection JSONs that filled
# the disk during merge and exhausted memory (Simon's 8 GB file). Every
# downstream consumer already caps at the top 5: `trim_classification_results`
# (app/ml/json_utils.py, default 5, "matches SpeciesNet API"), the rollup
# (`taxonomic_rollup.py`, top-5), and the DB load (top-1). This MUST stay >=
# that downstream trim default, or we would silently drop classes it still
# wants; the test in tests/ml/test_classification_topk_cap.py pins that.
MAX_CLASSIFICATIONS_KEPT = 5


def _has_nonfinite_confidence(classifications: list) -> bool:
    """True if any (label, conf) pair has a NaN or inf confidence."""
    for entry in classifications:
        try:
            conf = entry[1]
        except (IndexError, TypeError):
            return True
        if not isinstance(conf, (int, float)) or not math.isfinite(conf):  # noqa: UP038 (Python 3.8 compat)
            return True
    return False


def load_inference_class(model_dir: Path, model_path: Path):
    """
    Dynamically load and instantiate ModelInference class from model directory.
    """
    inference_script = model_dir / "inference.py"

    if not inference_script.exists():
        raise FileNotFoundError(
            f"Custom inference script not found: {inference_script}\n"
            f"Model developers must provide inference.py in their model directory."
        )

    module_name = f"custom_inference_{model_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, inference_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {inference_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "ModelInference"):
        raise AttributeError(
            "inference.py must define a 'ModelInference' class.\n"
            "See /backend/templates/inference_template.py for reference."
        )

    return module.ModelInference(model_dir, model_path)


def validate_interface(model_inference) -> None:
    """Validate that ModelInference instance provides required methods."""
    required_methods = [
        "check_gpu",
        "load_model",
        "get_crop",
        "get_classification",
        "get_class_names",
    ]
    missing = [m for m in required_methods if not hasattr(model_inference, m)]
    if missing:
        raise ValueError(
            f"ModelInference class missing required methods: {', '.join(missing)}\n"
            f"Required: {', '.join(required_methods)}"
        )
    for method_name in required_methods:
        if not callable(getattr(model_inference, method_name)):
            raise ValueError(f"Required attribute '{method_name}' is not callable")


def detect_device_name(gpu_available: bool) -> str:
    """Detect friendly device name from ML frameworks loaded in this process."""
    if not gpu_available:
        return "CPU"
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "GPU (Apple Silicon)"
        if torch.cuda.is_available():
            return "GPU (NVIDIA)"
    except ImportError:
        pass
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            if platform.system() == "Darwin":
                return "GPU (Apple Silicon)"
            return "GPU (NVIDIA)"
    except ImportError:
        pass
    return "GPU"


def emit(data: dict) -> None:
    """Emit a JSON line to stderr for the parent process."""
    print(json.dumps(data), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Per-item classification work
# ---------------------------------------------------------------------------


def _classify_one(
    model_inference,
    image: Image.Image,
    bbox,
    batched_state: dict | None,
    orig_idx: int,
    results: list,
) -> None:
    """
    Run the classifier on a single (image, bbox) crop.

    `batched_state` is None in per-crop mode. In batch mode it's the
    accumulator dict shared across calls (`batch_indices`, `batch_tensors`,
    `flush_batch`, `batch_size`).
    """
    try:
        crop = model_inference.get_crop(image, tuple(bbox))
        if crop is None:
            results[orig_idx] = {
                "success": False,
                "error": f"Invalid crop for bbox {bbox}",
            }
            return

        if batched_state is None:
            classifications = model_inference.get_classification(crop)
            if not classifications:
                results[orig_idx] = {
                    "success": False,
                    "error": f"Empty result for bbox {bbox}",
                }
                return
            if _has_nonfinite_confidence(classifications):
                results[orig_idx] = {
                    "success": False,
                    "error": "Model produced non-finite confidence (NaN/inf)",
                }
                return
            sorted_cls = sorted(classifications, key=lambda x: x[1], reverse=True)
            results[orig_idx] = {
                "success": True,
                "classifications": sorted_cls[:MAX_CLASSIFICATIONS_KEPT],
            }
            return

        tensor = model_inference.get_tensor(crop)
        batched_state["batch_tensors"].append(tensor)
        batched_state["batch_indices"].append(orig_idx)
        if len(batched_state["batch_tensors"]) >= batched_state["batch_size"]:
            batched_state["flush_batch"]()
    except Exception as e:
        results[orig_idx] = {"success": False, "error": str(e)}


def _make_batched_state(model_inference, batch_size: int, items: list, results: list, emit_fn):
    """Build the shared batched-state dict used by `_classify_one`."""
    state: dict = {
        "batch_indices": [],
        "batch_tensors": [],
        "batch_size": batch_size,
        "processed": 0,
        "total": len(items),
    }

    def flush_batch() -> None:
        if not state["batch_tensors"]:
            return
        batch = np.stack(state["batch_tensors"])
        batch_results = model_inference.classify_batch(batch)
        for idx, classifications in zip(state["batch_indices"], batch_results):  # noqa: B905
            if _has_nonfinite_confidence(classifications):
                src = items[idx]
                src_label = src.get("image_path") or src.get("video_path")
                print(
                    f"[Worker] Non-finite confidence from classify_batch "
                    f"for source={src_label!r} bbox={src.get('bbox')!r}",
                    file=sys.stderr, flush=True,
                )
                results[idx] = {
                    "success": False,
                    "error": "Model produced non-finite confidence (NaN/inf)",
                }
                continue
            sorted_cls = sorted(classifications, key=lambda x: x[1], reverse=True)
            results[idx] = {
                "success": True,
                "classifications": sorted_cls[:MAX_CLASSIFICATIONS_KEPT],
            }
        state["processed"] += len(state["batch_indices"])
        emit_fn({"current": state["processed"], "total": state["total"]})
        state["batch_indices"].clear()
        state["batch_tensors"].clear()

    state["flush_batch"] = flush_batch
    return state


# ---------------------------------------------------------------------------
# Group items by source and run them
# ---------------------------------------------------------------------------


def _process_image_group(
    model_inference,
    image_path: str,
    image_items: list[tuple[int, dict]],
    results: list,
    batched_state: dict | None,
    per_crop_progress,
) -> None:
    """Classify every item in one image. PIL opens the image exactly once."""
    path = Path(image_path)
    if not path.exists():
        for orig_idx, _ in image_items:
            results[orig_idx] = {
                "success": False,
                "error": f"Image not found: {image_path}",
            }
        if batched_state is None:
            per_crop_progress(len(image_items))
        return

    # Convert to RGB at open time so every model's get_crop receives a
    # 3-channel image. Camera-trap frames are almost always RGB JPEGs, but
    # a grayscale, palette, or RGBA source would otherwise reach a model
    # with the wrong channel count if that model crops before converting.
    image = Image.open(path).convert("RGB")
    for orig_idx, item in image_items:
        _classify_one(
            model_inference, image, item["bbox"], batched_state, orig_idx, results
        )
        if batched_state is None:
            per_crop_progress(1)


def _process_video_group(
    model_inference,
    video_path: str,
    video_items: list[tuple[int, dict]],
    scoring_dets: list[dict],
    best_frame_dest_dir: Path | None,
    results: list,
    batched_state: dict | None,
    per_crop_progress,
) -> int | None:
    """
    Stream a single source video, classify every item pinned to one of
    its frames, and score the best frame in the same pass. Returns the
    chosen `best_frame_number` (or None if the video couldn't be opened
    / had no frames).

    `video_items` is what gets classified: animals above the project's
    classification gate. `scoring_dets` is what best-frame selection runs
    on: every detection on this video, any category, as
    `{"frame_number", "conf", "bbox"}`. The two populations are different
    on purpose. Scoring off `video_items` meant a clip containing only
    people scored nothing and fell through to sharpness over three
    arbitrary samples, and it made the chosen frame depend on which
    classifier was configured. Detection confidence is the one signal
    every detector emits, so scoring on it behaves identically with or
    without a classifier and needs no knowledge of what the categories
    mean.

    `best_frame_dest_dir` is the directory the parent expects the chosen
    frame JPEG to land in. None means "skip the best-frame write" (only
    happens if the parent didn't supply a destination).
    """
    # Group items by their frame_number so each yielded frame can serve
    # multiple bboxes in one pass.
    items_by_frame: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for orig_idx, item in video_items:
        items_by_frame[int(item["frame_number"])].append((orig_idx, item))

    # Nothing to classify on this clip, so the thumbnail is the only
    # reason to touch it: go straight to that one frame instead of
    # walking to it. Worth having because this is the common case, not a
    # corner: it covers every empty clip (most of a deployment) and
    # every clip holding only people or vehicles, none of which are
    # classified. A clip that *does* have crops keeps walking, because
    # it wants 11 to 51 frames and seeking each of those is slower.
    #
    # `results` and `per_crop_progress` are only ever driven by
    # `video_items`, which is empty here, so there is no bookkeeping to
    # do. Any failure falls through to the walk below unchanged.
    if not items_by_frame:
        cap = open_video(video_path)
        if cap is not None:
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                best_frame_number = choose_frame_number(
                    scoring_dets, total_frames
                )
                pixels = read_frame_by_seek(
                    cap, best_frame_number, total_frames
                )
            finally:
                cap.release()
            if pixels is not None:
                # Decode even when there is nowhere to write it: the
                # decode is what proves the number, and `json_pipeline`
                # builds `best_frame_path` from it without checking that
                # the file exists.
                if best_frame_dest_dir is not None:
                    dest = (
                        best_frame_dest_dir
                        / f"frame{best_frame_number:06d}.jpg"
                    )
                    try:
                        write_best_frame(pixels, dest)
                    except Exception as e:
                        print(
                            f"[Worker] Failed to write best frame for "
                            f"{video_path}: {e}",
                            file=sys.stderr, flush=True,
                        )
                return best_frame_number

    cap = open_video(video_path)
    if cap is None:
        # Best-frame scoring also impossible. All items for this video
        # report failure so callers can surface them.
        for orig_idx, _ in video_items:
            results[orig_idx] = {
                "success": False,
                "error": f"Failed to open video: {video_path}",
            }
        if batched_state is None:
            per_crop_progress(len(video_items))
        return None

    try:
        # Decide the frame before decoding anything: the JSON already
        # holds every input the choice needs. That is what keeps this
        # loop down to two retained images instead of one per candidate.
        best_frame_number = choose_frame_number(
            scoring_dets, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        # Frames to decode: everything we classify on, the frame we
        # intend to keep, and frame 0 as the insurance policy. A
        # container can advertise more frames than it yields, so the
        # chosen frame may never arrive; frame 0 always does if the
        # video opened at all.
        wanted = set(items_by_frame.keys()) | {best_frame_number, 0}

        chosen_pixels: Image.Image | None = None
        first_pixels: Image.Image | None = None

        for frame_num, pil_image in iter_wanted_frames(cap, wanted, video_path):
            # Classify every item pinned to this frame.
            for orig_idx, item in items_by_frame.get(frame_num, []):
                _classify_one(
                    model_inference,
                    pil_image,
                    item["bbox"],
                    batched_state,
                    orig_idx,
                    results,
                )
                if batched_state is None:
                    per_crop_progress(1)

            if frame_num == best_frame_number:
                chosen_pixels = pil_image
            if frame_num == 0:
                first_pixels = pil_image

        # In batch mode, flush leftover crops before we finish (so
        # failures in classify_batch propagate before we report success).
        if batched_state is not None:
            batched_state["flush_batch"]()
    finally:
        cap.release()

    if chosen_pixels is None:
        # The frame we picked never decoded. Fall back to frame 0 and
        # move `best_frame_number` with it: the number and the JPEG must
        # describe the same moment, or the Labels grid draws one frame's
        # boxes over another frame's picture.
        if first_pixels is None:
            return None
        best_frame_number, chosen_pixels = 0, first_pixels

    if best_frame_dest_dir is not None:
        dest = best_frame_dest_dir / f"frame{best_frame_number:06d}.jpg"
        try:
            write_best_frame(chosen_pixels, dest)
        except Exception as e:
            print(
                f"[Worker] Failed to write best frame for {video_path}: {e}",
                file=sys.stderr, flush=True,
            )
    return best_frame_number


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


def _run_items(
    model_inference,
    items: list[dict],
    best_frame_outputs: dict[str, str],
    scoring_detections: dict[str, list[dict]],
    batch_size: int | None,
    emit_fn,
) -> tuple[list, dict[str, int]]:
    """
    Classify all items and return (per-item results, best_frames map).

    Groups items by source (image vs video), iterates each group, fuses
    best-frame scoring into the video pass.
    """
    total = len(items)
    results: list = [None] * total

    # Choose strategy once based on the model's capabilities.
    supports_batching = (
        hasattr(model_inference, "get_tensor")
        and callable(model_inference.get_tensor)
        and hasattr(model_inference, "classify_batch")
        and callable(model_inference.classify_batch)
    )

    if supports_batching:
        gpu_available = model_inference.check_gpu()
        effective_batch_size = batch_size if batch_size is not None else (
            8 if gpu_available else 1
        )
        print(
            f"[Worker] Using batch inference (batch_size={effective_batch_size}, "
            f"device={'GPU' if gpu_available else 'CPU'})",
            file=sys.stderr, flush=True,
        )
        batched_state = _make_batched_state(
            model_inference, effective_batch_size, items, results, emit_fn
        )

        def per_crop_progress(_n: int) -> None:  # not used in batched mode
            return
    else:
        print(
            "[Worker] Using per-crop inference (no batch support)",
            file=sys.stderr, flush=True,
        )
        batched_state = None
        processed = {"n": 0}

        def per_crop_progress(n: int) -> None:
            processed["n"] += n
            if processed["n"] % 5 == 0 or processed["n"] == total:
                emit_fn({"current": processed["n"], "total": total})

    # Group items by (source, key).
    images: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    videos: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, item in enumerate(items):
        source = item.get("source", "image")
        if source == "video":
            videos[item["video_path"]].append((i, item))
        else:
            images[item["image_path"]].append((i, item))

    # Best-frame is also picked for videos that have NO detections (we
    # still want a thumbnail). Include them as empty groups so the video
    # processor sees them.
    for video_path in best_frame_outputs:
        if video_path not in videos:
            videos[video_path] = []

    best_frames: dict[str, int] = {}

    for image_path, image_items in images.items():
        _process_image_group(
            model_inference, image_path, image_items, results,
            batched_state, per_crop_progress,
        )

    for video_path, video_items in videos.items():
        # Every crop queued for classification on this video is itself one of
        # that video's detections, so an empty scoring list alongside a
        # non-empty item list cannot happen in correct operation. It means the
        # two maps were built from different data, or keyed differently: both
        # are keyed by a resolved absolute path built independently in the
        # parent, which is exactly the kind of agreement that rots quietly.
        scoring_dets = scoring_detections.get(video_path, [])
        if video_items and not scoring_dets:
            raise RuntimeError(
                f"{len(video_items)} detection(s) queued for classification "
                f"on {video_path}, but no scoring detections for it. Those "
                f"crops are themselves detections, so this list cannot be "
                f"empty unless the caller built the two maps inconsistently."
            )

        dest_dir = best_frame_outputs.get(video_path)
        best_frame_number = _process_video_group(
            model_inference,
            video_path,
            video_items,
            scoring_dets,
            Path(dest_dir) if dest_dir else None,
            results,
            batched_state,
            per_crop_progress,
        )
        if best_frame_number is not None:
            best_frames[video_path] = best_frame_number

    # Flush any leftover batch crops from the last video.
    if batched_state is not None:
        batched_state["flush_batch"]()

    return results, best_frames


def main() -> None:
    """One-shot batch classification: load model, classify all items, write results, exit."""
    if len(sys.argv) != 5:
        print(
            f"Usage: {sys.argv[0]} <model_dir> <model_path> <input_json> <output_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    input_json = Path(sys.argv[3])
    output_json = Path(sys.argv[4])

    try:
        model_inference = load_inference_class(model_dir, model_path)
        validate_interface(model_inference)

        gpu_available = model_inference.check_gpu()
        model_inference.load_model()
        device_name = detect_device_name(gpu_available)

        emit({"status": "ready", "compute_device": device_name})

    except Exception as e:
        print(f"[Worker] Fatal error during startup: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_json) as f:
            data = json.load(f)

        items = data["items"]
        best_frame_outputs: dict[str, str] = data.get("best_frame_outputs", {})

        # A payload with videos but no `scoring_detections` key at all means
        # the caller is older than this worker. That combination is not a
        # degraded run, it is a wrong one: every video would score nothing
        # and take the blank-video fallback, so each clip gets a thumbnail
        # from its middle frame with no detection on it, the Labels grid
        # shows no cards for it, and its species never becomes an
        # observation. It looks like the analysis found nothing rather than
        # like a failure. Classification subprocesses read this script fresh
        # from disk on every spawn while the parent process does not, so a
        # dev backend that has not restarted since an edit lands exactly
        # here.
        if "scoring_detections" not in data and best_frame_outputs:
            raise RuntimeError(
                f"Payload has {len(best_frame_outputs)} video(s) but no "
                f"'scoring_detections' key. The calling process is running "
                f"older code than this worker (restart the backend). "
                f"Best-frame selection would silently fall back to the "
                f"middle frame of every clip."
            )
        scoring_detections: dict[str, list[dict]] = data.get(
            "scoring_detections", {}
        )
        batch_size = data.get("batch_size")
        total = len(items)
        print(
            f"[Worker DEBUG] Read {total} items "
            f"({len(best_frame_outputs)} video best-frame targets) "
            f"from {input_json}",
            file=sys.stderr, flush=True,
        )

        class_names = model_inference.get_class_names()
        print(
            f"[Worker DEBUG] Got {len(class_names)} class names",
            file=sys.stderr, flush=True,
        )

        results, best_frames = _run_items(
            model_inference, items, best_frame_outputs, scoring_detections,
            batch_size, emit,
        )

        success_count = sum(1 for r in results if r and r.get("success"))
        fail_count = total - success_count
        print(
            f"[Worker DEBUG] Done: {success_count} succeeded, {fail_count} failed, "
            f"{len(best_frames)} best frames picked, writing to {output_json}",
            file=sys.stderr, flush=True,
        )

        with open(output_json, "w") as f:
            json.dump(
                {
                    "class_names": class_names,
                    "results": results,
                    "best_frames": best_frames,
                },
                f,
            )

        sys.exit(0)

    except Exception as e:
        print(f"[Worker] Fatal error during classification: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
