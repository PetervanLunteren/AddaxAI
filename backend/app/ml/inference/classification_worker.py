"""
One-shot batch classification worker for subprocess execution.

Usage: python classification_worker.py <model_dir> <model_path> <input_json> <output_json>

Reads all detections from input_json, classifies them, writes results to output_json.
Progress and status are streamed via stderr as JSON lines.

Supports two modes:
- Batch mode: if the model implements get_tensor() + classify_batch(),
  crops are grouped by image, preprocessed, stacked into batches, and
  processed in one GPU forward pass per batch. Much faster.
- Per-crop mode (fallback): calls get_classification() one crop at a time.
  Works with any model but slower on GPU.

Created by Claude Code on 2026-01-05
Updated on 2026-03-14 - Simplified from persistent worker to one-shot batch
Updated on 2026-03-26 - Added image caching and batch inference support
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

import numpy as np
from PIL import Image


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

    Args:
        model_dir: Path to model directory
        model_path: Path to main model file

    Returns:
        Instantiated ModelInference object

    Raises:
        ImportError: If module loading fails
        AttributeError: If ModelInference class not found
    """
    inference_script = model_dir / "inference.py"

    if not inference_script.exists():
        raise FileNotFoundError(
            f"Custom inference script not found: {inference_script}\n"
            f"Model developers must provide inference.py in their model directory."
        )

    # Create unique module name
    module_name = f"custom_inference_{model_dir.name}"

    # Load module from file
    spec = importlib.util.spec_from_file_location(module_name, inference_script)

    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {inference_script}")

    module = importlib.util.module_from_spec(spec)

    # Add to sys.modules so imports within inference.py work
    sys.modules[module_name] = module

    # Execute module
    spec.loader.exec_module(module)

    # Instantiate ModelInference class
    if not hasattr(module, "ModelInference"):
        raise AttributeError(
            "inference.py must define a 'ModelInference' class.\n"
            "See /backend/templates/inference_template.py for reference."
        )

    model_inference = module.ModelInference(model_dir, model_path)
    return model_inference


def validate_interface(model_inference):
    """
    Validate that ModelInference instance provides required methods.

    Args:
        model_inference: ModelInference instance

    Raises:
        ValueError: If required methods are missing
    """
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

    # Validate methods are callable
    for method_name in required_methods:
        if not callable(getattr(model_inference, method_name)):
            raise ValueError(f"Required attribute '{method_name}' is not callable")


def detect_device_name(gpu_available: bool) -> str:
    """Detect friendly device name from ML frameworks loaded in this process."""
    if not gpu_available:
        return "CPU"
    # Check PyTorch
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "GPU (Apple Silicon)"
        if torch.cuda.is_available():
            return "GPU (NVIDIA)"
    except ImportError:
        pass
    # Check TensorFlow
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


def _classify_batched(model_inference, items, batch_size, emit_fn):
    """
    Classify items using batch inference with image caching.

    Groups items by image path (open each image once), preprocesses
    crops via get_tensor(), stacks into batches, and runs batch
    inference via classify_batch().

    Args:
        model_inference: ModelInference instance with get_tensor() and classify_batch()
        items: List of {"image_path": str, "bbox": [x, y, w, h]} dicts
        batch_size: Number of crops per batch. Resolved by the parent backend
            process from the project's classification_batch_size override (or
            the per-pipeline default) and passed in via the input JSON.
        emit_fn: Callable to emit progress updates

    Returns:
        List of result dicts (parallel to items)
    """
    total = len(items)

    # Group items by image path for caching (preserve original indices)
    items_by_image: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, item in enumerate(items):
        items_by_image[item["image_path"]].append((i, item))

    results: list[dict | None] = [None] * total
    processed = 0
    batch_indices: list[int] = []
    batch_tensors: list[np.ndarray] = []

    def flush_batch():
        nonlocal processed
        if not batch_tensors:
            return
        batch = np.stack(batch_tensors)
        batch_results = model_inference.classify_batch(batch)
        for idx, classifications in zip(batch_indices, batch_results):  # noqa: B905 (Python 3.8 compat)
            if _has_nonfinite_confidence(classifications):
                # Numerically-unstable model output (e.g. softmax of NaN
                # logits on a degenerate crop). Log the offending item
                # so it can be investigated, then mark this row failed
                # so it loads as unclassified rather than crashing the
                # batch.
                src = items[idx]
                print(
                    f"[Worker] Non-finite confidence from classify_batch "
                    f"for image={src.get('image_path')!r} "
                    f"bbox={src.get('bbox')!r}",
                    file=sys.stderr, flush=True,
                )
                results[idx] = {
                    "success": False,
                    "error": "Model produced non-finite confidence (NaN/inf)",
                }
                continue
            sorted_cls = sorted(classifications, key=lambda x: x[1], reverse=True)
            results[idx] = {"success": True, "classifications": sorted_cls}
        processed += len(batch_indices)
        emit_fn({"current": processed, "total": total})
        batch_indices.clear()
        batch_tensors.clear()

    for image_path, image_items in items_by_image.items():
        path = Path(image_path)
        if not path.exists():
            for orig_idx, _ in image_items:
                results[orig_idx] = {
                    "success": False,
                    "error": f"Image not found: {image_path}",
                }
                processed += 1
            continue

        image = Image.open(path)

        for orig_idx, item in image_items:
            try:
                crop = model_inference.get_crop(image, tuple(item["bbox"]))
                if crop is None:
                    results[orig_idx] = {
                        "success": False,
                        "error": f"Invalid crop for bbox {item['bbox']}",
                    }
                    processed += 1
                    continue

                tensor = model_inference.get_tensor(crop)
                batch_tensors.append(tensor)
                batch_indices.append(orig_idx)

                if len(batch_tensors) >= batch_size:
                    flush_batch()

            except Exception as e:
                results[orig_idx] = {"success": False, "error": str(e)}
                processed += 1

    # Flush remaining partial batch
    flush_batch()

    return results


def _classify_per_crop(model_inference, items, emit_fn):
    """
    Classify items one crop at a time (fallback for models without batch support).

    Groups items by image path to avoid re-opening the same image.

    Args:
        model_inference: ModelInference instance
        items: List of {"image_path": str, "bbox": [x, y, w, h]} dicts
        emit_fn: Callable to emit progress updates

    Returns:
        List of result dicts (parallel to items)
    """
    total = len(items)

    # Group items by image path for caching
    items_by_image: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, item in enumerate(items):
        items_by_image[item["image_path"]].append((i, item))

    results: list[dict | None] = [None] * total
    processed = 0

    for image_path, image_items in items_by_image.items():
        path = Path(image_path)
        if not path.exists():
            for orig_idx, _ in image_items:
                results[orig_idx] = {
                    "success": False,
                    "error": f"Image not found: {image_path}",
                }
                processed += 1
            continue

        image = Image.open(path)

        for orig_idx, item in image_items:
            try:
                crop = model_inference.get_crop(image, tuple(item["bbox"]))
                if crop is None:
                    results[orig_idx] = {
                        "success": False,
                        "error": f"Invalid crop for bbox {item['bbox']}",
                    }
                    processed += 1
                    continue

                classifications = model_inference.get_classification(crop)
                if not classifications:
                    results[orig_idx] = {
                        "success": False,
                        "error": f"Empty result for bbox {item['bbox']}",
                    }
                    processed += 1
                    continue

                if _has_nonfinite_confidence(classifications):
                    print(
                        f"[Worker] Non-finite confidence from "
                        f"get_classification for image={image_path!r} "
                        f"bbox={item['bbox']!r}",
                        file=sys.stderr, flush=True,
                    )
                    results[orig_idx] = {
                        "success": False,
                        "error": "Model produced non-finite confidence (NaN/inf)",
                    }
                    processed += 1
                    continue

                sorted_cls = sorted(
                    classifications, key=lambda x: x[1], reverse=True
                )
                results[orig_idx] = {
                    "success": True,
                    "classifications": sorted_cls,
                }

            except Exception as e:
                results[orig_idx] = {"success": False, "error": str(e)}

            processed += 1
            if processed % 5 == 0 or processed == total:
                emit_fn({"current": processed, "total": total})

    return results


def main():
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
        # Load and instantiate ModelInference class
        model_inference = load_inference_class(model_dir, model_path)
        validate_interface(model_inference)

        # Check GPU and load model
        gpu_available = model_inference.check_gpu()
        model_inference.load_model()
        device_name = detect_device_name(gpu_available)

        # Signal ready with device info
        emit({"status": "ready", "compute_device": device_name})

    except Exception as e:
        print(f"[Worker] Fatal error during startup: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        # Read input
        with open(input_json) as f:
            data = json.load(f)

        items = data["items"]
        # batch_size comes from the project's Custom override when set,
        # or is absent when the user left it at Default. When absent,
        # fall back to the subprocess's own GPU-aware default.
        batch_size = data.get("batch_size")
        total = len(items)
        print(f"[Worker DEBUG] Read {total} items from {input_json}", file=sys.stderr, flush=True)

        # Get class names
        class_names = model_inference.get_class_names()
        print(f"[Worker DEBUG] Got {len(class_names)} class names", file=sys.stderr, flush=True)

        # Choose classification strategy
        supports_batching = (
            hasattr(model_inference, "get_tensor")
            and callable(model_inference.get_tensor)
            and hasattr(model_inference, "classify_batch")
            and callable(model_inference.classify_batch)
        )

        if supports_batching:
            gpu_available = model_inference.check_gpu()
            # Use the Custom override if provided, otherwise auto-detect
            effective_batch_size = batch_size if batch_size is not None else (
                8 if gpu_available else 1
            )
            print(
                f"[Worker] Using batch inference (batch_size={effective_batch_size}, "
                f"device={'GPU' if gpu_available else 'CPU'})",
                file=sys.stderr, flush=True,
            )
            results = _classify_batched(model_inference, items, effective_batch_size, emit)
        else:
            print(
                "[Worker] Using per-crop inference (no batch support)",
                file=sys.stderr, flush=True,
            )
            results = _classify_per_crop(model_inference, items, emit)

        # Write output
        success_count = sum(1 for r in results if r and r.get("success"))
        fail_count = total - success_count
        print(
            f"[Worker DEBUG] Done: {success_count} succeeded, {fail_count} failed, "
            f"writing to {output_json}",
            file=sys.stderr, flush=True,
        )

        with open(output_json, "w") as f:
            json.dump({"class_names": class_names, "results": results}, f)

        sys.exit(0)

    except Exception as e:
        print(f"[Worker] Fatal error during classification: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
