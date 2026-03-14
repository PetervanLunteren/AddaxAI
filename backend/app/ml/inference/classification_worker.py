"""
One-shot batch classification worker for subprocess execution.

Usage: python classification_worker.py <model_dir> <model_path> <input_json> <output_json>

Reads all detections from input_json, classifies them, writes results to output_json.
Progress and status are streamed via stderr as JSON lines.

Created by Claude Code on 2026-01-05
Updated on 2026-03-14 - Simplified from persistent worker to one-shot batch
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import traceback
from pathlib import Path

from PIL import Image


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
        total = len(items)
        print(f"[Worker DEBUG] Read {total} items from {input_json}", file=sys.stderr, flush=True)

        # Get class names
        class_names = model_inference.get_class_names()
        print(f"[Worker DEBUG] Got {len(class_names)} class names", file=sys.stderr, flush=True)

        # Classify each item
        results = []
        for i, item in enumerate(items):
            try:
                image_path = Path(item["image_path"])
                bbox = tuple(item["bbox"])

                if not image_path.exists():
                    results.append({"success": False, "error": f"Image not found: {image_path}"})
                    continue

                image = Image.open(image_path)
                crop = model_inference.get_crop(image, bbox)

                if crop is None:
                    results.append({
                        "success": False,
                        "error": f"Invalid crop for bbox {bbox} (too small or out of bounds)",
                    })
                    continue

                classifications = model_inference.get_classification(crop)

                if not classifications:
                    results.append({
                        "success": False,
                        "error": f"Classification returned empty results for bbox {bbox}",
                    })
                    continue

                # Sort by confidence descending
                sorted_results = sorted(classifications, key=lambda x: x[1], reverse=True)
                results.append({"success": True, "classifications": sorted_results})

            except Exception as e:
                results.append({"success": False, "error": str(e)})

            # Emit progress periodically (every item — parent decides how to throttle)
            if (i + 1) % 5 == 0 or (i + 1) == total:
                emit({"current": i + 1, "total": total})

        # Write output
        success_count = sum(1 for r in results if r.get("success"))
        fail_count = len(results) - success_count
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
