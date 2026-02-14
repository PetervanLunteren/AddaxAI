"""
Persistent classification worker for subprocess execution.

This worker process runs in the model's designated environment and maintains
a loaded model in memory to process multiple classification requests efficiently.

The worker loads the model's inference.py file, instantiates the ModelInference
class, and calls its methods to perform classifications.

Communication via stdin/stdout using JSON:
- Input: {"command": "classify", "image_path": "...", "bbox": [x,y,w,h]}
- Output: {"success": true, "classifications": [["species", conf], ...]}
- Input: {"command": "get_class_names"}
- Output: {"success": true, "class_names": {"1": "species1", ...}}
- Shutdown: {"command": "stop"} → {"status": "stopped"}

Created by Claude Code on 2026-01-05
Updated on 2026-01-13 - Migrated to class-based interface
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
    if not hasattr(module, 'ModelInference'):
        raise AttributeError(
            f"inference.py must define a 'ModelInference' class.\n"
            f"See /backend/templates/inference_template.py for reference."
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
    required_methods = ["check_gpu", "load_model", "get_crop", "get_classification", "get_class_names"]

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


def send_response(data: dict) -> None:
    """
    Send JSON response to stdout.

    Args:
        data: Dictionary to send as JSON
    """
    json_str = json.dumps(data)
    print(json_str, flush=True)
    sys.stdout.flush()


def detect_device_name(gpu_available: bool) -> str:
    """Detect friendly device name from ML frameworks loaded in this process."""
    if not gpu_available:
        return "CPU"
    # Check PyTorch
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "GPU (Apple Silicon)"
        if torch.cuda.is_available():
            return "GPU (NVIDIA)"
    except ImportError:
        pass
    # Check TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            if platform.system() == 'Darwin':
                return "GPU (Apple Silicon)"
            return "GPU (NVIDIA)"
    except ImportError:
        pass
    return "GPU"


def main():
    """Main worker loop."""
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} <model_dir> <model_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    model_path = Path(sys.argv[2])

    try:
        # Load and instantiate ModelInference class
        model_inference = load_inference_class(model_dir, model_path)

        # Validate interface
        validate_interface(model_inference)

        # Check GPU
        gpu_available = model_inference.check_gpu()

        # Load model (ONCE)
        model_inference.load_model()

        # Detect device name from loaded framework
        device_name = detect_device_name(gpu_available)

        # Send ready signal FIRST (before any stderr logging to avoid deadlock)
        send_response({"status": "ready", "gpu_available": gpu_available, "compute_device": device_name})

        # Now safe to log to stderr (after ready signal sent)
        print(f"[Worker] GPU available: {gpu_available}, Device: {device_name}", file=sys.stderr, flush=True)
        print("[Worker] Model loaded and ready", file=sys.stderr, flush=True)
        print("[Worker] Entering request loop", file=sys.stderr, flush=True)
        while True:
            try:
                # Read command from stdin
                line = sys.stdin.readline()
                if not line:
                    # EOF - parent process closed pipe
                    print("[Worker] EOF detected, shutting down", file=sys.stderr)
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse command
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    send_response(
                        {
                            "success": False,
                            "error": f"Invalid JSON: {e}",
                            "error_type": "JSONDecodeError",
                        }
                    )
                    continue

                command = request.get("command")

                if command == "stop":
                    print("[Worker] Stop command received", file=sys.stderr)
                    send_response({"status": "stopped"})
                    break

                elif command == "classify":
                    # Extract parameters
                    image_path = Path(request["image_path"])
                    bbox = tuple(request["bbox"])

                    if len(bbox) != 4:
                        send_response(
                            {
                                "success": False,
                                "error": f"Invalid bbox length: {len(bbox)}, expected 4",
                                "error_type": "ValueError",
                            }
                        )
                        continue

                    # Load image
                    if not image_path.exists():
                        send_response(
                            {
                                "success": False,
                                "error": f"Image not found: {image_path}",
                                "error_type": "FileNotFoundError",
                            }
                        )
                        continue

                    image = Image.open(image_path)

                    # Get crop
                    crop = model_inference.get_crop(image, bbox)

                    # Check if crop is valid
                    if crop is None:
                        print(
                            f"[Worker] Invalid crop for bbox {bbox} on image {image_path.name} "
                            f"(too small or out of bounds)",
                            file=sys.stderr
                        )
                        send_response(
                            {
                                "success": False,
                                "error": f"Invalid crop for bbox {bbox} (too small or out of bounds)",
                                "error_type": "CropError",
                            }
                        )
                        continue

                    # Run classification
                    results = model_inference.get_classification(crop)

                    # Check if results are empty
                    if not results:
                        print(
                            f"[Worker] Empty classification results for bbox {bbox} on image {image_path.name}",
                            file=sys.stderr
                        )
                        send_response(
                            {
                                "success": False,
                                "error": f"Classification returned empty results for bbox {bbox}",
                                "error_type": "EmptyClassification",
                            }
                        )
                        continue

                    # Sort by confidence descending (so parent always gets highest confidence first)
                    # This way model developers don't need to duplicate sorting logic in each inference.py
                    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

                    # Send results
                    send_response(
                        {
                            "success": True,
                            "classifications": sorted_results,
                        }
                    )

                elif command == "get_class_names":
                    # Get class names from model
                    class_names = model_inference.get_class_names()

                    # Send results
                    send_response(
                        {
                            "success": True,
                            "class_names": class_names,
                        }
                    )

                else:
                    send_response(
                        {
                            "success": False,
                            "error": f"Unknown command: {command}",
                            "error_type": "ValueError",
                        }
                    )

            except Exception as e:
                # Classification error - send error but keep worker alive
                print(f"[Worker] Classification error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                send_response(
                    {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

        print("[Worker] Exiting cleanly", file=sys.stderr)
        sys.exit(0)

    except Exception as e:
        # Startup error - worker cannot continue
        print(f"[Worker] Fatal error during startup: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        send_response(
            {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "fatal": True,
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
