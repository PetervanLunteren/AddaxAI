"""
Persistent classification worker for subprocess execution.

This worker process runs in the model's designated environment and maintains
a loaded model in memory to process multiple classification requests efficiently.

Communication via stdin/stdout using JSON:
- Input: {"command": "classify", "image_path": "...", "bbox": [x,y,w,h]}
- Output: {"success": true, "classifications": [["species", conf], ...]}
- Input: {"command": "get_class_names"}
- Output: {"success": true, "class_names": {"0": "species1", ...}}
- Shutdown: {"command": "stop"} → {"status": "stopped"}

Created by Claude Code on 2026-01-05
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

from PIL import Image


def load_inference_module(model_dir: Path, model_path: Path):
    """
    Dynamically load the inference.py module from model directory.

    Args:
        model_dir: Path to model directory
        model_path: Path to main model file

    Returns:
        Loaded inference module

    Raises:
        ImportError: If module loading fails
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

    # Execute module (this will initialize MODEL_DIR/MODEL_PATH to None)
    spec.loader.exec_module(module)

    # Inject required variables AFTER execution
    # (must happen after exec_module which resets the module state)
    module.MODEL_DIR = model_dir
    module.MODEL_PATH = model_path

    return module


def validate_interface(module):
    """
    Validate that module provides required functions.

    Args:
        module: Loaded inference module

    Raises:
        ValueError: If required functions are missing
    """
    required_functions = ["check_gpu", "load_model", "get_crop", "get_classification", "get_class_names"]

    missing = [f for f in required_functions if not hasattr(module, f)]

    if missing:
        raise ValueError(
            f"Custom inference script missing required functions: {', '.join(missing)}\n"
            f"Required: {', '.join(required_functions)}"
        )

    # Validate functions are callable
    for func_name in required_functions:
        if not callable(getattr(module, func_name)):
            raise ValueError(f"Required attribute '{func_name}' is not callable")


def send_response(data: dict) -> None:
    """
    Send JSON response to stdout.

    Args:
        data: Dictionary to send as JSON
    """
    json_str = json.dumps(data)
    print(json_str, flush=True)
    sys.stdout.flush()


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
        # Load inference module
        module = load_inference_module(model_dir, model_path)

        # Validate interface
        validate_interface(module)

        # Check GPU
        gpu_available = module.check_gpu()

        # Load model (ONCE)
        module.load_model()

        # Send ready signal FIRST (before any stderr logging to avoid deadlock)
        send_response({"status": "ready", "gpu_available": gpu_available})

        # Now safe to log to stderr (after ready signal sent)
        print(f"[Worker] GPU available: {gpu_available}", file=sys.stderr, flush=True)
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
                    crop = module.get_crop(image, bbox)

                    # Run classification
                    results = module.get_classification(crop)

                    # Send results
                    send_response(
                        {
                            "success": True,
                            "classifications": results,
                        }
                    )

                elif command == "get_class_names":
                    # Get class names from model
                    class_names = module.get_class_names()

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
