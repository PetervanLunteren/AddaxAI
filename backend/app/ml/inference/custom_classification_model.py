"""
Custom Classification Model with Persistent Worker Process.

Manages a long-lived subprocess worker that loads the model once and processes
multiple classifications efficiently. Provides proper environment isolation.

Following DEVELOPERS.md principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere

Created by Claude Code on 2026-01-05
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.base import BoundingBox, ClassificationModel, ClassificationResult

logger = get_logger(__name__)


class CustomClassificationModel(ClassificationModel):
    """
    Classification model that manages a persistent worker process.

    Architecture:
    1. Start worker process in model's environment (loads model once)
    2. Send classification requests via stdin (JSON)
    3. Read results from stdout (JSON)
    4. Stop worker when done (context manager)

    The worker loads the model once and reuses it for all detections,
    dramatically improving performance for models with expensive loading (e.g., Keras).
    """

    def __init__(
        self,
        model_dir: Path,
        model_path: Path,
        env_name: str,
        env_manager: EnvironmentManager,
    ):
        """
        Initialize custom classification model.

        Args:
            model_dir: Path to model directory containing inference.py
            model_path: Path to main model file
            env_name: Environment name from manifest (e.g., "pytorch", "tensorflow-v2")
            env_manager: Environment manager for accessing conda environments

        Raises:
            FileNotFoundError: If inference.py not found
            RuntimeError: If environment setup fails
        """
        self.model_dir = model_dir
        self.model_path = model_path
        self.env_name = env_name
        self.env_manager = env_manager

        # Worker process state
        self.worker_process: subprocess.Popen | None = None
        self.temp_dir: tempfile.TemporaryDirectory | None = None

        # Verify inference.py exists
        inference_script = model_dir / "inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(
                f"Custom inference script not found: {inference_script}\n"
                f"Model developers must provide inference.py in their model directory."
            )

        # Get Python path from designated environment
        try:
            env_full_name = f"env-{env_name}"
            self.python_path = env_manager.get_python(env_full_name)
            logger.info(
                f"Custom classification model ({model_dir.name}) using Python: {self.python_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get Python environment '{env_full_name}': {e}"
            ) from e

        # Get path to classification_worker.py script
        self.worker_script = Path(__file__).parent / "classification_worker.py"
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"Classification worker script not found: {self.worker_script}"
            )

        logger.info(f"Custom classification model initialized: {model_dir.name}")

    def start_worker(self) -> None:
        """
        Start the persistent worker process.

        Launches classification_worker.py in the model's environment,
        waits for "ready" signal, and prepares for classification requests.

        Raises:
            RuntimeError: If worker fails to start
        """
        if self.worker_process is not None:
            logger.warning("Worker already running, skipping start")
            return

        logger.info(f"Starting classification worker for {self.model_dir.name}")

        # Build command
        cmd = [
            str(self.python_path),
            str(self.worker_script),
            str(self.model_dir),
            str(self.model_path),
        ]

        logger.debug(f"Worker command: {' '.join(cmd)}")

        # Start worker process
        try:
            self.worker_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start worker process: {e}") from e

        # Wait for ready signal (with timeout)
        try:
            ready_line = self.worker_process.stdout.readline()
            if not ready_line:
                # Worker died immediately
                stderr = self.worker_process.stderr.read()
                raise RuntimeError(f"Worker died during startup. Stderr: {stderr}")

            ready_response = json.loads(ready_line.strip())

            if ready_response.get("status") != "ready":
                raise RuntimeError(f"Worker sent unexpected response: {ready_response}")

            gpu_available = ready_response.get("gpu_available", False)
            logger.info(
                f"Worker ready (GPU: {gpu_available}) for {self.model_dir.name}"
            )

        except json.JSONDecodeError as e:
            stderr = self.worker_process.stderr.read()
            raise RuntimeError(
                f"Worker sent invalid JSON during startup: {ready_line}\nStderr: {stderr}"
            ) from e
        except Exception as e:
            self._kill_worker()
            raise RuntimeError(f"Worker startup failed: {e}") from e

        # Create temp directory for image files
        self.temp_dir = tempfile.TemporaryDirectory()

    def stop_worker(self) -> None:
        """
        Stop the persistent worker process gracefully.

        Sends stop command and waits for clean shutdown.
        """
        if self.worker_process is None:
            return

        logger.info(f"Stopping classification worker for {self.model_dir.name}")

        try:
            # Send stop command
            stop_cmd = json.dumps({"command": "stop"}) + "\n"
            self.worker_process.stdin.write(stop_cmd)
            self.worker_process.stdin.flush()

            # Wait for response (with timeout)
            try:
                response_line = self.worker_process.stdout.readline()
                if response_line:
                    response = json.loads(response_line.strip())
                    if response.get("status") == "stopped":
                        logger.debug("Worker stopped gracefully")
            except Exception:
                pass  # Best effort

            # Wait for process to exit
            try:
                self.worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Worker didn't stop gracefully, killing")
                self._kill_worker()

        except Exception as e:
            logger.error(f"Error during worker shutdown: {e}")
            self._kill_worker()
        finally:
            self.worker_process = None

            # Clean up temp directory
            if self.temp_dir is not None:
                try:
                    self.temp_dir.cleanup()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory: {e}")
                self.temp_dir = None

    def _kill_worker(self) -> None:
        """Force kill the worker process."""
        if self.worker_process is not None:
            try:
                self.worker_process.kill()
                self.worker_process.wait(timeout=2)
            except Exception:
                pass

    def get_class_names(self) -> dict[str, str]:
        """
        Get class name mapping from the model.

        Sends get_class_names command to worker and returns mapping.

        Returns:
            Dict mapping class ID (str) to class name (str)
            Example: {"0": "aardwolf", "1": "african wild cat", ...}

        Raises:
            RuntimeError: If worker not started or command fails
        """
        if self.worker_process is None:
            raise RuntimeError("Worker not started - call start_worker() first")

        logger.debug(f"Getting class names from {self.model_dir.name}")

        try:
            # Send get_class_names command
            request = {"command": "get_class_names"}
            request_json = json.dumps(request) + "\n"
            self.worker_process.stdin.write(request_json)
            self.worker_process.stdin.flush()

            # Read response
            response_line = self.worker_process.stdout.readline()
            if not response_line:
                stderr = self.worker_process.stderr.read()
                raise RuntimeError(f"Worker died during get_class_names. Stderr: {stderr}")

            response = json.loads(response_line.strip())

            if not response.get("success"):
                error = response.get("error", "Unknown error")
                raise RuntimeError(f"Worker failed to get class names: {error}")

            class_names = response["class_names"]
            logger.debug(f"Retrieved {len(class_names)} class names")

            return class_names

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from worker: {e}") from e
        except Exception as e:
            logger.error(f"get_class_names failed: {e}")
            raise RuntimeError(f"Failed to get class names: {e}") from e

    def classify(
        self,
        image: Image.Image,
        bbox: BoundingBox,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ClassificationResult | None:
        """
        Classify a detection using the persistent worker.

        Workflow:
        1. Save image to temp file
        2. Send classification request to worker via stdin
        3. Read response from worker via stdout
        4. Parse into ClassificationResult

        Args:
            image: PIL Image to classify
            bbox: BoundingBox for the detection (normalized coordinates)
            progress_callback: Optional progress callback (unused)

        Returns:
            ClassificationResult with top species and all probabilities,
            or None if classification fails (best effort mode)

        Raises:
            RuntimeError: If worker is not running
        """
        if self.worker_process is None:
            raise RuntimeError(
                "Worker not started. Use context manager or call start_worker() first."
            )

        if self.temp_dir is None:
            raise RuntimeError("Temp directory not initialized")

        try:
            # Save image to temp file (PNG for lossless preservation)
            temp_path = Path(self.temp_dir.name)
            image_file = temp_path / f"img_{int(time.time() * 1000000)}.png"
            image.save(image_file, "PNG")

            # Build request
            request = {
                "command": "classify",
                "image_path": str(image_file),
                "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
            }

            # Send request
            request_json = json.dumps(request) + "\n"
            self.worker_process.stdin.write(request_json)
            self.worker_process.stdin.flush()

            # Read response (with timeout)
            # Note: readline is blocking, but worker should respond quickly
            response_line = self.worker_process.stdout.readline()

            if not response_line:
                # Worker died
                stderr = self.worker_process.stderr.read()
                logger.error(f"Worker died during classification. Stderr: {stderr}")
                return None

            response = json.loads(response_line.strip())

            # Clean up temp image
            try:
                image_file.unlink()
            except Exception:
                pass

            # Check for errors
            if not response.get("success", False):
                error = response.get("error", "Unknown error")
                error_type = response.get("error_type", "Error")
                logger.warning(
                    f"Classification failed: {error_type}: {error} - Skipping detection"
                )
                return None

            # Parse classifications
            classifications = response["classifications"]

            if not classifications:
                logger.warning("Classification returned empty results - Skipping")
                return None

            # Convert list of tuples to dict
            all_probs_dict = {name: conf for name, conf in classifications}

            # Get top prediction (already sorted by worker)
            top_species, top_confidence = classifications[0]

            # Build ClassificationResult
            result_obj = ClassificationResult(
                species=top_species,
                confidence=top_confidence,
                all_probabilities=all_probs_dict,
            )

            logger.debug(
                f"Classification result: {result_obj.species} ({result_obj.confidence:.3f})"
            )

            return result_obj

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from worker: {e} - Skipping detection")
            return None
        except Exception as e:
            logger.error(f"Classification failed: {e} - Skipping detection", exc_info=True)
            return None

    def __enter__(self):
        """Context manager entry: start worker."""
        self.start_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop worker."""
        self.stop_worker()
        return False
