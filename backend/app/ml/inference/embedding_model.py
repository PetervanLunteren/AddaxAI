"""
DINOv2 embedding model wrapper — subprocess invocation.

Thin wrapper around embedding_script.py, same pattern as MegaDetector wrapper.
Runs as subprocess in env-addaxai-base environment.

Following CONVENTIONS.md: crash early and loudly, no silent failures.
"""

import re
import subprocess
from collections import deque
from collections.abc import Callable
from pathlib import Path

from app.core.job_cancellation import (
    JobCancelledError,
    is_cancel_requested,
    track_subprocess,
)
from app.core.logging_config import get_logger
from app.core.subprocess_group import popen_group
from app.ml.environment_manager import EnvironmentManager
from app.ml.schemas.model_manifest import ModelManifest
from app.utils.subprocess_env import clean_python_env

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Runs DINOv2 embedding computation via subprocess.

    Wraps embedding_script.py with progress parsing from stderr.
    """

    def __init__(
        self,
        model_path: Path,
        manifest: ModelManifest,
        env_manager: EnvironmentManager,
    ):
        """
        Initialize embedding model.

        Args:
            model_path: Path to .pth weights file
            manifest: Model manifest with embedding_dim, input_size, torch_hub_model
            env_manager: Environment manager for accessing conda environments

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If manifest missing required embedding fields
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not manifest.embedding_dim:
            raise ValueError(f"Manifest missing embedding_dim: {manifest.model_id}")
        if not manifest.input_size:
            raise ValueError(f"Manifest missing input_size: {manifest.model_id}")
        if not manifest.torch_hub_model:
            raise ValueError(f"Manifest missing torch_hub_model: {manifest.model_id}")

        self.model_path = model_path
        self.manifest = manifest
        self.python_path = env_manager.get_python("env-addaxai-base")
        self.script_path = Path(__file__).parent / "embedding_script.py"

        if not self.script_path.exists():
            raise FileNotFoundError(f"Embedding script not found: {self.script_path}")

    def compute_embeddings(
        self,
        input_json_path: Path,
        output_npz_path: Path,
        batch_size: int | None = None,
        progress_callback: Callable[[str, float, dict | None], None] | None = None,
        job_id: str | None = None,
    ) -> int:
        """
        Run embedding subprocess. Returns number of embeddings computed.

        Parses tqdm output from stderr for progress (same pattern as MegaDetector).

        Args:
            input_json_path: Path to input JSON with detection list
            output_npz_path: Path for output .npz file
            batch_size: Number of crops processed per batch. None means let the
                embedding script auto-select based on its own GPU detection.
                A non-None integer is the user's Custom override.
            progress_callback: Optional callback(message, phase_progress, metrics)

        Returns:
            Number of embeddings computed

        Raises:
            RuntimeError: If subprocess fails
            FileNotFoundError: If input JSON doesn't exist
        """
        if not input_json_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

        cmd = [
            str(self.python_path),
            str(self.script_path),
            "--input",
            str(input_json_path),
            "--output",
            str(output_npz_path),
            "--weights",
            str(self.model_path),
            "--model-arch",
            self.manifest.torch_hub_model,
            "--embedding-dim",
            str(self.manifest.embedding_dim),
            "--input-size",
            str(self.manifest.input_size),
        ]

        # Only override batch size when the user explicitly set a Custom
        # value. None = let the script auto-select (GPU=64, MPS=32, CPU=8).
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])

        logger.info(f"Running embedding: {' '.join(cmd)}")

        if progress_callback:
            progress_callback("Loading embedding model...", 0.0, None)


        env = clean_python_env(PYTHONUNBUFFERED="1")

        process = popen_group(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        embedded_count = 0
        # Keep a rolling tail of stderr so a non-zero exit produces a
        # useful error. The progress-parsing loop below consumes stderr
        # line by line and would otherwise discard the traceback that
        # caused the crash; without this buffer, error messages look like
        # "Embedding script failed (exit code 1): ... Stdout: " with no
        # actual cause attached.
        stderr_tail: deque[str] = deque(maxlen=200)

        with track_subprocess(job_id, process):
            # Parse stderr for tqdm progress and compute device
            for raw_line in iter(process.stderr.readline, ""):
                if not raw_line:
                    break
                stderr_tail.append(raw_line)

                line = raw_line.strip()
                if not line:
                    continue

                logger.debug(f"embedding: {line}")

                # Parse compute device (printed once during init)
                if line.startswith("COMPUTE_DEVICE:") and progress_callback:
                    device_type = line.split(":", 1)[1].strip()
                    device_name = _format_device_name(device_type)
                    progress_callback(
                        "Computing embeddings...",
                        0.0,
                        {"compute_device": device_name},
                    )
                    continue

                # Parse tqdm progress
                if progress_callback and "%" in line:
                    metrics = _parse_tqdm_metrics(line)
                    if metrics:
                        progress = (
                            metrics["current"] / metrics["total"] if metrics["total"] > 0 else 0.0
                        )
                        progress_callback(
                            metrics.get("raw_line", line[:80]),
                            progress,
                            metrics,
                        )
                        embedded_count = metrics["current"]

                # Parse final count from "Saved N embeddings" line
                if "Saved" in line and "embeddings" in line:
                    match = re.search(r"Saved (\d+) embeddings", line)
                    if match:
                        embedded_count = int(match.group(1))

            process.stderr.close()
            stdout, _ = process.communicate()

        if job_id and is_cancel_requested(job_id):
            raise JobCancelledError()

        if process.returncode != 0:
            stderr_text = "".join(stderr_tail).rstrip()
            raise RuntimeError(
                f"Embedding script failed (exit code {process.returncode}):\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stdout: {stdout}\n"
                f"Stderr (last {len(stderr_tail)} lines):\n{stderr_text}"
            )

        logger.info(f"Embedding complete: {embedded_count} detections embedded")
        return embedded_count


def _format_device_name(device_type: str) -> str:
    """Convert device type string to user-friendly name."""
    device_type = device_type.lower()
    if "mps" in device_type:
        return "GPU (Apple Silicon)"
    if "cuda" in device_type:
        return "GPU (NVIDIA)"
    return "CPU"


def _parse_tqdm_metrics(line: str) -> dict | None:
    """
    Parse tqdm metrics from stderr line.

    Expected format: "N%|...| current/total [elapsed<remaining, rate unit/s]"

    Returns:
        Dict with {current, total, elapsed, remaining, rate, unit, raw_line}
        or None if parsing fails.
    """
    try:
        metrics: dict = {"raw_line": line}

        # Extract current/total
        progress_match = re.search(r"(\d+)/(\d+)", line)
        if progress_match:
            metrics["current"] = int(progress_match.group(1))
            metrics["total"] = int(progress_match.group(2))
        else:
            return None

        # Extract rate: "N.Ncrop/s" or "Ns/crop"
        rate_match = re.search(r"(\d+\.?\d*)([\w]+)/s", line)
        if rate_match:
            metrics["rate"] = float(rate_match.group(1))
            metrics["unit"] = rate_match.group(2)
        else:
            inverse_match = re.search(r"(\d+\.?\d*)s/([\w]+)", line)
            if inverse_match:
                time_per_item = float(inverse_match.group(1))
                if time_per_item > 0:
                    metrics["rate"] = 1.0 / time_per_item
                    metrics["unit"] = inverse_match.group(2)

        # Extract elapsed time
        time_match = re.search(r"\[(\d{2}:\d{2}(?::\d{2})?)<", line)
        if time_match:
            metrics["elapsed"] = time_match.group(1)

        # Extract remaining time
        remaining_match = re.search(r"<(\d{2}:\d{2}(?::\d{2})?)", line)
        if remaining_match:
            metrics["remaining"] = remaining_match.group(1)

        return metrics

    except (ValueError, IndexError, AttributeError):
        return None
