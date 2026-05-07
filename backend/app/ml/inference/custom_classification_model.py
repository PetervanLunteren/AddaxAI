"""
Custom Classification Model with One-Shot Batch Subprocess.

Runs classification_worker.py as a one-shot subprocess: parent writes all
detections to a temp JSON file, subprocess classifies everything, writes
results to output JSON, exits. No persistent process, no stdin/stdout protocol.

Created by Claude Code on 2026-01-05
Updated on 2026-03-14 - Simplified from persistent worker to one-shot batch
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import tempfile
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
from app.ml.inference.base import ClassificationResult

logger = get_logger(__name__)


class CustomClassificationModel:
    """
    Classification model that runs a one-shot batch subprocess.

    Architecture:
    1. Write all detections to temp input JSON
    2. Launch classification_worker.py (loads model, classifies all, writes output, exits)
    3. Read results from output JSON
    4. Progress streamed via stderr JSON lines
    """

    def __init__(
        self,
        model_dir: Path,
        model_path: Path,
        env_name: str,
        env_manager: EnvironmentManager,
    ):
        self.model_dir = model_dir
        self.model_path = model_path
        self.env_name = env_name

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
            raise RuntimeError(f"Failed to get Python environment '{env_full_name}': {e}") from e

        # Get path to classification_worker.py script
        self.worker_script = Path(__file__).parent / "classification_worker.py"
        if not self.worker_script.exists():
            raise FileNotFoundError(f"Classification worker script not found: {self.worker_script}")

        logger.info(f"Custom classification model initialized: {model_dir.name}")

    def classify_detections(
        self,
        items: list[dict],
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        job_id: str | None = None,
    ) -> tuple[list[ClassificationResult | None], dict[str, str], str]:
        """
        Classify a batch of detections in a one-shot subprocess.

        Args:
            items: List of {"image_path": str, "bbox": [x, y, w, h]} dicts
            batch_size: Number of crops processed per batch. None means let the
                classification worker use its own default (auto-detects GPU).
                A non-None integer is the user's Custom override.
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Tuple of (results, class_names, compute_device) where:
            - results: List parallel to items, ClassificationResult or None per item
            - class_names: Dict mapping class ID (str) to class name (str)
            - compute_device: Friendly device name (e.g., "GPU (Apple Silicon)")

        Raises:
            RuntimeError: If subprocess fails to start or crashes
        """
        input_file = None
        output_file = None

        try:
            # Write input JSON
            input_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="cls_input_", delete=False
            )
            payload: dict = {"items": items}
            if batch_size is not None:
                payload["batch_size"] = batch_size
            json.dump(payload, input_file)
            input_file.close()
            logger.info(
                f"[DEBUG] Wrote {len(items)} items (batch_size={batch_size}) "
                f"to input file: {input_file.name}"
            )

            # Create output file path
            output_fd, output_path = tempfile.mkstemp(suffix=".json", prefix="cls_output_")
            os.close(output_fd)
            output_file = output_path

            # Build command
            cmd = [
                str(self.python_path),
                str(self.worker_script),
                str(self.model_dir),
                str(self.model_path),
                input_file.name,
                output_file,
            ]

            # Prepare environment
            env = os.environ.copy()
            if platform.system() == "Darwin":
                env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            logger.info(f"Starting one-shot classification worker for {self.model_dir.name}")
            logger.info(f"[DEBUG] Command: {' '.join(cmd)}")

            # Launch subprocess in its own process group so cancel can
            # kill the whole tree; stderr is used for progress messages.
            process = popen_group(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            compute_device = "CPU"
            # Watch worker output for the signature of a corrupted env:
            # the interpreter exits before any user code runs because
            # `Lib/encodings/` is missing or unreadable. Most often
            # caused by Windows Defender quarantining files under
            # `~/AddaxAI/envs/` between sessions. Without this signal
            # we'd surface a useless "exited with code 1" to the user.
            env_corrupted = False
            CORRUPTED_ENV_MARKERS = (
                "init_fs_encoding",
                "No module named 'encodings'",
            )
            with track_subprocess(job_id, process):
                # Read stderr line by line for progress
                for line in process.stderr:
                    line = line.strip()
                    if not line:
                        continue

                    if any(m in line for m in CORRUPTED_ENV_MARKERS):
                        env_corrupted = True

                    # Try parsing as JSON status/progress
                    try:
                        msg = json.loads(line)
                        if "status" in msg and msg["status"] == "ready":
                            compute_device = msg.get("compute_device", "CPU")
                            logger.info(
                                f"Worker ready (Device: {compute_device}) "
                                f"for {self.model_dir.name}"
                            )
                        elif "current" in msg and "total" in msg:
                            if progress_callback:
                                progress_callback(msg["current"], msg["total"])
                    except json.JSONDecodeError:
                        # Regular log line from worker
                        logger.info(f"[Worker] {line}")
                        # TODO: remove this print after verifying batching works in production
                        print(f"[Worker] {line}")

                # Wait for process to finish
                process.wait()

            logger.info(f"[DEBUG] Worker exited with code {process.returncode}")

            if job_id and is_cancel_requested(job_id):
                raise JobCancelledError()

            if process.returncode != 0:
                if env_corrupted:
                    raise RuntimeError(
                        f"The analysis environment 'env-{self.env_name}' is "
                        f"corrupted (its Python stdlib is missing). This "
                        f"usually means antivirus or system cleanup removed "
                        f"files under your AddaxAI folder. Restart AddaxAI: "
                        f"it will detect the broken environment and prompt "
                        f"you to rebuild it. If that does not help, reinstall "
                        f"AddaxAI."
                    )
                raise RuntimeError(
                    f"Classification worker exited with code {process.returncode} "
                    f"for model {self.model_dir.name}"
                )

            # Read output JSON
            with open(output_file) as f:
                output_data = json.load(f)

            class_names = output_data["class_names"]
            raw_results = output_data["results"]
            logger.info(
                f"[DEBUG] Output: {len(raw_results)} results, "
                f"{len(class_names)} class names, device={compute_device}"
            )

            # Convert to ClassificationResult objects
            results: list[ClassificationResult | None] = []
            for i, raw in enumerate(raw_results):
                if not raw.get("success"):
                    logger.warning(f"Classification skipped: {raw.get('error', 'unknown')}")
                    results.append(None)
                    continue

                classifications = raw["classifications"]
                if not classifications:
                    results.append(None)
                    continue

                # Build all_probabilities dict and extract top prediction
                all_probs = {name: conf for name, conf in classifications}
                top_label, top_confidence = classifications[0]

                # NaN/inf confidences leak out of numerically-unstable model
                # output (e.g. softmax of all-NaN logits on a degenerate
                # crop). The strict ClassificationResult validator would
                # crash the whole batch job, so skip this row and treat it
                # as unclassified, same as the no-classifications branch
                # above.
                if not math.isfinite(top_confidence):
                    item_path = (
                        items[i].get("image_path", "<unknown>")
                        if i < len(items) else "<unknown>"
                    )
                    logger.warning(
                        f"Classification produced non-finite confidence "
                        f"({top_confidence}) for {item_path}, skipping"
                    )
                    results.append(None)
                    continue

                results.append(ClassificationResult(
                    label=top_label,
                    confidence=top_confidence,
                    all_probabilities=all_probs,
                ))

            success_count = sum(1 for r in results if r is not None)
            logger.info(
                f"[DEBUG] Returning {success_count}/{len(results)} successful classifications"
            )
            return results, class_names, compute_device

        finally:
            # Clean up temp files
            if input_file is not None:
                try:
                    os.unlink(input_file.name)
                except OSError:
                    pass
            if output_file is not None:
                try:
                    os.unlink(output_file)
                except OSError:
                    pass
