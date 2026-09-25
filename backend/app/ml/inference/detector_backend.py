"""Manifest-driven detector adapters.

The rest of AddaxAI consumes the JSON dialect historically emitted by
MegaDetector.  This module keeps that contract stable while allowing a model
manifest to select MegaDetector, Ultralytics YOLO, RF-DETR, or RT-DETR. Third-party
model packages are imported only in :mod:`detector_subprocess`; the API and
worker process never load an arbitrary model implementation from catalog data.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from app.core.job_cancellation import (
    JobCancelledError,
    is_cancel_requested,
    kill_subprocess_tree,
    track_subprocess,
)
from app.core.logging_config import get_logger
from app.core.subprocess_group import popen_group
from app.ml.inference.base import DetectionModel
from app.ml.inference.megadetector import MegaDetectorV1000
from app.ml.inference.video_detector import VideoDetectionModel
from app.ml.schemas.model_manifest import DetectorBackend, ModelManifest
from app.utils.subprocess_env import clean_python_env

logger = get_logger(__name__)

ProgressCallback = Callable[..., None]


def _finite_float(value: Any, *, default: float | None = None) -> float | None:
    """Return a finite confidence value in [0, 1], or ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    if number < 0.0 or number > 1.0:
        return default
    return number


def _normalise_bbox(value: Any) -> list[float] | None:
    """Validate and clamp an xywh-normalized bbox, rejecting degenerate data."""
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in (x, y, width, height)):
        return None
    if any(v < 0.0 or v > 1.0 for v in (x, y, width, height)):
        return None
    if width <= 0.0 or height <= 0.0 or x + width > 1.0 or y + height > 1.0:
        return None
    return [x, y, width, height]


def class_category_map(
    class_names: dict[str, str] | list[str] | None,
) -> dict[str, str]:
    """Return the MegaDetector-compatible ``detection_categories`` map."""
    if isinstance(class_names, list):
        return {str(i): str(name) for i, name in enumerate(class_names)}
    if isinstance(class_names, dict):
        return {str(k): str(v) for k, v in class_names.items()}
    return {}


def normalize_detections(
    detections: Iterable[dict[str, Any]] | None,
    *,
    class_names: dict[str, str] | list[str] | None = None,
    frame_number: int | None = None,
) -> list[dict[str, Any]]:
    """Normalize YOLO/RF-DETR detections to MD's ``conf`` + ``bbox`` shape.

    Invalid or missing boxes/confidences are dropped.  This is intentional:
    downstream persistence requires a usable box and confidence, and inventing
    either value would make an otherwise auditable run impossible to interpret.
    """
    output: list[dict[str, Any]] = []
    for raw in detections or []:
        if not isinstance(raw, dict):
            continue
        bbox = _normalise_bbox(raw.get("bbox"))
        confidence = _finite_float(raw.get("conf", raw.get("confidence")))
        if bbox is None or confidence is None:
            continue
        category_raw = raw.get("category", raw.get("class_id", raw.get("class")))
        if category_raw is None:
            continue
        category = str(category_raw)
        item: dict[str, Any] = {"category": category, "conf": confidence, "bbox": bbox}
        if frame_number is not None:
            item["frame_number"] = int(frame_number)
        output.append(item)
    return output


def normalize_image_results(
    results: Iterable[dict[str, Any]] | None,
    *,
    deployment_folder: Path,
    class_names: dict[str, str] | list[str] | None = None,
) -> dict[str, Any]:
    """Build a MegaDetector-compatible image results document."""
    images: list[dict[str, Any]] = []
    categories = class_category_map(class_names)
    for raw in results or []:
        if not isinstance(raw, dict) or not raw.get("file"):
            continue
        raw_names = raw.get("class_names")
        if isinstance(raw_names, dict):
            # Explicit manifest labels are authoritative.  Child model names
            # still fill in classes absent from the manifest, which keeps
            # detection_categories useful when a manifest omits class_names.
            for key, value in raw_names.items():
                categories.setdefault(str(key), str(value))
        file_value = Path(str(raw["file"]))
        try:
            relative = str(file_value.resolve().relative_to(deployment_folder.resolve()))
        except ValueError:
            relative = str(file_value)
        image: dict[str, Any] = {
            "file": relative,
            "detections": normalize_detections(
                raw.get("detections"),
                class_names=class_names if class_names is not None else raw_names,
            ),
        }
        for key in (
            "width",
            "height",
            "failure",
            "exif_metadata",
            "frame_rate",
            "frames_processed",
        ):
            if key in raw:
                image[key] = raw[key]
        images.append(image)
    return {"images": images, "detection_categories": categories, "info": {"format_version": "1.0"}}


def normalize_video_results(
    results: Iterable[dict[str, Any]] | None,
    *,
    deployment_folder: Path,
    class_names: dict[str, str] | list[str] | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """Normalize sampled video results while retaining frame/video metadata."""
    document = normalize_image_results(
        results, deployment_folder=deployment_folder, class_names=class_names
    )
    if fps is not None:
        document["info"]["frame_rate"] = float(fps)
        for image in document["images"]:
            image.setdefault("frame_rate", float(fps))
            image.setdefault("frames_processed", [])
    return document


class IsolatedSubprocessRunner:
    """Small injectable runner used by non-MegaDetector adapters."""

    def __init__(
        self,
        python_path: Path | str,
        *,
        env: dict[str, str] | None = None,
        script_path: Path | None = None,
    ):
        self.python_path = str(python_path)
        # Run the selected ML interpreter without inheriting a foreign
        # PYTHONHOME/PYTHONPATH from the backend interpreter.
        self.env = clean_python_env() if env is None else env
        self.script_path = script_path

    def run(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,
        job_id: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        script_path = self.script_path or Path(__file__).with_name("detector_subprocess.py")
        command = [self.python_path, str(script_path), *map(str, args)]
        logger.info("Running isolated detector subprocess: %s", " ".join(command))
        try:
            process = popen_group(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self.env,
            )
            with track_subprocess(job_id, process):
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    kill_subprocess_tree(process)
                    process.communicate()
                    raise

            if job_id is not None and is_cancel_requested(job_id):
                raise JobCancelledError(f"Detector subprocess cancelled for job {job_id}")

            completed = subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
            completed.check_returncode()
            return completed
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"exit code {exc.returncode}"
            name = self.backend_name if hasattr(self, "backend_name") else "detector"
            raise RuntimeError(f"{name} subprocess failed: {detail}") from exc


class _GenericAdapter(DetectionModel):
    """Common isolated adapter implementation for YOLO and RF-DETR."""

    backend: DetectorBackend

    def __init__(
        self,
        manifest: ModelManifest,
        model_path: Path,
        env_manager: Any,
        *,
        runner: IsolatedSubprocessRunner | None = None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.manifest = manifest
        self.model_path = model_path
        self.env_manager = env_manager
        python_path = env_manager.get_python(f"env-{manifest.env}")
        self.runner = runner or IsolatedSubprocessRunner(python_path)

    def _run_images(
        self,
        image_paths: list[Path],
        deployment_folder: Path,
        confidence_threshold: float,
        *,
        batch_size: int | None = None,
        image_size: int | None = None,
        augment: bool = False,
        progress_callback: ProgressCallback | None = None,
        output_path: Path | None = None,
        job_id: str | None = None,
    ) -> Path:
        del batch_size, augment
        if not image_paths:
            raise ValueError("No image paths provided")
        if job_id is not None and is_cancel_requested(job_id):
            raise JobCancelledError(f"Detection cancelled for job {job_id}")
        for path in image_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
        output_path = output_path or deployment_folder / ".addaxai" / "detection_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="addaxai-detector-") as temp_dir:
            temp = Path(temp_dir)
            file_list = temp / "inputs.json"
            raw_output = temp / "raw.json"
            file_list.write_text(json.dumps([str(p) for p in image_paths]), encoding="utf-8")
            args = [
                "--backend", self.backend,
                "--model", str(self.model_path),
                "--inputs", str(file_list),
                "--output", str(raw_output),
                "--threshold", str(confidence_threshold),
                "--class-names", json.dumps(class_category_map(self.manifest.class_names)),
            ]
            if self.backend == "rfdetr":
                args.extend(["--model-class", self.manifest.detector_model_class or "RFDETRMedium"])
            if self.backend == "rtdetr":
                args.extend(
                    ["--model-variant", self.manifest.detector_model_variant or "MDV6-apa-rtdetr-c"]
                )
            if self.backend == "rtdetrv2":
                config_name = self.manifest.detector_config_fname
                if not config_name:
                    raise ValueError("detector_config_fname is required for rtdetrv2")
                config_path = (self.model_path.parent / config_name).resolve()
                try:
                    config_path.relative_to(self.model_path.parent.resolve())
                except ValueError as exc:
                    raise ValueError(
                        "RT-DETRv2 config must remain inside its model directory"
                    ) from exc
                if not config_path.is_file():
                    raise FileNotFoundError(f"RT-DETRv2 config not found: {config_path}")
                source_path = (
                    Path(__file__).resolve().parents[1] / "third_party" / "rtdetrv2_pytorch"
                )
                if not (source_path / "src" / "core" / "__init__.py").is_file():
                    raise FileNotFoundError(f"Pinned RT-DETRv2 source not found: {source_path}")
                args.extend(
                    [
                        "--model-config",
                        str(config_path),
                        "--rtdetrv2-source",
                        str(source_path),
                    ]
                )
            if image_size is not None:
                args.extend(["--image-size", str(image_size)])
            if progress_callback:
                progress_callback(f"Starting {self.backend} detector...", 0.0)
            self.runner.run(args, job_id=job_id)
            raw_results = json.loads(raw_output.read_text(encoding="utf-8"))
            document = normalize_image_results(
                raw_results.get("images", raw_results),
                deployment_folder=deployment_folder,
                class_names=self.manifest.class_names,
            )
            document["info"]["detector_backend"] = self.backend
            output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        if progress_callback:
            progress_callback("Detection complete", 1.0)
        return output_path

    def detect_to_json(self, **kwargs: Any) -> Path:
        return self._run_images(**kwargs)

    def detect_videos_to_json(
        self,
        video_folder: Path,
        output_json: Path,
        fps: float,
        confidence_threshold: float,
        image_size: int | None = None,
        augment: bool = False,
        progress_callback: ProgressCallback | None = None,
        job_id: str | None = None,
    ) -> Path:
        """Use the shared video sampler, then run image inference in isolation."""
        del augment
        with tempfile.TemporaryDirectory(prefix="addaxai-video-frames-") as frames_dir:
            return self._detect_sampled_videos(
                video_folder=video_folder,
                output_json=output_json,
                frame_root=Path(frames_dir),
                fps=fps,
                confidence_threshold=confidence_threshold,
                image_size=image_size,
                progress_callback=progress_callback,
                job_id=job_id,
            )

    def _detect_sampled_videos(
        self,
        *,
        video_folder: Path,
        output_json: Path,
        frame_root: Path,
        fps: float,
        confidence_threshold: float,
        image_size: int | None,
        progress_callback: ProgressCallback | None,
        job_id: str | None,
    ) -> Path:
        """Sample video frames under a private temp root and write MD JSON."""
        import cv2

        samples: list[dict[str, Any]] = []
        video_records: dict[str, dict[str, Any]] = {}
        sample_period = 1.0 / max(float(fps), 1e-6)

        def raise_if_cancelled() -> None:
            if job_id is not None and is_cancel_requested(job_id):
                raise JobCancelledError(f"Detection cancelled for job {job_id}")

        for video in sorted(video_folder.rglob("*")):
            raise_if_cancelled()
            if video.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
                continue
            capture = cv2.VideoCapture(str(video))
            relative_video = str(video.resolve().relative_to(video_folder.resolve()))
            frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or fps)
            record: dict[str, Any] = {
                "file": relative_video,
                "frame_rate": frame_rate,
                "frames_processed": [],
                "detections": [],
            }
            video_records[relative_video] = record
            if not capture.isOpened():
                record.update({"failure": "Unable to open video", "detections": None})
                capture.release()
                continue
            frame_index = 0
            next_sample = 0.0
            while capture.isOpened():
                if job_id is not None and is_cancel_requested(job_id):
                    capture.release()
                    raise_if_cancelled()
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / max(frame_rate, 1e-6)
                if timestamp + 1e-9 >= next_sample:
                    frame_path = frame_root / f"frame-{len(samples):08d}.jpg"
                    if not cv2.imwrite(str(frame_path), frame):
                        record.update(
                            {"failure": "Unable to write sampled frame", "detections": None}
                        )
                        break
                    samples.append(
                        {
                            "file": str(video),
                            "relative_file": relative_video,
                            "frame": frame_index,
                            "path": frame_path,
                        }
                    )
                    record["frames_processed"].append(frame_index)
                    next_sample += sample_period
                frame_index += 1
            capture.release()
            if not record["frames_processed"] and "failure" not in record:
                record.update({"failure": "No frames decoded", "detections": None})
        if not video_records:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps({"images": [], "detection_categories": {}}),
                encoding="utf-8",
            )
            return output_json
        detected: dict[str, Any] = {
            "images": [],
            "detection_categories": {},
            "info": {"detector_backend": self.backend},
        }
        if samples:
            raw_paths = [sample["path"] for sample in samples]
            with tempfile.TemporaryDirectory(prefix="addaxai-video-") as temp_dir:
                intermediate = Path(temp_dir) / "images.json"
                self._run_images(
                    raw_paths,
                    video_folder,
                    confidence_threshold,
                    image_size=image_size,
                    progress_callback=progress_callback,
                    output_path=intermediate,
                    job_id=job_id,
                )
                detected = json.loads(intermediate.read_text(encoding="utf-8"))
        sample_by_name = {sample["path"].name: sample for sample in samples}
        for image in detected.get("images", []):
            sample = sample_by_name.get(Path(image.get("file", "")).name)
            if sample is None:
                continue
            record = video_records[sample["relative_file"]]
            for det in image.get("detections", []):
                det["frame_number"] = int(sample["frame"])
                record["detections"].append(det)
            detected.setdefault("detection_categories", {})
            detected["detection_categories"].update(
                {str(k): str(v) for k, v in (image.get("class_names") or {}).items()}
            )
        detected["images"] = list(video_records.values())
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(detected, indent=2), encoding="utf-8")
        return output_json


class YoloDetectorAdapter(_GenericAdapter):
    backend: DetectorBackend = "yolo"


class RFDETRDetectorAdapter(_GenericAdapter):
    backend: DetectorBackend = "rfdetr"


class RTDETRDetectorAdapter(_GenericAdapter):
    backend: DetectorBackend = "rtdetr"


class RTDETRv2DetectorAdapter(_GenericAdapter):
    backend: DetectorBackend = "rtdetrv2"


class MegaDetectorAdapter(DetectionModel):
    """Compatibility adapter that preserves the existing MD code path."""

    def __init__(self, model_path: Path, env_manager: Any) -> None:
        self._image = MegaDetectorV1000(model_path, env_manager)
        self._video = VideoDetectionModel(model_path, env_manager)

    def detect_to_json(self, **kwargs: Any) -> Path:
        return self._image.detect_to_json(**kwargs)

    def detect_videos_to_json(self, **kwargs: Any) -> Path:
        return self._video.detect_videos_to_json(**kwargs)


def create_detector(
    manifest: ModelManifest,
    model_path: Path,
    env_manager: Any,
    *,
    runner: IsolatedSubprocessRunner | None = None,
) -> DetectionModel:
    """Create a detector from a validated manifest, defaulting to MegaDetector."""
    backend = manifest.detector_backend or "megadetector"
    if backend == "megadetector":
        return MegaDetectorAdapter(model_path, env_manager)
    if backend == "yolo":
        return YoloDetectorAdapter(manifest, model_path, env_manager, runner=runner)
    if backend == "rfdetr":
        return RFDETRDetectorAdapter(manifest, model_path, env_manager, runner=runner)
    if backend == "rtdetr":
        return RTDETRDetectorAdapter(manifest, model_path, env_manager, runner=runner)
    if backend == "rtdetrv2":
        return RTDETRv2DetectorAdapter(manifest, model_path, env_manager, runner=runner)
    raise ValueError(f"Unsupported detector backend: {backend}")


DetectorFactory = create_detector
