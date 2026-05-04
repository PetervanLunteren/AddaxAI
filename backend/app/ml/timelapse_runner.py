"""
Timelapse runner: DB-less analysis pipeline for the Timelapse Analyser
integration.

Mirrors the phase order in `app.workers.detection_worker` (video detection,
video classification, image detection, image classification, merge, then
postprocessing) but reuses the JSON-level primitives directly so no
Project / Deployment / File / Detection rows are ever created.

The user-visible artifact is `<folder>/results.json` in the same shape the
main app produces. Intermediate per-phase JSONs are kept in
`<folder>/.addaxai/timelapse/` for diagnostics; that directory is hidden
and recreated each run.

Progress is streamed through the same `ws_manager` + `/ws/jobs/{job_id}`
channel the main worker uses, so the frontend's `useTaskProgress` hook
works unchanged.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.custom_classification_model import CustomClassificationModel
from app.ml.inference.megadetector import MegaDetectorV1000
from app.ml.json_pipeline import merge_json_files, run_classification_on_json
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage
from app.ml.postprocessing import run_postprocessing_on_json
from app.workers.detection_worker import (
    scan_folder_for_images,
    scan_folder_for_videos,
)

logger = get_logger(__name__)


@dataclass
class TimelapseRunRequest:
    """All inputs needed for a single Timelapse run.

    Defaults match the main-app project defaults so behavior matches what
    users already know.
    """

    folder_path: Path
    classification_model_id: str | None
    detection_model_id: str = "MD5A-0-0"
    excluded_classes: list[str] | None = None
    detection_threshold: float = 0.5
    detection_batch_size: int = 1
    classification_batch_size: int = 16
    video_fps: float = 1.0
    independence_interval: int = 1800  # seconds, matches main app
    smoothing_strength: str = "normal"  # off | mild | normal | aggressive
    taxonomic_rollup: bool = True


def _resolve_model_paths(request: TimelapseRunRequest) -> dict:
    """Resolve detection + (optional) classification model paths.

    Same lookup the worker performs at startup (`ManifestManager` +
    `ModelStorage`), only extracted into a helper so the orchestrator
    function reads top-down.
    """
    manifest_manager = ManifestManager()
    model_storage = ModelStorage()
    env_manager = EnvironmentManager()

    det_manifest = manifest_manager.get_model(request.detection_model_id)
    det_model_path = model_storage.get_model_file(det_manifest)

    cls_model_path: Path | None = None
    cls_model_dir: Path | None = None
    cls_env_name: str | None = None
    if request.classification_model_id:
        cls_manifest = manifest_manager.get_model(request.classification_model_id)
        cls_model_path = model_storage.get_model_file(cls_manifest)
        cls_model_dir = model_storage.get_model_path(cls_manifest)
        cls_env_name = cls_manifest.env

        inference_script = cls_model_dir / "inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(
                f"Custom inference script not found: {inference_script}"
            )

    return {
        "env_manager": env_manager,
        "det_model_path": det_model_path,
        "cls_model_path": cls_model_path,
        "cls_model_dir": cls_model_dir,
        "cls_env_name": cls_env_name,
    }


async def run(request: TimelapseRunRequest, job_id: str) -> Path:
    """Run the full Timelapse analysis. Returns the path to the written JSON.

    Streams progress via `ws_manager` keyed by `job_id`. The caller is
    responsible for spawning this coroutine on the event loop and surfacing
    `job_id` to the client (see `app.api.routers.timelapse`).
    """
    folder_path = request.folder_path.resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    artifacts_folder = folder_path / ".addaxai" / "timelapse"
    artifacts_folder.mkdir(parents=True, exist_ok=True)

    video_json_path = artifacts_folder / "detection_video.json"
    image_json_path = artifacts_folder / "detection_image.json"
    merged_json_path = artifacts_folder / "results.json"
    final_json_path = folder_path / "results.json"

    await ws_manager.send_progress(job_id, "Initializing models...", 0.01)

    paths = _resolve_model_paths(request)
    env_manager: EnvironmentManager = paths["env_manager"]
    detection_model = MegaDetectorV1000(paths["det_model_path"], env_manager)

    classification_model: CustomClassificationModel | None = None
    if request.classification_model_id:
        classification_model = CustomClassificationModel(
            paths["cls_model_dir"],
            paths["cls_model_path"],
            paths["cls_env_name"],
            env_manager,
        )

    video_files = scan_folder_for_videos(folder_path)
    image_files = scan_folder_for_images(folder_path)

    if not video_files and not image_files:
        raise RuntimeError(f"No images or videos found in {folder_path}")

    has_classifier = classification_model is not None
    init_data = {
        "deployment_index": 1,
        "total_deployments": 1,
        "video_count": len(video_files),
        "image_count": len(image_files),
        "has_classifier": has_classifier,
        "has_embedding": False,
    }
    await ws_manager.send_progress(
        job_id, "", 0.02, phase="init", phase_progress=0.0, data=init_data
    )

    json_files_to_merge: list[Path] = []
    loop = asyncio.get_event_loop()

    async def progress(message: str, overall: float, phase: str,
                       phase_progress: float, metrics: dict | None = None) -> None:
        data = dict(init_data)
        if metrics and "compute_device" in metrics:
            data["compute_device"] = metrics.pop("compute_device")
        if metrics:
            data["metrics"] = metrics
        await ws_manager.send_progress(
            job_id, message, overall, phase, phase_progress, data
        )

    # Phase 1: Video detection.
    if video_files:
        from app.ml.inference.video_detector import VideoDetectionModel

        video_detector = VideoDetectionModel(paths["det_model_path"], env_manager)

        def video_detection_progress(
            message: str, phase_progress: float, metrics: dict | None = None,
        ) -> None:
            if metrics:
                metrics["unit"] = "video"
            asyncio.run_coroutine_threadsafe(
                progress(message, 0.0, "video_detection", phase_progress, metrics),
                loop,
            )

        await loop.run_in_executor(
            None,
            lambda: video_detector.detect_videos_to_json(
                video_folder=folder_path,
                output_json=video_json_path,
                fps=request.video_fps,
                confidence_threshold=request.detection_threshold,
                progress_callback=video_detection_progress,
                job_id=job_id,
            ),
        )
        json_files_to_merge.append(video_json_path)

    # Extract video frames (needed for video classification crops).
    if video_files and video_json_path.exists() and has_classifier:
        from app.ml.frame_extraction import extract_all_video_frames

        try:
            extract_all_video_frames(
                folder_path,
                request.video_fps,
                env_manager,
                output_dir=artifacts_folder / "video_frames",
                job_id=job_id,
            )
        except Exception as e:
            logger.error(f"Video frame extraction failed: {e}", exc_info=True)
            # Non-fatal — classification will skip videos with missing frames.

    # Phase 2: Video classification.
    if video_files and classification_model and video_json_path.exists():
        async def video_cls_progress(
            message: str, phase_progress: float, metrics: dict | None = None,
        ) -> None:
            if metrics:
                metrics["unit"] = "animal"
            await progress(message, 0.0, "video_classification", phase_progress, metrics)

        await run_classification_on_json(
            json_path=video_json_path,
            classification_model=classification_model,
            deployment_folder=folder_path,
            batch_size=request.classification_batch_size,
            progress_callback=video_cls_progress,
            classification_model_dir=paths["cls_model_dir"],
            video_frames_base_dir=artifacts_folder / "video_frames",
            job_id=job_id,
        )

    # Phase 3: Image detection.
    if image_files:
        def image_detection_progress(
            message: str, phase_progress: float, metrics: dict | None = None,
        ) -> None:
            if metrics:
                metrics["unit"] = "image"
            asyncio.run_coroutine_threadsafe(
                progress(message, 0.0, "image_detection", phase_progress, metrics),
                loop,
            )

        image_json_path = await loop.run_in_executor(
            None,
            lambda: detection_model.detect_to_json(
                image_paths=image_files,
                deployment_folder=folder_path,
                confidence_threshold=request.detection_threshold,
                batch_size=request.detection_batch_size,
                progress_callback=image_detection_progress,
                output_path=image_json_path,
                job_id=job_id,
            ),
        )
        json_files_to_merge.append(image_json_path)

    # Phase 4: Image classification.
    if image_files and classification_model and image_json_path.exists():
        async def image_cls_progress(
            message: str, phase_progress: float, metrics: dict | None = None,
        ) -> None:
            await progress(message, 0.0, "image_classification", phase_progress, metrics)

        await run_classification_on_json(
            json_path=image_json_path,
            classification_model=classification_model,
            deployment_folder=folder_path,
            batch_size=request.classification_batch_size,
            progress_callback=image_cls_progress,
            classification_model_dir=paths["cls_model_dir"],
            video_frames_base_dir=artifacts_folder / "video_frames",
            job_id=job_id,
        )

    # Phase 5: Merge JSONs.
    await progress("Merging results...", 0.85, "saving", 0.5)
    pseudo_deployment_id = str(uuid.uuid4())  # for info.addaxai metadata only
    merge_json_files(
        json_files_to_merge,
        merged_json_path,
        pseudo_deployment_id,
        detection_model_id=request.detection_model_id,
        classification_model_id=request.classification_model_id,
    )

    # Phase 6: Postprocessing on JSON.
    await progress("Postprocessing...", 0.92, "saving", 0.8)
    event_smoothing = request.smoothing_strength != "off"
    smoothing_strength = (
        request.smoothing_strength if event_smoothing else "normal"
    )
    await loop.run_in_executor(
        None,
        lambda: run_postprocessing_on_json(
            merged_json_path,
            folder_path,
            excluded_classes=request.excluded_classes,
            taxonomic_rollup=request.taxonomic_rollup,
            event_smoothing=event_smoothing,
            smoothing_strength=smoothing_strength,
            independence_interval=request.independence_interval,
            detection_threshold=request.detection_threshold,
            classification_model_dir=paths["cls_model_dir"],
            job_id=job_id,
        ),
    )

    # Final: copy postprocessed JSON to <folder>/results.json. Keep
    # artifacts_folder around for diagnostics.
    final_json_path.write_bytes(merged_json_path.read_bytes())

    await ws_manager.send_complete(
        job_id,
        True,
        f"Wrote {final_json_path}",
        data={"output_path": str(final_json_path)},
    )
    return final_json_path
