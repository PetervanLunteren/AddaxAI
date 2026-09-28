"""Unit tests for manifest-driven detector adapters (no model packages)."""

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.job_cancellation import (
    JobCancelledError,
    clear_cancel,
    request_cancel,
)
from app.ml.inference.detector_backend import (
    IsolatedSubprocessRunner,
    MegaDetectorAdapter,
    RFDETRDetectorAdapter,
    YoloDetectorAdapter,
    class_category_map,
    create_detector,
    normalize_detections,
    normalize_image_results,
    normalize_video_results,
)
from app.ml.model_storage import ModelStorage
from app.ml.schemas.model_manifest import ModelManifest


def manifest(**overrides):
    values = {
        "model_id": "local-detector",
        "friendly_name": "Local detector",
        "env": "addaxai-base",
        "model_fname": "weights.pt",
        "description": "test",
        "developer": "test",
        "info_url": "https://example.invalid",
        "min_app_version": "0",
    }
    values.update(overrides)
    return ModelManifest(**values)


def test_manifest_defaults_to_megadetector_and_rejects_unallowlisted_rfdetr():
    assert manifest().detector_backend == "megadetector"
    with pytest.raises(ValueError, match="not supported"):
        manifest(detector_backend="rfdetr", detector_model_class="os.system")


def test_manifest_normalises_class_name_list():
    assert manifest(detector_backend="yolo", class_names=["fox", "deer"]).class_names == {
        "0": "fox",
        "1": "deer",
    }


def test_class_category_map_and_normalization_drop_invalid_values():
    assert class_category_map(["animal"]) == {"0": "animal"}
    detections = normalize_detections(
        [
            {"class_id": 0, "conf": 0.5, "bbox": [0.1, 0.2, 0.3, 0.4]},
            {"class_id": 1, "conf": None, "bbox": [0, 0, 1, 1]},
            {"class_id": 2, "conf": 0.8, "bbox": [0, 0, 0, 1]},
            {"class_id": 3, "conf": 0.8, "bbox": [0, 0, 2, 2]},
        ],
        class_names=["fox"],
        frame_number=4,
    )
    assert detections == [
        {
            "category": "0",
            "conf": 0.5,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "frame_number": 4,
        },
    ]


def test_image_and_video_normalization_preserve_metadata(tmp_path: Path):
    image = tmp_path / "nested" / "photo.jpg"
    image.parent.mkdir()
    image.touch()
    document = normalize_image_results(
        [{"file": str(image), "width": 100, "height": 80, "detections": []}],
        deployment_folder=tmp_path,
        class_names={"0": "fox"},
    )
    assert document["images"][0]["file"].replace("\\", "/") == "nested/photo.jpg"
    assert document["images"][0]["width"] == 100
    inferred = normalize_image_results(
        [
            {
                "file": str(image),
                "class_names": {"7": "otter"},
                "detections": [{"class_id": 7, "conf": 0.7, "bbox": [0, 0, 0.5, 0.5]}],
            }
        ],
        deployment_folder=tmp_path,
    )
    assert inferred["detection_categories"] == {"7": "otter"}
    video = normalize_video_results(
        [
            {
                "file": str(image),
                "frame_number": 12,
                "detections": [
                    {"category": "0", "conf": 0.9, "bbox": [0, 0, 1, 1]}
                ],
            }
        ],
        deployment_folder=tmp_path,
        class_names={"0": "fox"},
        fps=2,
    )
    assert video["info"]["frame_rate"] == 2


def test_manifest_class_names_override_child_names_but_fill_missing_classes(tmp_path: Path):
    image = tmp_path / "photo.jpg"
    image.touch()
    document = normalize_image_results(
        [
            {
                "file": str(image),
                "class_names": {"0": "model fox", "1": "deer"},
                "detections": [
                    {"class_id": 0, "conf": 0.8, "bbox": [0, 0, 0.5, 0.5]},
                    {"class_id": 1, "conf": 0.7, "bbox": [0.5, 0.5, 0.5, 0.5]},
                ],
            }
        ],
        deployment_folder=tmp_path,
        class_names={"0": "manifest fox"},
    )
    assert document["detection_categories"] == {"0": "manifest fox", "1": "deer"}


def test_detector_signature_changes_with_class_map_and_weight_file(tmp_path: Path):
    from app.workers.detection_worker import _detection_model_signature

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    original = manifest(detector_backend="yolo", class_names={"0": "fox"})
    initial = _detection_model_signature(original, weights)
    assert _detection_model_signature(original, weights) == initial

    changed_classes = manifest(detector_backend="yolo", class_names={"0": "deer"})
    assert _detection_model_signature(changed_classes, weights) != initial

    weights.write_bytes(b"different weights")
    assert _detection_model_signature(original, weights) != initial


def test_yolo_child_omits_none_image_size(monkeypatch, tmp_path: Path):
    """Ultralytics receives its default image size when no override is set."""
    captured: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, _model_path: str):
            pass

        def predict(self, **kwargs):
            captured.update(kwargs)
            result = SimpleNamespace(
                boxes=SimpleNamespace(xyxy=[[0, 0, 10, 10]], conf=[0.8], cls=[0]),
                names={0: "fox"},
                orig_shape=(20, 20),
            )
            return [result]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    from app.ml.inference import detector_subprocess

    output = detector_subprocess._run_yolo(
        tmp_path / "weights.pt", [tmp_path / "photo.jpg"], 0.1, None
    )
    assert "imgsz" not in captured
    assert captured["source"] == str(tmp_path / "photo.jpg")
    assert captured["stream"] is False
    assert output[0]["width"] == 20
    assert output[0]["height"] == 20
    assert output[0]["class_names"] == {"0": "fox"}


def test_yolo_child_reads_source_dimensions_when_result_shape_is_missing(
    monkeypatch, tmp_path: Path
):
    from PIL import Image

    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (37, 23)).save(image_path)

    class FakeYOLO:
        def __init__(self, _model_path: str):
            pass

        def predict(self, **_kwargs):
            return [SimpleNamespace(boxes=None, names={}, orig_shape=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    from app.ml.inference import detector_subprocess

    output = detector_subprocess._run_yolo(
        tmp_path / "weights.pt", [image_path], 0.1, None
    )

    assert (output[0]["width"], output[0]["height"]) == (37, 23)


def test_rfdetr_child_includes_source_dimensions(monkeypatch, tmp_path: Path):
    from PIL import Image

    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (31, 19)).save(image_path)

    class FakeRFDETR:
        classes = {0: "fox"}

        def __init__(self, **_kwargs):
            pass

        def predict(self, _path: str, *, threshold: float):
            assert threshold == 0.25
            return SimpleNamespace(xyxy=[], confidence=[], class_id=[])

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRMedium=FakeRFDETR))
    from app.ml.inference import detector_subprocess

    output = detector_subprocess._run_rfdetr(
        tmp_path / "weights.pth", [image_path], 0.25, None, "RFDETRMedium"
    )

    assert (output[0]["width"], output[0]["height"]) == (31, 19)


def test_yolo_child_streams_100_results_in_input_order(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {"sources": []}
    paths = [tmp_path / f"image-{index:03}.jpg" for index in range(100)]

    class FakeYOLO:
        def __init__(self, _model_path: str):
            pass

        def predict(self, **kwargs):
            source = kwargs["source"]
            assert isinstance(source, str)
            assert kwargs["stream"] is False
            captured["sources"].append(source)
            index = int(Path(source).stem.rsplit("-", 1)[1])
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=[[index, 0, index + 10, 10]],
                        conf=[0.8],
                        cls=[0],
                    ),
                    names={0: "fox"},
                    orig_shape=(1000, 1000),
                )
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    from app.ml.inference import detector_subprocess

    output = detector_subprocess._run_yolo(
        tmp_path / "weights.pt", paths, 0.1, None
    )

    assert len(output) == 100
    assert [Path(item["file"]) for item in output] == paths
    assert [item["detections"][0]["bbox"][0] for item in output] == [
        index / 1000 for index in range(100)
    ]
    assert captured["sources"] == [str(path) for path in paths]


def test_generic_video_sampler_uses_unique_temporary_frames_and_preserves_json(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "videos"
    source.mkdir()
    video_files = [source / "clip.mp4", source / "clip.avi"]
    for video in video_files:
        video.touch()
    written_frames: list[Path] = []
    opened_videos: list[str] = []

    class FakeCapture:
        def __init__(self, path: str):
            opened_videos.append(path)
            self.position = 0

        def get(self, _property):
            return 2.0

        def isOpened(self):
            return True

        def read(self):
            if self.position >= 4:
                return False, None
            self.position += 1
            return True, bytes([self.position])

        def release(self):
            return None

    cv2 = SimpleNamespace(
        CAP_PROP_FPS=5,
        VideoCapture=FakeCapture,
        imwrite=lambda name, _frame: (
            written_frames.append(Path(name)),
            Path(name).write_bytes(b"frame"),
            True,
        )[-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    detector = YoloDetectorAdapter(
        manifest(detector_backend="yolo"),
        weights,
        SimpleNamespace(get_python=lambda _env: Path("python")),
        runner=object(),
    )
    inferred_paths: list[Path] = []

    def fake_run_images(
        image_paths,
        deployment_folder,
        _confidence_threshold,
        *,
        output_path,
        **_kwargs,
    ):
        inferred_paths.extend(image_paths)
        assert all(path.parent != deployment_folder for path in image_paths)
        output_path.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "file": str(path),
                            "class_names": {"0": "fox"},
                            "detections": [
                                {"category": "0", "conf": 0.9, "bbox": [0, 0, 0.5, 0.5]}
                            ],
                        }
                        for path in image_paths
                    ]
                }
            ),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(detector, "_run_images", fake_run_images)
    output_path = tmp_path / "output.json"
    detector.detect_videos_to_json(
        video_folder=source,
        output_json=output_path,
        fps=1.0,
        confidence_threshold=0.1,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(opened_videos) == {str(path) for path in video_files}
    assert len(inferred_paths) == 4
    assert len({path.name for path in written_frames}) == 4
    assert all(not path.exists() for path in written_frames)
    assert sorted(path.name for path in source.iterdir()) == ["clip.avi", "clip.mp4"]
    records = {Path(video["file"]).name: video for video in result["images"]}
    assert set(records) == {"clip.avi", "clip.mp4"}
    assert records["clip.mp4"]["frames_processed"] == [0, 2]
    assert records["clip.mp4"]["frame_rate"] == 2.0
    assert [item["frame_number"] for item in records["clip.mp4"]["detections"]] == [0, 2]
    assert result["detection_categories"] == {"0": "fox"}


def test_factory_selects_backends_without_importing_external_packages(tmp_path: Path, monkeypatch):
    weight = tmp_path / "weights.pt"
    weight.touch()

    class Env:
        def get_python(self, name):
            return Path("python")

    md = manifest()
    monkeypatch.setattr(
        "app.ml.inference.detector_backend.MegaDetectorV1000",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "app.ml.inference.detector_backend.VideoDetectionModel",
        lambda *_args: object(),
    )
    assert isinstance(create_detector(md, weight, Env()), MegaDetectorAdapter)
    yolo = create_detector(
        manifest(detector_backend="yolo"), weight, Env(),
        runner=IsolatedSubprocessRunner("python"),
    )
    assert isinstance(yolo, YoloDetectorAdapter)
    rf = create_detector(
        manifest(detector_backend="rfdetr", detector_model_class="RFDETRSmall"), weight, Env(),
        runner=IsolatedSubprocessRunner("python"),
    )
    assert isinstance(rf, RFDETRDetectorAdapter)


def test_isolated_runner_builds_script_command(monkeypatch):
    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, **_kwargs):
            return "", ""

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("app.ml.inference.detector_backend.popen_group", fake_popen)
    completed = IsolatedSubprocessRunner("python").run(["--backend", "yolo"])
    assert captured["command"][0] == "python"
    assert captured["command"][1].replace("\\", "/").endswith(
        "app/ml/inference/detector_subprocess.py"
    )
    assert "-m" not in captured["command"]
    assert captured["kwargs"]["stdout"] == subprocess.PIPE
    assert completed.returncode == 0


def test_isolated_runner_cancellation_kills_child_process_tree(tmp_path: Path):
    ready_path = tmp_path / "ready.pid"
    script_path = tmp_path / "sleep.py"
    script_path.write_text(
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid()))\n"
        "while True: time.sleep(0.05)\n",
        encoding="utf-8",
    )
    job_id = "generic-detector-cancel-test"
    runner = IsolatedSubprocessRunner(sys.executable, script_path=script_path)
    process_id: int | None = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner.run, [str(ready_path)], job_id=job_id)
        deadline = time.monotonic() + 10
        try:
            while not ready_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ready_path.exists(), "child did not signal that it started"
            process_id = int(ready_path.read_text(encoding="utf-8"))
            request_cancel(job_id)
            with pytest.raises(JobCancelledError, match="cancelled"):
                future.result(timeout=10)
        finally:
            # Keep the test process from leaking a child if the cancellation
            # path regresses and the assertion above fails.
            if process_id is not None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process_id)],
                        capture_output=True,
                    )
                else:
                    try:
                        os.kill(process_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if not future.done():
                request_cancel(job_id)
                future.result(timeout=10)
            clear_cancel(job_id)


def test_generic_adapter_passes_job_id_to_isolated_runner(tmp_path: Path):
    model_path = tmp_path / "weights.pt"
    image_path = tmp_path / "image.jpg"
    model_path.touch()
    image_path.touch()
    captured: dict[str, object] = {}

    class StubRunner:
        def run(self, args, *, job_id=None):
            captured["job_id"] = job_id
            output_path = Path(args[args.index("--output") + 1])
            output_path.write_text(
                '{"images": [{"file": '
                + json.dumps(str(image_path))
                + ', "detections": [], "class_names": {}}]}',
                encoding="utf-8",
            )

    adapter = YoloDetectorAdapter(
        manifest(detector_backend="yolo"),
        model_path,
        SimpleNamespace(get_python=lambda _env: Path("python")),
        runner=StubRunner(),
    )
    output_path = adapter.detect_to_json(
        image_paths=[image_path],
        deployment_folder=tmp_path,
        confidence_threshold=0.1,
        job_id="job-42",
    )

    assert captured["job_id"] == "job-42"
    assert output_path.is_file()


def test_local_only_model_readiness_uses_digest_and_blocks_download(tmp_path: Path):
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"weights")
    import hashlib

    model = manifest(
        model_category="detection",
        local_only=True,
        weights_sha256=hashlib.sha256(b"weights").hexdigest(),
    )
    storage = ModelStorage(tmp_path)
    model_dir = tmp_path / "det" / model.model_id
    model_dir.mkdir(parents=True)
    (model_dir / model.model_fname).write_bytes(weight.read_bytes())
    assert storage.check_weights_ready(model)
    (model_dir / model.model_fname).write_bytes(b"tampered")
    assert not storage.check_weights_ready(model)
    with pytest.raises(RuntimeError, match="local-only"):
        storage.download_weights(model)


def test_generic_adapter_runs_images_in_chunks_with_progress(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.ml.inference.detector_backend.INFERENCE_CHUNK_SIZE", 2)
    model_path = tmp_path / "weights.pt"
    model_path.touch()
    images = [tmp_path / f"image-{index}.jpg" for index in range(5)]
    for image in images:
        image.touch()
    chunks: list[list[str]] = []
    progress: list[tuple[str, float]] = []

    class StubRunner:
        def run(self, args, *, job_id=None):
            inputs = json.loads(
                Path(args[args.index("--inputs") + 1]).read_text(encoding="utf-8")
            )
            chunks.append(inputs)
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "file": path,
                                "detections": [
                                    {"category": "0", "conf": 0.8, "bbox": [0, 0, 0.1, 0.1]}
                                ],
                            }
                            for path in inputs
                        ]
                    }
                ),
                encoding="utf-8",
            )

    adapter = YoloDetectorAdapter(
        manifest(detector_backend="yolo", class_names={"0": "fox"}),
        model_path,
        SimpleNamespace(get_python=lambda _env: Path("python")),
        runner=StubRunner(),
    )
    output_path = adapter.detect_to_json(
        image_paths=images,
        deployment_folder=tmp_path,
        confidence_threshold=0.1,
        progress_callback=lambda message, value: progress.append((message, value)),
    )

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [value for _, value in progress] == [0.0, 0.4, 0.8, 1.0]
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert [image["file"] for image in document["images"]] == [p.name for p in images]
    assert document["detection_categories"] == {"0": "fox"}


def test_generic_video_sampler_flushes_frames_in_chunks_for_all_video_types(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr("app.ml.inference.detector_backend.INFERENCE_CHUNK_SIZE", 2)
    source = tmp_path / "videos"
    source.mkdir()
    for name in ("a.wmv", "b.mpg", "notes.txt"):
        (source / name).touch()
    opened: list[str] = []

    class FakeCapture:
        def __init__(self, path: str):
            opened.append(Path(path).name)
            self.position = 0

        def get(self, _property):
            return 1.0

        def isOpened(self):
            return True

        def read(self):
            if self.position >= 3:
                return False, None
            self.position += 1
            return True, b"frame"

        def release(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            CAP_PROP_FPS=5,
            VideoCapture=FakeCapture,
            imwrite=lambda name, _frame: (Path(name).write_bytes(b"x"), True)[-1],
        ),
    )
    weights = tmp_path / "weights.pt"
    weights.touch()
    detector = YoloDetectorAdapter(
        manifest(detector_backend="yolo", class_names={"0": "fox"}),
        weights,
        SimpleNamespace(get_python=lambda _env: Path("python")),
        runner=object(),
    )
    frames_on_disk: list[int] = []

    def fake_run_images(image_paths, _folder, _threshold, *, output_path, **_kwargs):
        frames_on_disk.append(len(list(image_paths[0].parent.glob("*.jpg"))))
        output_path.write_text(
            json.dumps(
                {
                    "images": [
                        {"file": str(path), "detections": [
                            {"category": "0", "conf": 0.9, "bbox": [0, 0, 0.5, 0.5]}
                        ]}
                        for path in image_paths
                    ],
                    "detection_categories": {"0": "fox"},
                }
            ),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(detector, "_run_images", fake_run_images)
    output_path = tmp_path / "output.json"
    detector.detect_videos_to_json(
        video_folder=source, output_json=output_path, fps=1.0, confidence_threshold=0.1
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert sorted(opened) == ["a.wmv", "b.mpg"]
    # Six sampled frames in chunks of two: never more than one chunk on disk.
    assert frames_on_disk == [2, 2, 2]
    assert {record["file"] for record in result["images"]} == {"a.wmv", "b.mpg"}
    assert all(len(record["detections"]) == 3 for record in result["images"])


def test_generic_video_sampler_declares_manifest_categories_without_frames(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "videos"
    source.mkdir()
    (source / "broken.mp4").touch()

    class ClosedCapture:
        def __init__(self, _path: str):
            pass

        def get(self, _property):
            return 0.0

        def isOpened(self):
            return False

        def release(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(CAP_PROP_FPS=5, VideoCapture=ClosedCapture, imwrite=None),
    )
    weights = tmp_path / "weights.pt"
    weights.touch()
    detector = YoloDetectorAdapter(
        manifest(detector_backend="yolo", class_names={"0": "fox", "1": "deer"}),
        weights,
        SimpleNamespace(get_python=lambda _env: Path("python")),
        runner=object(),
    )
    output_path = tmp_path / "output.json"
    detector.detect_videos_to_json(
        video_folder=source, output_json=output_path, fps=1.0, confidence_threshold=0.1
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["detection_categories"] == {"0": "fox", "1": "deer"}
    assert result["images"][0]["failure"] == "Unable to open video"
