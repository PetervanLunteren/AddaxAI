"""RT-DETR adapters and the constrained RT-DETRv2 config bridge."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from app.ml.inference import detector_subprocess
from app.ml.inference.detector_backend import (
    RTDETRDetectorAdapter,
    RTDETRv2DetectorAdapter,
    create_detector,
)
from app.ml.inference.rtdetrv2_config import load_rtdetrv2_config
from app.ml.schemas.model_manifest import ModelManifest


def manifest(**overrides):
    values = {
        "model_id": "local-rtdetr",
        "friendly_name": "Local RT-DETR",
        "env": "rtdetr",
        "model_fname": "weights.pth",
        "description": "test",
        "developer": "test",
        "info_url": "https://example.invalid",
        "min_app_version": "0",
        "local_only": True,
    }
    values.update(overrides)
    return ModelManifest(**values)


class StubEnvironment:
    def get_python(self, _name: str) -> Path:
        return Path("isolated-python")


def test_manifest_validates_rtdetr_variants_and_local_only():
    parsed = manifest(detector_backend="rtdetr")
    assert parsed.detector_model_variant == "MDV6-apa-rtdetr-c"
    assert manifest(
        detector_backend="rtdetr", detector_model_variant="MDV6-apa-rtdetr-e"
    ).detector_model_variant == "MDV6-apa-rtdetr-e"
    with pytest.raises(ValueError, match="detector_model_variant"):
        manifest(detector_backend="rtdetr", detector_model_variant="custom")
    with pytest.raises(ValueError, match="local_only"):
        manifest(detector_backend="rtdetr", local_only=False)


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.yml", r"C:\models\config.yml", "/tmp/config.yml"],
)
def test_rtdetrv2_manifest_rejects_unsafe_config_path(unsafe_path: str):
    with pytest.raises(ValueError, match="safe relative path"):
        manifest(
            detector_backend="rtdetrv2",
            detector_config_fname=unsafe_path,
        )


def test_rtdetrv2_manifest_requires_config_and_local_only():
    with pytest.raises(ValueError, match="detector_config_fname is required"):
        manifest(detector_backend="rtdetrv2")
    with pytest.raises(ValueError, match="local_only"):
        manifest(
            detector_backend="rtdetrv2",
            detector_config_fname="rtdetrv2.yml",
            local_only=False,
        )
    parsed = manifest(
        detector_backend="rtdetrv2",
        detector_config_fname="configs/rtdetrv2.yml",
        class_names=["animal", "person"],
    )
    assert parsed.class_names == {"0": "animal", "1": "person"}


def test_factory_selects_both_rtdetr_backends_without_loading_packages(tmp_path: Path):
    weight = tmp_path / "weights.pth"
    weight.touch()
    runner = object()

    rtdetr = create_detector(
        manifest(detector_backend="rtdetr"), weight, StubEnvironment(), runner=runner
    )
    rtdetrv2 = create_detector(
        manifest(detector_backend="rtdetrv2", detector_config_fname="model.yml"),
        weight,
        StubEnvironment(),
        runner=runner,
    )
    assert isinstance(rtdetr, RTDETRDetectorAdapter)
    assert isinstance(rtdetrv2, RTDETRv2DetectorAdapter)
    assert rtdetr.runner is runner
    assert rtdetrv2.runner is runner


def _make_rtdetrv2_roots(tmp_path: Path) -> tuple[Path, Path]:
    model_root = tmp_path / "model-pack"
    source_root = tmp_path / "pinned-source"
    (source_root / "src" / "core").mkdir(parents=True)
    (source_root / "src" / "core" / "__init__.py").touch()
    (source_root / "configs" / "base.yml").parent.mkdir(parents=True)
    (source_root / "configs" / "base.yml").write_text(
        "nested:\n  source: true\n  override: source\n", encoding="utf-8"
    )
    model_root.mkdir()
    return model_root, source_root


def test_rtdetrv2_config_safely_flattens_model_and_pinned_source_includes(tmp_path: Path):
    model_root, source_root = _make_rtdetrv2_roots(tmp_path)
    (model_root / "base.yml").write_text(
        "nested:\n  model: true\n  override: model\n", encoding="utf-8"
    )
    config_path = model_root / "model.yml"
    config_path.write_text(
        "__include__:\n"
        "  - base.yml\n"
        "  - ../third_party/rtdetr/rtdetrv2_pytorch/configs/base.yml\n"
        "nested:\n  override: root\nmodel: RTDETR\npostprocessor: RTDETRPostProcessor\n",
        encoding="utf-8",
    )

    config = load_rtdetrv2_config(
        config_path, model_root=model_root, source_root=source_root
    )

    assert config == {
        "nested": {"model": True, "override": "root", "source": True},
        "model": "RTDETR",
        "postprocessor": "RTDETRPostProcessor",
    }


@pytest.mark.parametrize(
    "include_text",
    [
        "../escape.yml",
        r"C:\outside.yml",
        "~/outside.yml",
    ],
)
def test_rtdetrv2_config_rejects_includes_outside_allowlisted_roots(
    tmp_path: Path, include_text: str
):
    model_root, source_root = _make_rtdetrv2_roots(tmp_path)
    config_path = model_root / "model.yml"
    config_path.write_text(f"__include__:\n  - '{include_text}'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not allowed|escapes allowed roots"):
        load_rtdetrv2_config(config_path, model_root=model_root, source_root=source_root)


def test_rtdetrv2_config_rejects_cycles_and_python_yaml_tags(tmp_path: Path):
    model_root, source_root = _make_rtdetrv2_roots(tmp_path)
    root_config = model_root / "root.yml"
    child_config = model_root / "child.yml"
    root_config.write_text("__include__: [child.yml]\n", encoding="utf-8")
    child_config.write_text("__include__: [root.yml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="include cycle"):
        load_rtdetrv2_config(root_config, model_root=model_root, source_root=source_root)

    root_config.write_text(
        "value: !!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8"
    )
    with pytest.raises(yaml.YAMLError):
        load_rtdetrv2_config(root_config, model_root=model_root, source_root=source_root)


def test_rtdetrv2_state_extraction_supports_ema_module_and_model_fallback():
    ema_state = {"weight": object()}
    model_state = {"bias": object()}
    assert detector_subprocess._extract_rtdetrv2_state(
        {"ema": {"module": ema_state}, "model": model_state}
    ) is ema_state
    assert detector_subprocess._extract_rtdetrv2_state({"model": model_state}) is model_state
    with pytest.raises(ValueError, match="ema.module or model"):
        detector_subprocess._extract_rtdetrv2_state({"ema": {"module": []}})
    with pytest.raises(ValueError, match="keys must be strings"):
        detector_subprocess._extract_rtdetrv2_state({"model": {1: object()}})


def test_rtdetrv2_detection_output_converts_boxes_and_filters_invalid_scores():
    output = detector_subprocess._rtdetrv2_detections(
        [4, 2, 8],
        [[10, 20, 60, 80], [0, 0, 50, 50], [0, 0, 10, 10]],
        [0.9, 0.49, float("nan")],
        width=200,
        height=100,
        threshold=0.5,
    )

    assert output == [
        {"bbox": [0.05, 0.2, 0.25, 0.6], "conf": 0.9, "class_id": "4"}
    ]


def test_rtdetr_child_uses_local_pywildlife_weights_without_pretrained_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}
    weights = tmp_path / "weights.pth"
    weights.touch()
    image = tmp_path / "image.jpg"
    image.touch()

    class FakeModel:
        CLASS_NAMES = {3: "red fox"}

        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def single_image_detection(self, image_path, *, det_conf_thres):
            captured["detect"] = (image_path, det_conf_thres)
            return {
                "detections": SimpleNamespace(
                    xyxy=[[10, 5, 60, 45]], confidence=[0.8], class_id=[3]
                )
            }

    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    image_stub = SimpleNamespace(open=lambda _path: SimpleNamespace(size=(100, 80)))
    package = ModuleType("PytorchWildlife")
    models_module = ModuleType("PytorchWildlife.models")
    detection_module = ModuleType("PytorchWildlife.models.detection")
    base_module = ModuleType(
        "PytorchWildlife.models.detection.rtdetr_apache.rtdetr_apache_base"
    )

    def yaml_config_factory(_config_path, **_kwargs):
        return SimpleNamespace(yaml_cfg={"PResNet": {"pretrained": True}})

    base_module.YAMLConfig = yaml_config_factory
    detection_module.MegaDetectorV6Apache = FakeModel
    models_module.detection = detection_module
    package.models = models_module
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=image_stub))
    monkeypatch.setitem(sys.modules, "PytorchWildlife", package)
    monkeypatch.setitem(sys.modules, "PytorchWildlife.models", models_module)
    monkeypatch.setitem(sys.modules, "PytorchWildlife.models.detection", detection_module)
    monkeypatch.setitem(
        sys.modules,
        "PytorchWildlife.models.detection.rtdetr_apache.rtdetr_apache_base",
        base_module,
    )

    result = detector_subprocess._run_rtdetr(
        weights, [image], 0.25, "MDV6-apa-rtdetr-c"
    )

    assert captured["init"] == {
        "weights": str(weights),
        "device": "cpu",
        "pretrained": False,
        "version": "MDV6-apa-rtdetr-c",
    }
    assert captured["detect"] == (str(image), 0.25)
    assert base_module.YAMLConfig("bundled-config.yml").yaml_cfg["PResNet"]["pretrained"] is False
    assert result[0]["class_names"] == {"3": "red fox"}
    assert result[0]["detections"] == [
        {"bbox": [0.1, 0.0625, 0.5, 0.5], "conf": 0.8, "class_id": "3"}
    ]
