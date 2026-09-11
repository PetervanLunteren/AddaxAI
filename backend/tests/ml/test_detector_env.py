"""A detector runs in the environment its catalog entry names.

Detection used to be pinned to env-addaxai-base inside both detector
classes, which is what kept every detector in one YAML. The underwater
detectors live in env-marine (the rfdetr package must not land in the
base env), so the worker passes the manifest's `env` through, exactly
as classifiers already do.
"""

from pathlib import Path

import pytest

from app.ml.inference.megadetector import MegaDetectorV1000
from app.ml.inference.video_detector import VideoDetectionModel
from app.ml.schemas.model_manifest import ModelManifest


class _RecordingEnvManager:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def get_python(self, env_name: str) -> Path:
        self.asked.append(env_name)
        return Path("/usr/bin/python3")


@pytest.fixture
def weights(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"pt")
    return p


@pytest.mark.parametrize("env", ["addaxai-base", "marine"])
def test_image_detector_asks_for_the_manifests_env(weights, env):
    env_manager = _RecordingEnvManager()
    MegaDetectorV1000(weights, env_manager, env_name=env)
    assert env_manager.asked == [f"env-{env}"]


@pytest.mark.parametrize("env", ["addaxai-base", "marine"])
def test_video_detector_asks_for_the_manifests_env(weights, env):
    env_manager = _RecordingEnvManager()
    VideoDetectionModel(weights, env_manager, env_name=env)
    assert env_manager.asked == [f"env-{env}"]


def test_env_name_is_required(weights):
    """No default: a detector that does not say where it runs is a
    configuration error, not a base-env run."""
    with pytest.raises(TypeError):
        MegaDetectorV1000(weights, _RecordingEnvManager())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        VideoDetectionModel(weights, _RecordingEnvManager())  # type: ignore[call-arg]


def _manifest(**overrides) -> ModelManifest:
    fields = dict(
        model_id="X",
        friendly_name="X",
        env="marine",
        model_fname="x.pt",
        description="x",
        developer="x",
        info_url="https://example.org",
        min_app_version="7.7.0",
    )
    fields.update(overrides)
    return ModelManifest(**fields)


def test_manifest_detector_fields_default_to_megadetector_without_the_filter():
    """Every shipped MegaDetector entry predates these fields, so the
    defaults must describe MegaDetector: it names its own classes, and
    there is no shark false-positive filter."""
    m = _manifest()
    assert m.class_mapping is None
    assert m.track_filter is False


def test_manifest_accepts_a_class_mapping_and_the_filter():
    m = _manifest(class_mapping={"0": "elasmobranch"}, track_filter=True)
    assert m.class_mapping == {"0": "elasmobranch"}
    assert m.track_filter is True


def test_manifest_ignores_a_retired_field():
    """`detector_runtime` chose between two loaders; there is one now, so
    an installed manifest.json written before the catalog dropped the
    field must still parse. The next catalog sync rewrites it away."""
    m = _manifest(detector_runtime="ultralytics")
    assert not hasattr(m, "detector_runtime")
