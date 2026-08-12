"""
Tests for what a model download leaves behind when it does not complete.

Two rules, and the split between them is the whole point:

- A **failed** download leaves everything alone. Files land at their final
  path only once complete and size-verified, so a retry fetches what is
  missing instead of starting over.
- A **cancelled** download throws the downloaded files away, because the
  user asked it to stop, but keeps `manifest.json`.

`manifest.json` is what makes both rules matter. It is written from
models.json by the catalog updater, no HF repo ships one, so a download can
delete it and nothing short of the next launch's sync puts it back. Losing
it drops the model out of the catalog, which surfaces far from here as
"Classification model '<id>' not found" when a project is created.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.job_cancellation import JobCancelledError
from app.ml.model_storage import ModelStorage
from app.ml.schemas.model_manifest import ModelManifest

MANIFEST_JSON = {
    "model_id": "TEST-v1",
    "friendly_name": "Test model",
    "env": "pytorch",
    "model_fname": "weights.pt",
    "description": "...",
    "developer": "x",
    "info_url": "https://example.org",
    "min_app_version": "7.0.1",
}


def _manifest(**overrides: object) -> ModelManifest:
    m = ModelManifest(**{**MANIFEST_JSON, **overrides})
    m.model_category = overrides.pop("model_category", "classification")  # type: ignore[arg-type]
    return m


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A catalog stub: manifest only, nothing downloaded yet."""
    d = tmp_path / "models" / "cls" / "TEST-v1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(MANIFEST_JSON))
    return d


def _storage(model_dir: Path) -> ModelStorage:
    return ModelStorage(models_dir=model_dir.parent.parent)


def _partial(model_dir: Path):
    """A download where the big file lands and one small file fails."""

    def _run(**kwargs: object) -> bool:
        (model_dir / "weights.pt").write_bytes(b"w" * 2048)
        (model_dir / "taxonomy.csv").write_text("model_class,class\n")
        return False  # inference.py failed

    return _run


def test_failed_download_keeps_the_manifest(model_dir: Path) -> None:
    """
    The 2026-08-12 bug. A failed download used to rmtree the directory, so
    the model vanished from the catalog even though the retry restored every
    file the repo actually ships.
    """
    with patch(
        "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo",
        side_effect=_partial(model_dir),
    ):
        with pytest.raises(RuntimeError):
            _storage(model_dir).download_weights(_manifest())

    assert json.loads((model_dir / "manifest.json").read_text()) == MANIFEST_JSON


def test_failed_download_keeps_the_files_that_did_land(model_dir: Path) -> None:
    """
    The expensive half of the same bug. One unresolvable 12 KB file deleted a
    1.13 GB weights file that had downloaded perfectly, so the retry paid for
    the whole model again. Keeping completed files lets `download_file`'s
    size check skip them (see test_hf_downloader.py).
    """
    with patch(
        "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo",
        side_effect=_partial(model_dir),
    ):
        with pytest.raises(RuntimeError):
            _storage(model_dir).download_weights(_manifest())

    assert (model_dir / "weights.pt").read_bytes() == b"w" * 2048
    assert (model_dir / "taxonomy.csv").exists()


def test_failed_repair_download_keeps_installed_weights(tmp_path: Path) -> None:
    """
    The variant that could destroy gigabytes. A model with its weights but
    without the architecture source reports not-ready, so Prepare re-runs the
    download over a full install. A failure there must not take the weights.
    """
    models_dir = tmp_path / "models"
    model_dir = models_dir / "emb" / "TEST-v1"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(json.dumps(MANIFEST_JSON))
    (model_dir / "weights.pt").write_bytes(b"w" * 4096)

    manifest = _manifest(model_category="embedding", torch_hub_model="dinov2_vitl14")
    storage = ModelStorage(models_dir=models_dir)
    # hubconf.py missing, so the model is not ready and Prepare downloads.
    assert storage.check_weights_ready(manifest) is False

    with patch(
        "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo",
        side_effect=lambda **kwargs: False,
    ):
        with pytest.raises(RuntimeError):
            storage.download_weights(manifest)

    assert (model_dir / "weights.pt").read_bytes() == b"w" * 4096
    assert (model_dir / "manifest.json").exists()


def test_cancelled_download_clears_files_but_keeps_the_manifest(
    model_dir: Path,
) -> None:
    """
    Cancel means throw the download away, so the next attempt starts clean.
    It must not also mean "forget this model exists": cancelling a prepare
    would otherwise reproduce the failure bug through a different door.
    """
    def _cancelled(**kwargs: object) -> bool:
        (model_dir / "weights.pt").write_bytes(b"w" * 2048)
        (model_dir / "vendored").mkdir()
        (model_dir / "vendored" / "layer.py").write_text("x = 1\n")
        raise JobCancelledError()

    with patch(
        "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo",
        side_effect=_cancelled,
    ):
        with pytest.raises(JobCancelledError):
            _storage(model_dir).download_weights(_manifest())

    assert [p.name for p in model_dir.iterdir()] == ["manifest.json"]
    assert json.loads((model_dir / "manifest.json").read_text()) == MANIFEST_JSON


def test_an_already_installed_model_is_not_redownloaded(model_dir: Path) -> None:
    """Guards the early return, so the rules above are only ever reached
    when there is genuinely something to fetch."""
    (model_dir / "weights.pt").write_bytes(b"w" * 2048)

    with patch(
        "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo"
    ) as mock_download:
        _storage(model_dir).download_weights(_manifest())

    mock_download.assert_not_called()
