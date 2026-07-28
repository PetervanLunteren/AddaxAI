"""Tests for drift detection in app.ml.catalog_updater."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ml.catalog_updater import ModelCatalogUpdater


def _write_local_manifest(
    models_dir: Path,
    model_type: str,
    model_id: str,
    *,
    hf_repo: str | None = "Addax-Data-Science/test-model",
    hf_revision_sha: str | None = None,
) -> Path:
    """
    Plant a `manifest.json` on disk that looks like what the download
    flow would write. Returns the manifest path so tests can mutate it.
    """
    model_dir = models_dir / model_type / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": model_id,
        "friendly_name": "Test Model",
        "emoji": "🧪",
        "env": "addaxai-base",
        "model_fname": "weights.pt",
        "hf_repo": hf_repo,
        "description": "...",
        "developer": "x",
        "info_url": "https://example.com",
        "min_app_version": "0.1.0",
    }
    if hf_revision_sha is not None:
        manifest["hf_revision_sha"] = hf_revision_sha
    path = model_dir / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _make_updater(tmp_path: Path) -> ModelCatalogUpdater:
    return ModelCatalogUpdater(models_dir=tmp_path / "models")


@pytest.fixture
def updater(tmp_path: Path) -> ModelCatalogUpdater:
    return _make_updater(tmp_path)


def test_drift_check_returns_true_when_remote_sha_differs(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """Recorded SHA != upstream SHA → drift detected."""
    _write_local_manifest(
        tmp_path / "models", "cls", "test-model",
        hf_revision_sha="abc123" * 7,  # 42 chars, doesn't matter
    )
    fake_info = MagicMock(sha="def456" * 7)
    with patch(
        "huggingface_hub.HfApi.model_info", return_value=fake_info
    ):
        manifest = {
            "model_id": "test-model",
            "friendly_name": "X",
            "emoji": "🧪",
        }
        result = updater.check_model_drift("cls", manifest)
    assert result is True


def test_drift_check_returns_false_when_shas_match(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """Recorded SHA == upstream SHA → no drift."""
    sha = "abc123" * 7
    _write_local_manifest(
        tmp_path / "models", "cls", "test-model",
        hf_revision_sha=sha,
    )
    fake_info = MagicMock(sha=sha)
    with patch(
        "huggingface_hub.HfApi.model_info", return_value=fake_info
    ):
        manifest = {
            "model_id": "test-model",
            "friendly_name": "X",
            "emoji": "🧪",
        }
        result = updater.check_model_drift("cls", manifest)
    assert result is False


def test_drift_check_returns_none_when_no_local_sha(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """Legacy install with no recorded SHA → unknown but valid."""
    _write_local_manifest(
        tmp_path / "models", "cls", "test-model", hf_revision_sha=None
    )
    manifest = {
        "model_id": "test-model",
        "friendly_name": "X",
        "emoji": "🧪",
    }
    # HfApi should not even be called; assert that explicitly.
    with patch("huggingface_hub.HfApi.model_info") as mock_info:
        result = updater.check_model_drift("cls", manifest)
    assert result is None
    mock_info.assert_not_called()


def test_drift_check_returns_none_when_manifest_missing(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """Catalog stub the user never downloaded → skip the check."""
    manifest = {
        "model_id": "never-installed",
        "friendly_name": "X",
        "emoji": "🧪",
    }
    result = updater.check_model_drift("cls", manifest)
    assert result is None


def test_drift_check_returns_none_when_hf_api_fails(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """Network error / 404 / auth → silent skip, no startup crash."""
    _write_local_manifest(
        tmp_path / "models", "cls", "test-model",
        hf_revision_sha="abc123" * 7,
    )
    with patch(
        "huggingface_hub.HfApi.model_info",
        side_effect=ConnectionError("offline"),
    ):
        manifest = {
            "model_id": "test-model",
            "friendly_name": "X",
            "emoji": "🧪",
        }
        result = updater.check_model_drift("cls", manifest)
    assert result is None


def test_drift_check_returns_none_when_remote_has_no_sha(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """If the HF response is missing a sha attribute, treat as unknown."""
    _write_local_manifest(
        tmp_path / "models", "cls", "test-model",
        hf_revision_sha="abc123" * 7,
    )
    # Use a real object whose `sha` attribute is None.
    fake_info = MagicMock()
    fake_info.sha = None
    with patch(
        "huggingface_hub.HfApi.model_info", return_value=fake_info
    ):
        manifest = {
            "model_id": "test-model",
            "friendly_name": "X",
            "emoji": "🧪",
        }
        result = updater.check_model_drift("cls", manifest)
    assert result is None
