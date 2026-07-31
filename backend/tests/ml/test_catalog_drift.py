"""
Tests for how ModelCatalogUpdater.sync() reports models with files to update.

The per-file comparison itself lives in tests/ml/test_model_update.py. What
matters here is which models sync() even asks about, and that a failure to
ask never takes startup down.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ml.catalog_updater import ModelCatalogUpdater

ENTRY = {
    "model_id": "TEST-v1",
    "friendly_name": "Test model",
    "emoji": "🧪",
    "env": "addaxai-base",
    "model_fname": "weights.pt",
    "description": "...",
    "developer": "x",
    "info_url": "https://example.com",
    "min_app_version": "0.1.0",
}
CATALOG = {"models": {"det": [], "cls": [ENTRY], "emb": []}}


@pytest.fixture
def updater(tmp_path: Path) -> ModelCatalogUpdater:
    return ModelCatalogUpdater(models_dir=tmp_path / "models")


def _install(models_dir: Path, *, with_weights: bool) -> Path:
    """Put a model on disk the way a previous launch would have left it."""
    model_dir = models_dir / "cls" / "TEST-v1"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(json.dumps(ENTRY))
    (model_dir / "taxonomy.csv").write_text("model_class,class\n")
    if with_weights:
        (model_dir / "weights.pt").write_bytes(b"w")
    return model_dir


async def _sync(updater: ModelCatalogUpdater) -> dict:
    """Run sync() with the network and the env drift check stubbed out."""
    with (
        patch.object(updater, "fetch_catalog", return_value=CATALOG),
        patch.object(updater, "download_taxonomy"),
        patch("app.ml.environment_manager.EnvironmentManager") as mock_env,
    ):
        mock_env.return_value.check_yaml_drift.return_value = False
        return await updater.sync()


async def test_stub_without_weights_is_never_checked(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """
    A catalog stub is a manifest with no model next to it. Asking HuggingFace
    about it would be one HTTP call per launch for every model the user never
    downloaded.
    """
    _install(tmp_path / "models", with_weights=False)

    with patch("huggingface_hub.HfApi.model_info") as mock_info:
        result = await _sync(updater)

    mock_info.assert_not_called()
    assert result["drifted_models"] == []


async def test_installed_model_with_stale_files_is_reported(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """The wire shape is exactly three keys: the file names stay in the log."""
    _install(tmp_path / "models", with_weights=True)

    with patch(
        "app.ml.catalog_updater.find_stale_files", return_value=["inference.py"]
    ):
        result = await _sync(updater)

    assert result["drifted_models"] == [
        {"model_id": "TEST-v1", "friendly_name": "Test model", "emoji": "🧪"}
    ]


async def test_model_in_sync_is_not_reported(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    _install(tmp_path / "models", with_weights=True)

    with patch("app.ml.catalog_updater.find_stale_files", return_value=[]):
        result = await _sync(updater)

    assert result["drifted_models"] == []


async def test_unreachable_huggingface_does_not_fail_the_sync(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """
    Offline is a normal state for plenty of users. Exercised through the real
    code path rather than by mocking the guard that is supposed to catch it.
    """
    _install(tmp_path / "models", with_weights=True)

    with patch(
        "huggingface_hub.HfApi.model_info", side_effect=ConnectionError("offline")
    ):
        result = await _sync(updater)

    assert result["drifted_models"] == []
    assert "error" not in result


async def test_fresh_install_is_never_checked(updater: ModelCatalogUpdater) -> None:
    """First launch has nothing on disk that could be out of date."""
    with patch("huggingface_hub.HfApi.model_info") as mock_info:
        result = await _sync(updater)

    mock_info.assert_not_called()
    assert result["drifted_models"] == []
    assert result["new_models"] == []


async def test_manifest_is_not_rewritten_on_a_second_launch(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    """
    Nothing local is stored in manifest.json any more, so a second launch
    finds nothing to rewrite and reports nothing as refreshed. This is the
    end of the churn the old recorded-SHA scheme caused.
    """
    _install(tmp_path / "models", with_weights=True)

    with patch("app.ml.catalog_updater.find_stale_files", return_value=None):
        first = await _sync(updater)
        second = await _sync(updater)

    assert first["refreshed_models"] == []
    assert second["refreshed_models"] == []


async def test_hf_is_asked_once_per_installed_model(
    updater: ModelCatalogUpdater, tmp_path: Path
) -> None:
    _install(tmp_path / "models", with_weights=True)

    with patch(
        "huggingface_hub.HfApi.model_info", return_value=MagicMock(siblings=[])
    ) as mock_info:
        await _sync(updater)

    assert mock_info.call_count == 1
