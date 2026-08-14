"""
Tests for taxonomy fetching in app.ml.catalog_updater.

Taxonomy lives in the model's HuggingFace repo, not in the catalog, so
`write_manifest` is what pulls it down. Two failure modes are pinned here
because both shipped and both were silent:

1. The repo was hardcoded to the default org, so the one model that
   overrides `hf_repo` fetched from the wrong place and 404'd.
2. The fetch only ran when the model directory was first created, so a
   stub whose taxonomy never landed stayed broken forever.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.ml.catalog_updater import ModelCatalogUpdater
from app.ml.schemas.model_manifest import resolve_hf_repo


def _entry(model_id: str = "TEST-v1", **overrides) -> dict:
    """A minimal catalog entry, shaped like a real models.json row."""
    entry = {
        "model_id": model_id,
        "friendly_name": "Test Model",
        "env": "pytorch",
        "model_fname": "weights.pt",
        "description": "...",
        "developer": "x",
        "info_url": "https://example.com",
        "min_app_version": "0.1.0",
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def updater(tmp_path: Path) -> ModelCatalogUpdater:
    return ModelCatalogUpdater(models_dir=tmp_path / "models")


# --------------------------------------------------------------------
# resolve_hf_repo
# --------------------------------------------------------------------


def test_resolve_hf_repo_falls_back_to_default_org():
    assert resolve_hf_repo("NAM-ADS-v1") == "Addax-Data-Science/NAM-ADS-v1"
    assert resolve_hf_repo("NAM-ADS-v1", None) == "Addax-Data-Science/NAM-ADS-v1"


def test_resolve_hf_repo_honours_explicit_override():
    assert resolve_hf_repo("X", "other-org/custom-repo") == "other-org/custom-repo"


# --------------------------------------------------------------------
# download_taxonomy: which repo does it ask for?
# --------------------------------------------------------------------


def test_download_taxonomy_uses_explicit_hf_repo(
    updater: ModelCatalogUpdater, tmp_path: Path
):
    """An explicit hf_repo must win over the default-org convention."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()

    with patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"csv"
        updater.download_taxonomy("AHDRIFT-v1", model_dir, "other-org/AHDRIFT-v1")

    url = mock_open.call_args[0][0]
    assert "other-org/AHDRIFT-v1" in url
    assert "Addax-Data-Science" not in url
    assert (model_dir / "taxonomy.csv").read_bytes() == b"csv"


def test_download_taxonomy_goes_through_the_mirror(
    updater: ModelCatalogUpdater, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    This was the one request in the app that hardcoded huggingface.co,
    so on a network that blocks it every classification model produced a
    failing request per launch even with a mirror configured.
    """
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADDAXAI_HF_ENDPOINT", "https://hf-mirror.com")

    with patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"csv"
        updater.download_taxonomy("TEST-v1", model_dir)

    url = mock_open.call_args[0][0]
    assert url.startswith("https://hf-mirror.com/")
    assert "huggingface.co" not in url


def test_download_taxonomy_uses_the_real_host_without_a_mirror(
    updater: ModelCatalogUpdater, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ADDAXAI_HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    with patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"csv"
        updater.download_taxonomy("TEST-v1", model_dir)

    assert mock_open.call_args[0][0].startswith("https://huggingface.co/")


def test_download_taxonomy_defaults_to_addax_org(
    updater: ModelCatalogUpdater, tmp_path: Path
):
    model_dir = tmp_path / "m"
    model_dir.mkdir()

    with patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"csv"
        updater.download_taxonomy("NAM-ADS-v1", model_dir, None)

    assert "Addax-Data-Science/NAM-ADS-v1" in mock_open.call_args[0][0]


# --------------------------------------------------------------------
# write_manifest: when is the fetch attempted?
# --------------------------------------------------------------------


def test_taxonomy_fetched_for_new_cls_model(updater: ModelCatalogUpdater):
    with patch.object(updater, "download_taxonomy") as mock_dl:
        assert updater.write_manifest("cls", _entry()) == "created"
    mock_dl.assert_called_once()


def test_taxonomy_fetch_passes_hf_repo_through(updater: ModelCatalogUpdater):
    """The override must survive the trip from catalog entry to fetch."""
    entry = _entry(hf_repo="Addax-Data-Science/AHDRIFT-v1")
    with patch.object(updater, "download_taxonomy") as mock_dl:
        updater.write_manifest("cls", entry)

    assert mock_dl.call_args[0][2] == "Addax-Data-Science/AHDRIFT-v1"


def test_taxonomy_retried_when_missing_from_unchanged_stub(
    updater: ModelCatalogUpdater,
):
    """
    The regression that stranded AHDRIFT-v1. An existing stub whose
    taxonomy never landed has a perfectly unchanged manifest, so an
    early "unchanged" return would skip the fetch and the model would
    never self-heal.
    """
    entry = _entry()
    with patch.object(updater, "download_taxonomy"):
        updater.write_manifest("cls", entry)  # creates the stub, no taxonomy

    with patch.object(updater, "download_taxonomy") as mock_dl:
        assert updater.write_manifest("cls", entry) == "unchanged"
    mock_dl.assert_called_once()


def test_taxonomy_not_refetched_when_already_on_disk(
    updater: ModelCatalogUpdater, tmp_path: Path
):
    """Present file means no request; a catalog refresh is not HF drift."""
    entry = _entry()
    with patch.object(updater, "download_taxonomy"):
        updater.write_manifest("cls", entry)
    (tmp_path / "models" / "cls" / "TEST-v1" / "taxonomy.csv").write_text("x")

    with patch.object(updater, "download_taxonomy") as mock_dl:
        updater.write_manifest("cls", entry)
    mock_dl.assert_not_called()


def test_taxonomy_fetched_when_manifest_refreshes(updater: ModelCatalogUpdater):
    """A changed manifest still returns "updated", fetch or no fetch."""
    with patch.object(updater, "download_taxonomy"):
        updater.write_manifest("cls", _entry())

    with patch.object(updater, "download_taxonomy") as mock_dl:
        result = updater.write_manifest("cls", _entry(friendly_name="Renamed"))

    assert result == "updated"
    mock_dl.assert_called_once()


def test_unchanged_manifest_is_not_rewritten(updater: ModelCatalogUpdater, tmp_path: Path):
    """
    Falling through to the taxonomy check must not start rewriting
    manifests that haven't moved. "Unchanged" is judged on parsed JSON,
    not bytes, so a differently-formatted but equivalent file is left
    exactly as it is.
    """
    entry = _entry()
    with patch.object(updater, "download_taxonomy"):
        updater.write_manifest("cls", entry)

    path = tmp_path / "models" / "cls" / "TEST-v1" / "manifest.json"
    reformatted = json.dumps(entry, indent=8, sort_keys=True)
    path.write_text(reformatted)

    with patch.object(updater, "download_taxonomy"):
        assert updater.write_manifest("cls", entry) == "unchanged"
    assert path.read_text() == reformatted


@pytest.mark.parametrize("model_type", ["det", "emb"])
def test_taxonomy_never_fetched_for_non_cls(
    updater: ModelCatalogUpdater, model_type: str
):
    with patch.object(updater, "download_taxonomy") as mock_dl:
        updater.write_manifest(model_type, _entry())
    mock_dl.assert_not_called()


def test_missing_taxonomy_is_not_fatal(updater: ModelCatalogUpdater):
    """A 404 (or any network error) must leave the stub usable."""
    with patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open:
        mock_open.side_effect = OSError("network down")
        assert updater.write_manifest("cls", _entry()) == "created"
