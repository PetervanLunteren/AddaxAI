"""
The model catalog must survive a blocked catalog host.

manifest.json is written from the catalog and from nowhere else, and
ManifestManager skips a model directory that has none. So a first launch
on a network that blocks raw.githubusercontent.com used to download the
weights and then show no models at all. The copy shipped in the app is
the fallback that closes that.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.ml.catalog_updater import ModelCatalogUpdater, _bundled_catalog_path, merge_catalogs
from app.ml.manifest_manager import ManifestManager

_REPO_CATALOG = Path(__file__).resolve().parents[3] / "models.json"


@pytest.fixture
def updater(tmp_path: Path) -> ModelCatalogUpdater:
    return ModelCatalogUpdater(models_dir=tmp_path / "models")


def test_the_bundled_catalog_is_found_from_source():
    """Frozen builds get it from backend.spec; this is the dev path."""
    assert _bundled_catalog_path() == _REPO_CATALOG


def test_an_unreachable_host_falls_back_to_the_bundled_catalog(
    updater: ModelCatalogUpdater,
):
    with patch(
        "app.ml.catalog_updater.urllib.request.urlopen",
        side_effect=OSError("blocked"),
    ):
        catalog = updater.fetch_catalog()

    assert catalog is not None
    assert catalog == json.loads(_REPO_CATALOG.read_text())


def test_a_malformed_response_falls_back_too(updater: ModelCatalogUpdater):
    """A proxy answering with a login page is not the same as no answer."""
    with patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            b"<html>sign in</html>"
        )
        catalog = updater.fetch_catalog()

    assert catalog is not None
    assert "det" in catalog["models"]


def _remote_with(**changes):
    """The shipped catalog as the remote would serve it, with edits."""
    remote = json.loads(_REPO_CATALOG.read_text())
    for entry in remote["models"]["det"]:
        if entry["model_id"] == "MD5A-0-0":
            entry.update(changes)
    return remote


def _fetch(updater: ModelCatalogUpdater, remote: dict) -> dict:
    with patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            json.dumps(remote).encode()
        )
        catalog = updater.fetch_catalog()
    assert catalog is not None
    return catalog


def test_a_working_host_still_wins_on_every_field_it_carries(updater: ModelCatalogUpdater):
    """The fallback is a fallback, not a cache that shadows upstream: the
    remote's description, and a model only it knows, come through."""
    remote = _remote_with(description="fresh text from main")
    remote["models"]["det"].append({**remote["models"]["det"][0], "model_id": "NEW-1-0"})
    catalog = _fetch(updater, remote)
    by_id = {e["model_id"]: e for e in catalog["models"]["det"]}
    assert by_id["MD5A-0-0"]["description"] == "fresh text from main"
    assert "NEW-1-0" in by_id


def test_a_remote_entry_missing_a_field_keeps_the_shipped_value(updater: ModelCatalogUpdater):
    """A field this build requires always exists for the models it shipped
    with, whatever main says. On 2026-09-14 a sync from main, whose
    models.json had no `domain` yet, rewrote every installed manifest
    without it and every video run failed."""
    remote = _remote_with()
    for entry in remote["models"]["det"]:
        entry.pop("domain", None)
        entry.pop("primary_class", None)
    catalog = _fetch(updater, remote)
    assert all(e.get("domain") for e in catalog["models"]["det"])
    assert all(e.get("primary_class") for e in catalog["models"]["det"])
    assert {e["model_id"] for e in catalog["models"]["det"]} == {
        e["model_id"] for e in json.loads(_REPO_CATALOG.read_text())["models"]["det"]
    }


def test_merge_is_pure_and_keeps_every_model_of_both_sides():
    bundled = {"models": {"det": [{"model_id": "A", "domain": "camera_trap", "x": 1}], "cls": []}}
    remote = {
        "models": {
            "det": [{"model_id": "A", "x": 2}, {"model_id": "B", "x": 3}],
            "cls": [],
            "emb": [],
        }
    }
    merged = merge_catalogs(bundled, remote)
    assert merged["models"]["det"] == [
        {"model_id": "A", "domain": "camera_trap", "x": 2},
        {"model_id": "B", "x": 3},
    ]
    assert merged["models"]["emb"] == []
    assert bundled["models"]["det"][0] == {"model_id": "A", "domain": "camera_trap", "x": 1}


@pytest.fixture
def offline(tmp_path: Path) -> ModelCatalogUpdater:
    """The updater as ADDAXAI_DISABLE_MODEL_UPDATES builds it."""
    return ModelCatalogUpdater(models_dir=tmp_path / "models", offline=True)


async def _sync_offline(updater: ModelCatalogUpdater) -> None:
    """Sync with every network door watched: none may open."""
    with (
        patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open,
        patch("huggingface_hub.HfApi.model_info") as mock_info,
    ):
        await updater.sync()
    mock_open.assert_not_called()
    mock_info.assert_not_called()


def _shipped_ids() -> dict[str, set[str]]:
    models = json.loads(_REPO_CATALOG.read_text())["models"]
    return {t: {e["model_id"] for e in entries} for t, entries in models.items()}


async def test_offline_writes_a_manifest_for_every_shipped_model(offline: ModelCatalogUpdater):
    """The flag turns off the network, not the catalog. Until 2026-11-19
    it skipped the whole sync, so a first launch with it set downloaded
    the weights and listed no models (issue #139)."""
    await _sync_offline(offline)
    for model_type, ids in _shipped_ids().items():
        for model_id in ids:
            assert (offline.models_dir / model_type / model_id / "manifest.json").is_file()


async def test_offline_gives_a_weights_only_folder_its_manifest(offline: ModelCatalogUpdater):
    """Simon's screenshot: the weights on disk, no manifest.json next to
    them, so the model did not exist for the app."""
    model_dir = offline.models_dir / "det" / "MD5A-0-0"
    model_dir.mkdir(parents=True)
    (model_dir / "md_v5a.0.0.pt").write_bytes(b"w")

    await _sync_offline(offline)

    assert ManifestManager(offline.models_dir).get_model("MD5A-0-0").model_id == "MD5A-0-0"


async def test_offline_replaces_a_manifest_the_shipped_catalog_disagrees_with(
    offline: ModelCatalogUpdater,
):
    """A models folder copied from an older version keeps working under a
    newer app: the same rule as online, the catalog wins."""
    shipped = next(
        e for e in json.loads(_REPO_CATALOG.read_text())["models"]["det"]
        if e["model_id"] == "MD5A-0-0"
    )
    model_dir = offline.models_dir / "det" / "MD5A-0-0"
    model_dir.mkdir(parents=True)
    old = {k: v for k, v in shipped.items() if k != "domain"}
    (model_dir / "manifest.json").write_text(json.dumps(old))

    await _sync_offline(offline)

    assert json.loads((model_dir / "manifest.json").read_text()) == shipped


async def test_offline_never_fetches_a_taxonomy(offline: ModelCatalogUpdater):
    """A classifier stub has no taxonomy.csv; online the sync fetches it
    from HuggingFace. Offline it stays missing until the model itself is
    downloaded, which brings the file along."""
    with patch.object(offline, "download_taxonomy") as mock_taxonomy:
        await _sync_offline(offline)
    mock_taxonomy.assert_not_called()


def test_no_bundled_file_means_no_catalog(updater: ModelCatalogUpdater):
    """Nothing is invented when the shipped file is missing."""
    with (
        patch(
            "app.ml.catalog_updater.urllib.request.urlopen",
            side_effect=OSError("blocked"),
        ),
        patch("app.ml.catalog_updater._bundled_catalog_path", return_value=None),
    ):
        assert updater.fetch_catalog() is None
