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
    catalog = _fetch(updater, remote)
    assert all(e.get("domain") for e in catalog["models"]["det"])
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
