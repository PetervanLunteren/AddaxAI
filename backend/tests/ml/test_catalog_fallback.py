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

from app.ml.catalog_updater import ModelCatalogUpdater, _bundled_catalog_path

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


def test_a_working_host_still_wins(updater: ModelCatalogUpdater):
    """The fallback is a fallback, not a cache that shadows upstream."""
    remote = {"models": {"det": [], "cls": [], "emb": []}}
    with patch("app.ml.catalog_updater.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            json.dumps(remote).encode()
        )
        assert updater.fetch_catalog() == remote


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
