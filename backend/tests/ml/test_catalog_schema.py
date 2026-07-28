"""The shipped catalog must validate against the shipped schema.

The model catalog (repo root ``models.json``) is fetched live at runtime
and written to every user's disk, then validated against whatever
``ModelManifest`` schema their build carries. If a field is required in the
schema but absent from the catalog, the manifests fail to validate. Since
``load_manifests`` now skips invalid manifests, the failure is quiet: the
affected models just vanish from the list instead of crashing. Either way
the model is unusable.

This test pins the invariant that the current schema can read every entry
in the current catalog, so a required-field change that the catalog does
not supply (the beta-tester "MD5A not found" report) fails in CI, not on a
user's machine.
"""

import json
from pathlib import Path

import pytest

from app.ml.schemas.model_manifest import ModelManifest

_CATALOG_PATH = Path(__file__).resolve().parents[3] / "models.json"


def _catalog_entries() -> list[tuple[str, dict]]:
    catalog = json.loads(_CATALOG_PATH.read_text())
    entries: list[tuple[str, dict]] = []
    for category, models in catalog["models"].items():
        for entry in models:
            entries.append((f"{category}/{entry['model_id']}", entry))
    return entries


def test_catalog_file_exists():
    assert _CATALOG_PATH.is_file(), f"catalog not found at {_CATALOG_PATH}"


@pytest.mark.parametrize(
    "model_key,entry",
    _catalog_entries(),
    ids=[key for key, _ in _catalog_entries()],
)
def test_catalog_entry_validates_against_schema(model_key: str, entry: dict):
    # Same call load_manifests makes on the user's synced copy. If this
    # raises, the current schema cannot read the current catalog and the
    # model would silently drop out on every build shipping this schema.
    ModelManifest(**entry)
