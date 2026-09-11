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


def test_every_detector_declares_its_classes() -> None:
    """A detector's classes are the only way the app can say what it finds
    before it has ever been run, which is what the label picker needs: it
    used to offer a hardcoded Animal / Person / Vehicle, so on a SharkTrack
    project the only category you could apply was "animal", overwriting the
    box's real "elasmobranch".

    Lookup and display only. Nothing in the detection or ingest path reads
    it: a run's own JSON carries the authoritative `detection_categories`,
    and `json_pipeline` refuses an id that map never declared. So a stale
    entry here shows a wrong option in a picker, it can never mislabel a
    stored box.
    """
    catalog = json.loads(_CATALOG_PATH.read_text())
    for entry in catalog["models"]["det"]:
        classes = entry.get("classes")
        assert classes, f"{entry['model_id']} declares no classes"
        assert all(isinstance(c, str) and c for c in classes), entry["model_id"]
        assert classes == [c.lower() for c in classes], (
            f"{entry['model_id']}: classes are matched against "
            f"Detection.category, which is stored lowercase"
        )


def test_the_declared_classes_match_what_the_detectors_emit() -> None:
    """Read off the weights themselves (SharkTrack's checkpoint says
    ``{0: 'elasmobranch'}``, the CFD checkpoints say ``{'0': 'fish'}``) and
    off the megadetector package's own ``DEFAULT_DETECTOR_LABEL_MAP`` for
    the MegaDetectors. Pinned here so a new entry cannot be invented."""
    catalog = json.loads(_CATALOG_PATH.read_text())
    by_id = {e["model_id"]: e["classes"] for e in catalog["models"]["det"]}
    megadetector = ["animal", "person", "vehicle"]
    for model_id, classes in by_id.items():
        if model_id.startswith(("MD5", "MD1000")):
            assert classes == megadetector, model_id
        elif model_id.startswith("CFD-"):
            assert classes == ["fish"], model_id
        elif model_id.startswith("SHARKTRACK"):
            assert classes == ["elasmobranch"], model_id
        else:
            pytest.fail(f"{model_id} has no pinned expectation; add one")
