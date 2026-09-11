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


def _declared_classes(entry: dict) -> list[str]:
    """What a detector says it finds, from whichever field carries it."""
    mapping = entry.get("class_mapping")
    return list(mapping.values()) if mapping else (entry.get("classes") or [])


def test_every_detector_declares_its_classes_in_exactly_one_field() -> None:
    """A detector's classes are the only way the app can say what it finds
    before it has ever been run, which is what the label picker needs: it
    used to offer a hardcoded Animal / Person / Vehicle, so on a SharkTrack
    project the only category you could apply was "animal", overwriting the
    box's real "elasmobranch".

    Two fields, and never both, so there is one place to read and nothing
    to drift. `classes` is lookup and display only, for a model the
    megadetector package names itself. `class_mapping` is for one it
    cannot: it is handed to the package as `--class_mapping_filename`, so
    it also decides the ids the run reports.
    """
    catalog = json.loads(_CATALOG_PATH.read_text())
    for entry in catalog["models"]["det"]:
        model_id = entry["model_id"]
        assert "detector_runtime" not in entry, (
            f"{model_id}: detector_runtime is retired, there is one loader"
        )
        has_classes = "classes" in entry
        has_mapping = "class_mapping" in entry
        assert has_classes != has_mapping, (
            f"{model_id} must declare exactly one of classes / class_mapping"
        )
        names = _declared_classes(entry)
        assert names, f"{model_id} declares no classes"
        assert all(isinstance(c, str) and c for c in names), model_id
        assert names == [c.lower() for c in names], (
            f"{model_id}: classes are matched against "
            f"Detection.category, which is stored lowercase"
        )
        if has_mapping:
            assert all(k.isdigit() for k in entry["class_mapping"]), (
                f"{model_id}: class ids are the model's own, starting at zero"
            )


def test_the_declared_classes_match_what_the_detectors_emit() -> None:
    """Read off the weights themselves (SharkTrack's checkpoint says
    ``{0: 'elasmobranch'}``, the CFD checkpoints say ``{'0': 'fish'}``) and
    off the megadetector package's own ``DEFAULT_DETECTOR_LABEL_MAP`` for
    the MegaDetectors. Pinned here so a new entry cannot be invented."""
    catalog = json.loads(_CATALOG_PATH.read_text())
    by_id = {e["model_id"]: e for e in catalog["models"]["det"]}
    megadetector = ["animal", "person", "vehicle"]
    for model_id, entry in by_id.items():
        classes = _declared_classes(entry)
        if model_id.startswith(("MD5", "MD1000")):
            assert classes == megadetector, model_id
        elif model_id.startswith("CFD-"):
            assert classes == ["fish"], model_id
        elif model_id.startswith("SHARKTRACK"):
            # The one model the package cannot name on its own, so its ids
            # matter as well as its names: index 0 is what the checkpoint
            # emits once the package is on native classes.
            assert entry["class_mapping"] == {"0": "elasmobranch"}, model_id
        else:
            pytest.fail(f"{model_id} has no pinned expectation; add one")
