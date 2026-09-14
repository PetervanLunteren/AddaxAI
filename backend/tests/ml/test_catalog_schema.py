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
import re
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
    """One rule: every detector says what it finds, in `classes`.

    That list is the only way the app can name a detector's classes before
    it has ever been run, which is what the label picker needs: it used to
    offer a hardcoded Animal / Person / Vehicle, so on a SharkTrack project
    the only category you could apply was "animal", overwriting the box's
    real "elasmobranch". Enforced here rather than by the schema, which is
    shared with classifiers and embedders that have no classes.
    """
    catalog = json.loads(_CATALOG_PATH.read_text())
    for entry in catalog["models"]["det"]:
        model_id = entry["model_id"]
        assert "detector_runtime" not in entry, (
            f"{model_id}: detector_runtime is retired, there is one loader"
        )
        names = entry.get("classes")
        assert names, f"{model_id} declares no classes"
        assert all(isinstance(c, str) and c for c in names), model_id
        assert names == [c.lower() for c in names], (
            f"{model_id}: classes are matched against "
            f"Detection.category, which is stored lowercase"
        )


def test_every_detector_declares_its_domain() -> None:
    """Every detector says what footage it is for, `camera_trap` or
    `underwater`, and the value matches the model: the track post-filter
    takes its two numbers from it (`app/ml/track_filter.py`). Enforced
    here like `classes`, since the schema is shared with classifiers."""
    catalog = json.loads(_CATALOG_PATH.read_text())
    for entry in catalog["models"]["det"]:
        model_id = entry["model_id"]
        assert "track_filter" not in entry, f"{model_id}: track_filter is retired, domain decides"
        expected = "underwater" if model_id.startswith(("SHARKTRACK", "CFD-")) else "camera_trap"
        assert entry.get("domain") == expected, f"{model_id}: domain {entry.get('domain')!r}"


def test_a_class_mapping_agrees_with_the_classes_beside_it() -> None:
    """`class_mapping` is the second half of the same fact, so the two
    cannot be allowed to drift: `classes` is what the picker offers, the
    mapping is what the run actually reports as its categories, and a
    disagreement would show the user a class their boxes never carry.

    Only a model the megadetector package cannot name needs one, so most
    detectors have no mapping at all and nothing to check.
    """
    catalog = json.loads(_CATALOG_PATH.read_text())
    checked = 0
    for entry in catalog["models"]["det"]:
        mapping = entry.get("class_mapping")
        if not mapping:
            continue
        checked += 1
        model_id = entry["model_id"]
        assert all(k.isdigit() for k in mapping), (
            f"{model_id}: class ids are the model's own, starting at zero"
        )
        in_id_order = [mapping[k] for k in sorted(mapping, key=int)]
        assert entry["classes"] == in_id_order, (
            f"{model_id}: classes {entry['classes']} disagree with "
            f"class_mapping {in_id_order}"
        )
    assert checked, "no detector declares a class_mapping; did the field move?"


def test_the_declared_classes_match_what_the_detectors_emit() -> None:
    """Read off the weights themselves (SharkTrack's checkpoint says
    ``{0: 'elasmobranch'}``, the CFD checkpoints say ``{'0': 'fish'}``) and
    off the megadetector package's own ``DEFAULT_DETECTOR_LABEL_MAP`` for
    the MegaDetectors. Pinned here so a new entry cannot be invented."""
    catalog = json.loads(_CATALOG_PATH.read_text())
    by_id = {e["model_id"]: e for e in catalog["models"]["det"]}
    megadetector = ["animal", "person", "vehicle"]
    for model_id, entry in by_id.items():
        classes = entry["classes"]
        if model_id.startswith(("MD5", "MD1000")):
            assert classes == megadetector, model_id
        elif model_id.startswith("CFD-"):
            assert classes == ["fish"], model_id
        elif model_id.startswith("SHARKTRACK"):
            assert classes == ["elasmobranch"], model_id
            # The one model the package cannot name on its own, so its ids
            # matter as well as its names: index 0 is what the checkpoint
            # emits once the package is on native classes.
            assert entry["class_mapping"] == {"0": "elasmobranch"}, model_id
        else:
            pytest.fail(f"{model_id} has no pinned expectation; add one")


_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_RELEASE_DATE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def test_release_dates_are_months_and_match_the_caption() -> None:
    """`release_date` is "YYYY-MM" by decision (month precision for every
    model, a day is not known for all). The same month is typed by hand
    at the end of `description_short` so the dropdown shows it at a
    glance; the two are one fact in two places, so this pins that they
    agree. A model nobody could date has neither."""
    catalog = json.loads(_CATALOG_PATH.read_text())
    dated = 0
    for models in catalog["models"].values():
        for entry in models:
            model_id = entry["model_id"]
            date = entry.get("release_date")
            if date is None:
                continue
            dated += 1
            assert _RELEASE_DATE.match(date), f"{model_id}: release_date {date!r} is not YYYY-MM"
            year, month = date.split("-")
            words = f"{_MONTHS[int(month) - 1]} {year}"
            caption = entry.get("description_short") or ""
            assert caption.endswith(f" • {words}"), (
                f"{model_id}: description_short {caption!r} does not end with "
                f"the release month {words!r}"
            )
    assert dated >= 40, f"only {dated} models carry a release_date; did the field move?"
