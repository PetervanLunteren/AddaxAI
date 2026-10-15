"""The models first-run setup installs must be the models the app offers.

Three places name the same two detectors, and they have to agree:

* ``routers/setup.py`` lists them inline, so setup works with no catalog
  (that is deliberate, see the comment there),
* ``models.json`` is what the app actually loads them from,
* ``frontend/src/lib/data-type.ts`` preselects them in the setup forms.

Drift is silent and lands on a brand new user: setup downloads one model
while the form starts on another, so the first thing they see is a
"not installed" badge on a fresh install. Nothing else checks this,
because the three live in two languages and a JSON file, so this test
does, by reading all three.
"""

import json
import re
from pathlib import Path

import pytest

from app.api.routers.setup import _DEFAULT_MODELS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CATALOG_PATH = _REPO_ROOT / "models.json"
_FRONTEND_DATA_TYPE = _REPO_ROOT / "frontend" / "src" / "lib" / "data-type.ts"

_SECTION = {"detection": "det", "embedding": "emb"}


def _catalog_entry(category: str, model_id: str) -> dict | None:
    catalog = json.loads(_CATALOG_PATH.read_text())
    for entry in catalog["models"][_SECTION[category]]:
        if entry["model_id"] == model_id:
            return entry
    return None


def _specs() -> list[tuple[str, str, dict]]:
    return [
        (data_type, spec["model_id"], spec)
        for data_type, specs in _DEFAULT_MODELS.items()
        for spec in specs
    ]


@pytest.mark.parametrize(
    "data_type,model_id,spec",
    _specs(),
    ids=[f"{dt}-{mid}" for dt, mid, _ in _specs()],
)
def test_each_default_model_matches_its_catalog_entry(
    data_type: str, model_id: str, spec: dict
) -> None:
    entry = _catalog_entry(spec["category"], model_id)
    assert entry is not None, (
        f"setup installs {model_id} for {data_type}, but the catalog has no "
        f"such {spec['category']} model"
    )
    # The weights filename is what `_models_present` looks for on disk, so
    # a catalog rename would leave setup downloading a file it then calls
    # missing, forever.
    assert entry["model_fname"] == spec["model_fname"], model_id
    # The env decides which environment setup builds; `marine` is the
    # whole reason the underwater install differs from the camera trap one.
    assert entry["env"] == spec["env"], model_id


@pytest.mark.parametrize("data_type", sorted(_DEFAULT_MODELS))
def test_the_default_detector_is_of_that_domain(data_type: str) -> None:
    detectors = [s for s in _DEFAULT_MODELS[data_type] if s["category"] == "detection"]
    assert len(detectors) == 1, f"{data_type} should install exactly one detector"
    entry = _catalog_entry("detection", detectors[0]["model_id"])
    assert entry is not None
    assert entry["domain"] == data_type, (
        f"setup installs {entry['model_id']} for {data_type}, but the catalog "
        f"calls it {entry['domain']!r}; the setup forms filter on the catalog, "
        f"so the installed model would not even be offered"
    )


def _frontend_default_detectors() -> dict[str, str]:
    """The ids out of `DEFAULT_DETECTOR` in lib/data-type.ts.

    Reading another language's source is unusual, and it is the only way
    to pin this: the constant cannot be shared, because the backend list
    has to keep working with no catalog and no frontend.
    """
    block = re.search(
        r"DEFAULT_DETECTOR[^{]*\{(.*?)\}", _FRONTEND_DATA_TYPE.read_text(), re.S
    )
    assert block, f"DEFAULT_DETECTOR not found in {_FRONTEND_DATA_TYPE}"
    return dict(re.findall(r'(\w+):\s*"([^"]+)"', block.group(1)))


def test_the_forms_start_on_the_detector_setup_installed() -> None:
    frontend = _frontend_default_detectors()
    backend = {
        data_type: next(
            s["model_id"] for s in specs if s["category"] == "detection"
        )
        for data_type, specs in _DEFAULT_MODELS.items()
    }
    assert frontend == backend
