"""The once-per-install licence step for models that run on ultralytics."""

import pytest

from app.ml import agpl_acceptance as agpl
from app.ml.environment_manager import shipped_env_names
from app.ml.schemas.model_manifest import ModelManifest


@pytest.fixture(autouse=True)
def own_data_dir(tmp_path, monkeypatch):
    """Never the real ~/AddaxAI."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    return tmp_path


def _manifest(env: str) -> ModelManifest:
    return ModelManifest(
        model_id="M-v1", friendly_name="M", env=env, model_fname="m.pt",
        description="", developer="d", info_url="", min_app_version="7.0.0",
    )


def test_nothing_is_accepted_on_a_fresh_install():
    assert not agpl.agpl_accepted()
    assert agpl.needs_agpl_acceptance(_manifest("ultralytics"))


def test_accepting_writes_a_marker_that_explains_itself(own_data_dir):
    marker = agpl.accept_agpl()
    assert marker == own_data_dir / ".agpl-accepted"
    text = marker.read_text()
    assert "AGPL-3.0" in text and agpl.AGPL_LICENCE_URL in text
    assert "Delete this file" in text
    # Idempotent.
    assert agpl.accept_agpl() == marker
    assert not agpl.needs_agpl_acceptance(_manifest("ultralytics"))


def test_only_a_model_in_the_ultralytics_env_is_asked():
    for env in ("addaxai-base", "marine", "pytorch"):
        assert not agpl.needs_agpl_acceptance(_manifest(env))


def test_the_gate_key_is_a_shipped_env():
    """The key is the env folder name; renaming one without the other
    would ask nobody, or ask about an env that does not exist."""
    assert agpl.AGPL_ENV in shipped_env_names()
