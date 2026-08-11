"""
Tests for the scripted setup CLI (app/setup_cli.py) and the run_setup
refactor it leans on. Everything network- or disk-heavy is stubbed:
these tests pin the dispatch, skip-fast, and failure wiring, not the
downloads themselves.
"""

from types import SimpleNamespace

import pytest

from app import setup_cli
from app.api.routers import setup as setup_router


@pytest.fixture(autouse=True)
def isolated_user_data_dir(monkeypatch, tmp_path):
    """Point the data dir at a throwaway dir so no test touches ~/AddaxAI."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    return tmp_path


def test_run_cli_skips_base_setup_when_complete(monkeypatch, capsys):
    monkeypatch.setattr(setup_router, "setup_complete", lambda: True)
    monkeypatch.setattr(
        setup_router,
        "run_setup",
        lambda *a, **k: pytest.fail("run_setup must not run"),
    )

    assert setup_cli.run_cli(["--setup"]) == 0
    out = capsys.readouterr().out
    assert "already present" in out
    assert "Setup complete" in out


def test_run_cli_unknown_model_fails_before_downloading(monkeypatch, capsys):
    monkeypatch.setattr(setup_router, "setup_complete", lambda: True)
    monkeypatch.setattr(setup_cli, "_sync_catalog", lambda: None)

    from app.ml.manifest_manager import ManifestManager

    monkeypatch.setattr(
        ManifestManager,
        "load_manifests",
        lambda self, force_refresh=False: {},
    )

    assert setup_cli.run_cli(["--setup", "NOPE-v1"]) == 1
    assert "Unknown model" in capsys.readouterr().err


def test_run_cli_treats_non_flag_args_as_model_ids(monkeypatch):
    monkeypatch.setattr(setup_router, "setup_complete", lambda: True)
    monkeypatch.setattr(setup_cli, "_sync_catalog", lambda: None)
    monkeypatch.setattr(setup_cli, "_resolve_models", lambda ids: ids)

    seen: list[list[str]] = []
    monkeypatch.setattr(setup_cli, "_install_models", seen.append)

    assert setup_cli.run_cli(["--setup", "A-v1", "B-v2"]) == 0
    assert seen == [["A-v1", "B-v2"]]


def test_list_models_prints_catalog(monkeypatch, capsys):
    monkeypatch.setattr(setup_cli, "_sync_catalog", lambda: None)

    from app.ml.manifest_manager import ManifestManager

    fake = SimpleNamespace(
        model_id="X-v1",
        model_category="classification",
        friendly_name="Model X",
    )
    monkeypatch.setattr(
        ManifestManager,
        "load_manifests",
        lambda self, force_refresh=False: {"X-v1": fake},
    )

    assert setup_cli.run_cli(["--list-models"]) == 0
    out = capsys.readouterr().out
    assert "X-v1" in out
    assert "Model X" in out


def test_list_models_empty_catalog_fails(monkeypatch, capsys):
    monkeypatch.setattr(setup_cli, "_sync_catalog", lambda: None)

    from app.ml.manifest_manager import ManifestManager

    monkeypatch.setattr(
        ManifestManager,
        "load_manifests",
        lambda self, force_refresh=False: {},
    )

    assert setup_cli.run_cli(["--list-models"]) == 1
    assert "No models found" in capsys.readouterr().err


def test_run_setup_noop_when_everything_present(monkeypatch, tmp_path):
    for spec in setup_router._DEFAULT_MODELS:
        weight = (
            tmp_path / "models" / spec["type_dir"] / spec["model_id"] / spec["model_fname"]
        )
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"x")
    monkeypatch.setattr(setup_router, "_env_present", lambda: True)
    monkeypatch.setattr(
        setup_router,
        "_get_env_manager",
        lambda: SimpleNamespace(envs_dir=tmp_path / "envs"),
    )

    calls: list[tuple[str, float]] = []
    setup_router.run_setup(lambda msg, prog: calls.append((msg, prog)))
    assert calls == []


def test_install_env_blocking_success_passes_force_envs(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        setup_router,
        "run_setup",
        lambda cb, force_envs=(): calls.append(force_envs),
    )

    setup_router._install_env_blocking(("pytorch",))
    assert calls == [("pytorch",)]
    assert setup_router._install_state.error is None
    assert setup_router._install_state.in_progress is False


def test_install_env_blocking_records_error(monkeypatch):
    def boom(cb, force_envs=()):
        raise RuntimeError("boom")

    monkeypatch.setattr(setup_router, "run_setup", boom)

    setup_router._install_env_blocking()
    assert setup_router._install_state.error == "boom"
    assert setup_router._install_state.in_progress is False
