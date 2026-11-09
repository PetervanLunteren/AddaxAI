"""
Tests for the scripted setup CLI (app/setup_cli.py) and the run_setup
refactor it leans on. Everything network- or disk-heavy is stubbed:
these tests pin the dispatch, skip-fast, and failure wiring, not the
downloads themselves.
"""

import shutil
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


# Replacing the base env. A rebuild once deleted the env in place, stopped
# at a locked file and reported success over a half-deleted env without
# exiftool (Windows beta report, 2026-11-04). These pin that run_setup
# replaces through retire_env, never swallows its failure, and treats a
# base env without exiftool as one to replace.


def _models_installed(tmp_path):
    for spec in setup_router._DEFAULT_MODELS:
        weight = (
            tmp_path / "models" / spec["type_dir"] / spec["model_id"] / spec["model_fname"]
        )
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"x")


class _StubEnvManager:
    """Records what run_setup retires and builds; `refuse` makes the
    retire fail the way a locked env does on Windows."""

    def __init__(self, envs_dir, refuse=False):
        self.envs_dir = envs_dir
        self.refuse = refuse
        self.retired = []
        self.built = []

    def retire_env(self, path):
        if self.refuse:
            raise RuntimeError("Windows refused to move the old environment aside")
        self.retired.append(path)
        shutil.rmtree(path)

    def get_or_create_env(self, manifest, cb):
        self.built.append(manifest.env)


def _base_env(tmp_path):
    env = tmp_path / "envs" / "env-addaxai-base"
    env.mkdir(parents=True)
    return env


def test_run_setup_force_rebuild_raises_when_the_env_cannot_be_moved(
    monkeypatch, tmp_path
):
    """The failure that used to be logged and ignored now ends the
    install, so the modal shows it instead of 'Environment updated'."""
    _models_installed(tmp_path)
    env = _base_env(tmp_path)
    em = _StubEnvManager(tmp_path / "envs", refuse=True)
    monkeypatch.setattr(setup_router, "_get_env_manager", lambda: em)
    monkeypatch.setattr(setup_router, "_env_present", lambda: True)

    setup_router._install_env_blocking(("addaxai-base",))

    assert "move the old environment aside" in setup_router._install_state.error
    assert em.built == []
    assert env.is_dir()


def test_run_setup_force_rebuild_retires_then_builds(monkeypatch, tmp_path):
    _models_installed(tmp_path)
    env = _base_env(tmp_path)
    em = _StubEnvManager(tmp_path / "envs")
    monkeypatch.setattr(setup_router, "_get_env_manager", lambda: em)
    # Present until it is moved away, like the real check.
    monkeypatch.setattr(setup_router, "_env_present", lambda: env.exists())

    setup_router.run_setup(lambda msg, prog: None, ("addaxai-base",))

    assert em.retired == [env]
    assert em.built == ["addaxai-base"]


def test_run_setup_replaces_an_incomplete_base_env(monkeypatch, tmp_path):
    """No force: a base env that fails the presence check (it boots but
    lost exiftool) is moved away and rebuilt, not reused."""
    _models_installed(tmp_path)
    env = _base_env(tmp_path)
    em = _StubEnvManager(tmp_path / "envs")
    monkeypatch.setattr(setup_router, "_get_env_manager", lambda: em)
    monkeypatch.setattr(setup_router, "_env_present", lambda: False)

    setup_router.run_setup(lambda msg, prog: None)

    assert em.retired == [env]
    assert em.built == ["addaxai-base"]


def test_run_setup_leaves_a_healthy_base_env_alone(monkeypatch, tmp_path):
    _models_installed(tmp_path)
    _base_env(tmp_path)
    em = _StubEnvManager(tmp_path / "envs")
    monkeypatch.setattr(setup_router, "_get_env_manager", lambda: em)
    monkeypatch.setattr(setup_router, "_env_present", lambda: True)

    setup_router.run_setup(lambda msg, prog: None)

    assert em.retired == []
    assert em.built == []


@pytest.mark.parametrize("has_exiftool", [True, False])
def test_env_present_requires_exiftool(monkeypatch, tmp_path, has_exiftool):
    """The boot probe cannot see exiftool, so a base env that boots but
    has no exiftool must still read as not installed."""
    from app.utils.exiftool_bin import bundled_exiftool_path

    env = _base_env(tmp_path)
    if has_exiftool:
        exiftool = bundled_exiftool_path(env)
        exiftool.parent.mkdir(parents=True)
        exiftool.write_text("")
    em = SimpleNamespace(envs_dir=tmp_path / "envs", _validate_env=lambda p: True)
    monkeypatch.setattr(setup_router, "_get_env_manager", lambda: em)

    assert setup_router._env_present() is has_exiftool
