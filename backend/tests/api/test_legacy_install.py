"""Endpoints that report and remove a legacy AddaxAI install.

`scan` is stubbed in every test: the real one reads absolute machine
paths, so a developer who genuinely has legacy AddaxAI installed would
otherwise see these fail. The path logic itself is covered by
tests/services/test_legacy_install.py.
"""

from pathlib import Path

import pytest

from app.api.routers import setup as setup_router
from app.services import legacy_install
from app.services.legacy_install import LegacyScan


@pytest.fixture(autouse=True)
def clean_purge_state():
    """Module-level state leaks between tests otherwise."""
    setup_router._purge_state.finish(None)
    yield
    setup_router._purge_state.finish(None)


@pytest.fixture
def no_legacy(monkeypatch):
    monkeypatch.setattr(legacy_install, "scan", lambda: LegacyScan())


@pytest.fixture
def with_legacy(monkeypatch):
    found = LegacyScan(
        root=Path("/Applications/AddaxAI_files"),
        version="6.37",
        manual=(Path("C:/Program Files/AddaxAI_files"),),
    )
    monkeypatch.setattr(legacy_install, "scan", lambda: found)
    return found


def test_reports_nothing_when_no_legacy_install(client, no_legacy):
    data = client.get("/api/setup/legacy-install").json()
    assert data["found"] is False
    assert data["version"] is None
    assert data["removable_paths"] == []
    assert data["manual_paths"] == []
    assert data["removal_in_progress"] is False
    assert data["removal_error"] is None


def test_reports_paths_and_version_when_found(client, with_legacy):
    data = client.get("/api/setup/legacy-install").json()
    assert data["found"] is True
    assert data["version"] == "6.37"
    assert data["removable_paths"] == ["/Applications/AddaxAI_files"]
    assert data["manual_paths"] == ["C:/Program Files/AddaxAI_files"]


def test_remove_returns_202(client, with_legacy, monkeypatch):
    monkeypatch.setattr(legacy_install, "remove", lambda: [])
    assert client.post("/api/setup/legacy-install/remove").status_code == 202


def test_second_remove_while_running_returns_409(client, with_legacy):
    assert setup_router._purge_state.start() is True
    assert client.post("/api/setup/legacy-install/remove").status_code == 409


def test_survivors_surface_as_a_retry_message(client, with_legacy, monkeypatch):
    monkeypatch.setattr(
        legacy_install, "remove", lambda: [Path("/Applications/AddaxAI_files")]
    )
    setup_router._purge_state.start()
    setup_router._remove_legacy_blocking()

    data = client.get("/api/setup/legacy-install").json()
    assert data["removal_in_progress"] is False
    assert "Close the old AddaxAI" in data["removal_error"]


def test_clean_removal_reports_no_error(client, with_legacy, monkeypatch):
    monkeypatch.setattr(legacy_install, "remove", lambda: [])
    setup_router._purge_state.start()
    setup_router._remove_legacy_blocking()

    data = client.get("/api/setup/legacy-install").json()
    assert data["removal_in_progress"] is False
    assert data["removal_error"] is None


def test_unexpected_failure_is_caught_and_reported(client, with_legacy, monkeypatch):
    def _boom():
        raise OSError("drive went away")

    monkeypatch.setattr(legacy_install, "remove", _boom)
    setup_router._purge_state.start()
    setup_router._remove_legacy_blocking()

    data = client.get("/api/setup/legacy-install").json()
    assert data["removal_in_progress"] is False
    assert "drive went away" in data["removal_error"]
