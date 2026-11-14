"""The wizard's side of a setup failure on a PC with Smart App Control on.

Smart App Control blocks unsigned programs that a package runs while it
installs, and the build then dies with a micromamba error that names no
cause (gdk-pixbuf's post-link step, beta report 2026-11-12). `error_kind`
lets the wizard point at the Smart App Control help section instead. These
tests pin that it is tagged only for a failure without a known cause, and
only when Windows reports Smart App Control as on.
"""

import pytest

from app.api.routers import setup as setup_router
from app.ml.hf_downloader import NetworkBlockedError

BUILD_ERROR = (
    "Failed to create environment env-addaxai-base: micromamba create failed "
    "(exit 1):\nfailed to execute pre/post link script for gdk-pixbuf"
)


@pytest.fixture(autouse=True)
def clean_install_state(monkeypatch):
    """Module-level state leaks between tests otherwise. The env check is
    stubbed so the status route never downloads micromamba."""
    monkeypatch.setattr(setup_router, "_env_present", lambda name=None: False)
    setup_router._install_state.finish(None)
    yield
    setup_router._install_state.finish(None)


def _fail_setup(monkeypatch, error: Exception) -> None:
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(setup_router, "run_setup", boom)
    setup_router._install_env_blocking()


def test_an_unexplained_failure_names_smart_app_control_when_it_is_on(
    client, monkeypatch
) -> None:
    monkeypatch.setattr(setup_router, "smart_app_control_state", lambda: "on")
    _fail_setup(monkeypatch, RuntimeError(BUILD_ERROR))

    body = client.get("/api/setup/status").json()

    assert body["error_kind"] == "smart_app_control"
    # The help line carries the hint; the message itself is unchanged.
    assert body["error"] == BUILD_ERROR


@pytest.mark.parametrize("state", ["off", "evaluation", None])
def test_no_kind_when_smart_app_control_is_not_enforced(
    client, monkeypatch, state
) -> None:
    """Evaluation mode only observes, it never blocks. None is every
    machine that is not Windows, or a registry that cannot be read."""
    monkeypatch.setattr(setup_router, "smart_app_control_state", lambda: state)
    _fail_setup(monkeypatch, RuntimeError(BUILD_ERROR))

    body = client.get("/api/setup/status").json()

    assert body["error_kind"] is None
    assert body["error"] == BUILD_ERROR


def test_a_failure_with_a_known_cause_keeps_its_own_kind(client, monkeypatch) -> None:
    monkeypatch.setattr(setup_router, "smart_app_control_state", lambda: "on")
    _fail_setup(monkeypatch, NetworkBlockedError("huggingface.co"))

    body = client.get("/api/setup/status").json()

    assert body["error_kind"] == "network_blocked"
