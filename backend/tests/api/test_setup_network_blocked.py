"""The wizard's side of a network that blocks the model downloads.

The failure happens inside a background thread, so the only way the user
learns about it is the polled status endpoint. `error_kind` is what lets
the wizard point at the right section of the help page instead of a bare
"Download failed", so these tests pin that it appears for exactly that
failure and for no other.
"""

import pytest

from app.api.routers import setup as setup_router
from app.ml.hf_downloader import NetworkBlockedError


@pytest.fixture(autouse=True)
def clean_install_state():
    """Module-level state leaks between tests otherwise."""
    setup_router._install_state.finish(None)
    yield
    setup_router._install_state.finish(None)


def test_status_reports_the_kind_for_a_blocked_network(client, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise NetworkBlockedError("huggingface.co")

    monkeypatch.setattr(setup_router, "run_setup", boom)
    setup_router._install_env_blocking()

    body = client.get("/api/setup/status").json()

    assert body["error_kind"] == "network_blocked"
    # The message names the host, so the user can hand it to IT as is.
    assert "huggingface.co" in body["error"]
    assert "blocks" in body["error"]


def test_a_bare_download_failure_carries_no_kind(client, monkeypatch) -> None:
    """A repo that really failed to download (timeout, disk) is not a block."""

    def boom(*args, **kwargs):
        raise RuntimeError("Failed to download MD5A-0-0: Download failed")

    monkeypatch.setattr(setup_router, "run_setup", boom)
    setup_router._install_env_blocking()

    body = client.get("/api/setup/status").json()

    assert body["error_kind"] is None
    assert "Download failed" in body["error"]
