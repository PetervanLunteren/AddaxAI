"""The wizard's side of the certificate-revocation opt-out.

The failure happens inside a background thread, so the only way the user
ever learns about it is the polled status endpoint. `error_kind` is what
tells the wizard it may offer skipping the revocation check, so these
tests pin that it appears for exactly that failure and for no other.
"""

import pytest

from app.api.routers import setup as setup_router
from app.ml.environment_manager import (
    TlsRevocationCheckError,
    revocation_skip_allowed,
)


@pytest.fixture(autouse=True)
def clean_install_state():
    """Module-level state leaks between tests otherwise."""
    setup_router._install_state.finish(None)
    yield
    setup_router._install_state.finish(None)


@pytest.fixture
def own_data_dir(tmp_path, monkeypatch):
    """Point the marker at a throwaway dir, never the real ~/AddaxAI."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    return tmp_path


def test_status_reports_the_kind_for_a_revocation_failure(
    client, monkeypatch
) -> None:
    def boom(*args, **kwargs):
        raise TlsRevocationCheckError("Windows could not check ... revoked.")

    monkeypatch.setattr(setup_router, "run_setup", boom)
    setup_router._install_env_blocking()

    body = client.get("/api/setup/status").json()

    assert body["error_kind"] == "tls_revocation"
    assert "revoked" in body["error"]


def test_status_reports_no_kind_for_an_ordinary_failure(
    client, monkeypatch
) -> None:
    """A generic build failure must not offer to weaken TLS."""

    def boom(*args, **kwargs):
        raise RuntimeError("micromamba create failed (exit 1):\ndisk full")

    monkeypatch.setattr(setup_router, "run_setup", boom)
    setup_router._install_env_blocking()

    body = client.get("/api/setup/status").json()

    assert body["error_kind"] is None
    assert "disk full" in body["error"]


def test_status_carries_no_kind_when_nothing_failed(client) -> None:
    body = client.get("/api/setup/status").json()

    assert body["error_kind"] is None


def test_a_new_attempt_clears_the_previous_kind(client, monkeypatch) -> None:
    """Otherwise the button would linger over an unrelated later failure."""

    def boom(*args, **kwargs):
        raise TlsRevocationCheckError("Windows could not check ... revoked.")

    monkeypatch.setattr(setup_router, "run_setup", boom)
    setup_router._install_env_blocking()
    assert client.get("/api/setup/status").json()["error_kind"] is not None

    setup_router._install_state.start()

    assert client.get("/api/setup/status").json()["error_kind"] is None


def test_allow_endpoint_writes_the_marker(client, own_data_dir) -> None:
    assert revocation_skip_allowed() is False

    response = client.post("/api/setup/allow-no-revocation-check")

    assert response.status_code == 200
    assert revocation_skip_allowed() is True
    marker = own_data_dir / ".allow-no-revocation-check"
    assert marker.is_file()
    assert response.json()["marker_path"] == str(marker)


def test_allow_endpoint_is_idempotent(client, own_data_dir) -> None:
    """The user can click it twice, or click it on a machine where a
    previous attempt already set it."""
    first = client.post("/api/setup/allow-no-revocation-check")
    second = client.post("/api/setup/allow-no-revocation-check")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert revocation_skip_allowed() is True
