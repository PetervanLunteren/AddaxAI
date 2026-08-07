"""Verify the CORS allow-list follows the configured API port.

The Electron app picks the backend port (ADDAXAI_BACKEND_PORT) and passes
it to the backend as API_PORT, so the renderer is served from whatever
port the backend bound. CORS origins pinned to a literal 8000 stopped
matching as soon as the port moved, which is exactly the case this
setting exists to support.

Preflight is answered by CORSMiddleware itself, so these never reach a
route and need no database.
"""

import pytest
from fastapi.testclient import TestClient


def _preflight(origin: str) -> str | None:
    """Send a CORS preflight from `origin`, return the allow-origin header."""
    from app.main import create_app

    client = TestClient(create_app())
    response = client.options(
        "/api/logs",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    return response.headers.get("access-control-allow-origin")


def test_configured_port_is_an_allowed_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A renderer served from the configured port is allowed."""
    monkeypatch.setenv("API_PORT", "8123")
    assert _preflight("http://localhost:8123") == "http://localhost:8123"
    assert _preflight("http://127.0.0.1:8123") == "http://127.0.0.1:8123"


def test_default_port_is_not_allowed_once_the_port_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old default is not left in the allow-list.

    Guards the regression directly: if 8000 were still hardcoded here it
    would pass this origin regardless of the setting.
    """
    monkeypatch.setenv("API_PORT", "8123")
    assert _preflight("http://localhost:8000") is None


def test_vite_dev_server_is_always_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving the port must not lock out the dev server on 5173."""
    monkeypatch.setenv("API_PORT", "8123")
    assert _preflight("http://localhost:5173") == "http://localhost:5173"
