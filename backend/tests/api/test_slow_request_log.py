"""The slow-request log: one line per request at or over the threshold.

It is what makes a user's "this page is slow" report readable, so the
line must name the request, and must never carry a query string: some
of those hold absolute folder paths, and the log ends up in diagnostics
bundles.
"""

import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

# Imported here, at collection: importing app.main runs setup_logging,
# which replaces the root handlers, caplog's among them, if it happens
# inside a test.
from app.main import create_app

SLOW_LOGGER = "addaxai.app.main"


@pytest.fixture()
def make_client(monkeypatch: pytest.MonkeyPatch) -> Iterator:
    """Build a client with a given threshold, plus a route that raises."""
    clients: list[TestClient] = []

    def _make(threshold: str) -> TestClient:
        monkeypatch.setenv("ADDAXAI_SLOW_REQUEST_SECONDS", threshold)
        with patch("app.main.init_db"):
            app = create_app()

        def _boom() -> dict:
            raise RuntimeError("intentional test failure")

        # Ahead of the SPA catch-all, which would match it first.
        app.routes.insert(0, APIRoute("/api/__test_500__", _boom, methods=["GET"]))
        client = TestClient(app)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()


def _slow_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == SLOW_LOGGER and r.getMessage().startswith("Slow request:")
    ]


def test_a_request_over_the_threshold_is_logged(make_client, caplog) -> None:
    client = make_client("0")
    with caplog.at_level(logging.WARNING, logger=SLOW_LOGGER):
        assert client.get("/health").status_code == 200

    [line] = _slow_lines(caplog)
    assert line.startswith("Slow request: GET /health 200 ")
    assert line.endswith("s")


def test_a_fast_request_is_not_logged(make_client, caplog) -> None:
    client = make_client("60")
    with caplog.at_level(logging.WARNING, logger=SLOW_LOGGER):
        client.get("/health")

    assert _slow_lines(caplog) == []


def test_the_query_string_is_never_logged(make_client, caplog) -> None:
    client = make_client("0")
    with caplog.at_level(logging.WARNING, logger=SLOW_LOGGER):
        client.get("/health", params={"path": "/Users/someone/camera-photos"})

    [line] = _slow_lines(caplog)
    assert "/health " in line
    assert "camera-photos" not in line
    assert "?" not in line


def test_a_failing_request_is_timed_with_its_500(make_client, caplog) -> None:
    client = make_client("0")
    with caplog.at_level(logging.WARNING, logger=SLOW_LOGGER):
        assert client.get("/api/__test_500__").status_code == 500

    [line] = _slow_lines(caplog)
    assert line.startswith("Slow request: GET /api/__test_500__ 500 ")
