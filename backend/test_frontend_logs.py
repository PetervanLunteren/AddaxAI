"""
Test frontend log forwarding to backend.

Verify that POST /api/logs endpoint works correctly.
"""

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def flush_logs() -> None:
    """Flush all log handlers to ensure logs are written to disk."""
    for handler in logging.getLogger("addaxai").handlers:
        handler.flush()


@pytest.fixture
def test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create test app with temporary log directory."""
    # Create temp directory
    user_data_dir = tmp_path / "AddaxAI"
    user_data_dir.mkdir()

    # Mock settings
    mock_settings = Mock()
    mock_settings.user_data_dir = user_data_dir
    mock_settings.environment = "test"
    mock_settings.debug = False
    mock_settings.database_url = f"sqlite:///{tmp_path}/test.db"
    mock_settings.api_host = "127.0.0.1"
    mock_settings.api_port = 8000
    mock_settings.models_dir = user_data_dir / "models"
    mock_settings.model_manifests_dir = user_data_dir / "models" / "manifests"
    mock_settings.model_weights_dir = user_data_dir / "models" / "weights"
    mock_settings.model_environments_dir = user_data_dir / "models" / "environments"

    # Create model directories
    mock_settings.models_dir.mkdir(parents=True, exist_ok=True)
    mock_settings.model_manifests_dir.mkdir(parents=True, exist_ok=True)
    mock_settings.model_weights_dir.mkdir(parents=True, exist_ok=True)
    mock_settings.model_environments_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("app.core.config.get_settings", lambda: mock_settings)
    monkeypatch.setattr("app.core.logging_config.get_settings", lambda: mock_settings)
    monkeypatch.setattr("app.db.base.get_settings", lambda: mock_settings)

    # Initialize logging with test settings (must be done BEFORE create_app)
    from app.core.logging_config import setup_logging

    setup_logging()

    # Initialize database tables
    from app.db.base import Base, get_engine
    from app.models import audit_log, deployment, event, file, job, project, site  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    app = create_app()
    return TestClient(app)


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    """Get path to log file."""
    return tmp_path / "AddaxAI" / "logs" / "backend.log"


def test_frontend_logs_endpoint_accepts_logs(test_app: TestClient) -> None:
    """Test that POST /api/logs accepts frontend logs."""
    response = test_app.post(
        "/api/logs",
        json={
            "logs": [
                {
                    "timestamp": "2024-12-15T10:30:00.000Z",
                    "level": "info",
                    "message": "User clicked button",
                    "context": {"buttonId": "create-project"},
                }
            ]
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "success"
    assert "1 entries" in response.json()["message"]


def test_frontend_logs_written_to_file(test_app: TestClient, log_file: Path) -> None:
    """Test that frontend logs are written to backend.log."""
    test_app.post(
        "/api/logs",
        json={
            "logs": [
                {
                    "timestamp": "2024-12-15T11:00:00.000Z",
                    "level": "info",
                    "message": "Frontend test message",
                    "context": None,
                }
            ]
        },
    )

    # Check log file
    flush_logs()
    content = log_file.read_text()
    assert "[INFO]" in content
    assert "[addaxai.frontend]" in content
    assert "Frontend test message" in content
    assert "2024-12-15T11:00:00.000Z" in content


def test_frontend_error_logs_with_context(test_app: TestClient, log_file: Path) -> None:
    """Test that frontend error logs include context."""
    test_app.post(
        "/api/logs",
        json={
            "logs": [
                {
                    "timestamp": "2024-12-15T12:00:00.000Z",
                    "level": "error",
                    "message": "API call failed",
                    "context": {
                        "endpoint": "/api/projects",
                        "error": "Network timeout",
                    },
                }
            ]
        },
    )

    # Check log file
    flush_logs()
    content = log_file.read_text()
    assert "[ERROR]" in content
    assert "[addaxai.frontend]" in content
    assert "API call failed" in content
    assert "endpoint" in content
    assert "/api/projects" in content


def test_frontend_warning_logs(test_app: TestClient, log_file: Path) -> None:
    """Test that frontend warning logs work."""
    test_app.post(
        "/api/logs",
        json={
            "logs": [
                {
                    "timestamp": "2024-12-15T13:00:00.000Z",
                    "level": "warn",
                    "message": "Slow response detected",
                    "context": {"duration": 5000},
                }
            ]
        },
    )

    # Check log file
    flush_logs()
    content = log_file.read_text()
    assert "[WARNING]" in content
    assert "[addaxai.frontend]" in content
    assert "Slow response detected" in content


def test_batched_frontend_logs(test_app: TestClient, log_file: Path) -> None:
    """Test that multiple frontend logs can be sent in one batch."""
    response = test_app.post(
        "/api/logs",
        json={
            "logs": [
                {
                    "timestamp": "2024-12-15T14:00:00.000Z",
                    "level": "info",
                    "message": "First log",
                    "context": None,
                },
                {
                    "timestamp": "2024-12-15T14:00:01.000Z",
                    "level": "info",
                    "message": "Second log",
                    "context": None,
                },
                {
                    "timestamp": "2024-12-15T14:00:02.000Z",
                    "level": "error",
                    "message": "Third log",
                    "context": {"error": "Something broke"},
                },
            ]
        },
    )

    assert response.status_code == 201
    assert "3 entries" in response.json()["message"]

    # Check all logs appear
    flush_logs()
    content = log_file.read_text()
    assert "First log" in content
    assert "Second log" in content
    assert "Third log" in content
    assert "Something broke" in content


def test_frontend_logs_with_no_context(test_app: TestClient, log_file: Path) -> None:
    """Test that logs without context work correctly."""
    test_app.post(
        "/api/logs",
        json={
            "logs": [
                {
                    "timestamp": "2024-12-15T15:00:00.000Z",
                    "level": "info",
                    "message": "Simple log message",
                    "context": None,
                }
            ]
        },
    )

    # Check log file
    flush_logs()
    content = log_file.read_text()
    assert "Simple log message" in content
    assert "[addaxai.frontend]" in content
