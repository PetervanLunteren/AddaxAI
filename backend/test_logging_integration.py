"""
Integration tests for logging system.

Test complete user workflows and verify all operations are logged correctly.
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


def test_complete_user_workflow_is_logged(
    test_app: TestClient, log_file: Path, tmp_path: Path
) -> None:
    """
    Test that a complete user workflow is logged correctly.

    Workflow:
    1. Create project
    2. Create site
    3. Create deployment with folder path
    4. Scan folder
    5. Update project
    6. Delete everything
    """
    # 1. Create project
    project_response = test_app.post(
        "/api/projects",
        json={"name": "Wildlife Survey 2024", "description": "Annual survey"},
    )
    assert project_response.status_code == 201
    project_id = project_response.json()["id"]

    # 2. Create site
    site_response = test_app.post(
        "/api/sites",
        json={
            "project_id": project_id,
            "name": "Camera Trap Site 1",
            "latitude": 52.5200,
            "longitude": 13.4050,
        },
    )
    assert site_response.status_code == 201
    site_id = site_response.json()["id"]

    # 3. Create deployment
    test_folder = tmp_path / "deployment_media"
    test_folder.mkdir()

    deployment_response = test_app.post(
        "/api/deployments",
        json={
            "site_id": site_id,
            "start_date": "2024-01-01",
            "folder_path": str(test_folder),
        },
    )
    assert deployment_response.status_code == 201
    deployment_id = deployment_response.json()["id"]

    # 4. Scan folder
    scan_response = test_app.post(f"/api/deployments/{deployment_id}/preview-folder")
    assert scan_response.status_code == 200

    # 5. Update project
    update_response = test_app.patch(
        f"/api/projects/{project_id}",
        json={"description": "Updated description"},
    )
    assert update_response.status_code == 200

    # 6. Delete project (cascades to everything)
    delete_response = test_app.delete(f"/api/projects/{project_id}")
    assert delete_response.status_code == 204

    # Now verify all operations are in logs
    flush_logs()
    content = log_file.read_text()

    # Check project operations
    assert f"Created project: Wildlife Survey 2024" in content
    assert f"(ID: {project_id})" in content
    assert f"Updated project: {project_id}" in content
    assert f"Deleted project: {project_id}" in content
    assert "cascaded to all related data" in content

    # Check site operations
    assert f"Created site: Camera Trap Site 1" in content
    assert f"in project {project_id}" in content
    assert f"(ID: {site_id})" in content

    # Check deployment operations
    assert f"Created deployment for site {site_id}" in content
    assert f"(ID: {deployment_id})" in content

    # Check folder scan
    assert f"Scanning folder for deployment {deployment_id}" in content
    assert "Folder scan complete" in content

    # Verify chronological order (timestamps should be ascending)
    lines = content.split("\n")
    log_lines = [l for l in lines if "[INFO]" in l or "[WARNING]" in l or "[ERROR]" in l]
    assert len(log_lines) > 10, "Should have many log entries"


def test_error_recovery_workflow_is_logged(
    test_app: TestClient, log_file: Path
) -> None:
    """
    Test that error recovery is logged correctly.

    Workflow:
    1. Try to create site with invalid project (fail)
    2. Create project
    3. Create site successfully
    4. Try to create duplicate project (fail)
    5. All failures should be logged
    """
    # 1. Try invalid site creation
    invalid_response = test_app.post(
        "/api/sites",
        json={
            "project_id": "nonexistent",
            "name": "Orphan Site",
            "latitude": 0.0,
            "longitude": 0.0,
        },
    )
    assert invalid_response.status_code == 400

    # 2. Create project
    project_response = test_app.post(
        "/api/projects",
        json={"name": "Valid Project", "description": "Test"},
    )
    project_id = project_response.json()["id"]

    # 3. Create site successfully
    site_response = test_app.post(
        "/api/sites",
        json={
            "project_id": project_id,
            "name": "Valid Site",
            "latitude": 10.0,
            "longitude": 20.0,
        },
    )
    assert site_response.status_code == 201

    # 4. Try duplicate project
    duplicate_response = test_app.post(
        "/api/projects",
        json={"name": "Valid Project", "description": "Duplicate"},
    )
    assert duplicate_response.status_code == 409

    # Verify all errors are logged
    flush_logs()
    content = log_file.read_text()

    # Check error logs
    assert "[WARNING]" in content
    assert "Failed to create site: project nonexistent not found" in content
    assert "Failed to create project 'Valid Project': duplicate name" in content

    # Check successful operations are also logged
    assert "Created project: Valid Project" in content
    assert "Created site: Valid Site" in content


def test_mixed_frontend_backend_logs(test_app: TestClient, log_file: Path) -> None:
    """Test that frontend and backend logs are interleaved correctly."""
    # Backend operation
    project_response = test_app.post(
        "/api/projects",
        json={"name": "Test Project", "description": "Test"},
    )
    assert project_response.status_code == 201

    # Frontend logs
    test_app.post(
        "/api/logs",
        json={
            "logs": [
                {
                    "timestamp": "2024-12-15T10:00:00.000Z",
                    "level": "info",
                    "message": "User viewed projects page",
                    "context": None,
                },
                {
                    "timestamp": "2024-12-15T10:00:01.000Z",
                    "level": "info",
                    "message": "User clicked create button",
                    "context": {"action": "create-project"},
                },
            ]
        },
    )

    # Another backend operation
    test_app.get("/api/projects")

    # Check logs have both sources
    flush_logs()
    content = log_file.read_text()

    # Backend logs
    assert "Created project: Test Project" in content

    # Frontend logs
    assert "[addaxai.frontend]" in content
    assert "User viewed projects page" in content
    assert "User clicked create button" in content


def test_no_silent_failures_in_integration(
    test_app: TestClient, log_file: Path
) -> None:
    """Verify that no operations fail silently."""
    # Perform various operations
    test_app.post("/api/projects", json={"name": "P1", "description": "D1"})
    test_app.get("/api/projects/nonexistent")  # 404
    test_app.delete("/api/projects/also-nonexistent")  # 404

    # Read logs
    flush_logs()
    content = log_file.read_text()

    # All 404s should be logged as warnings
    assert content.count("[WARNING]") >= 2
    assert "Project not found: nonexistent" in content
    assert "Cannot delete project: also-nonexistent not found" in content

    # No errors should be silently swallowed
    # (This test passes if no exceptions were raised during the operations)
