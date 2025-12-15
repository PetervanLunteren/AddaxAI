"""
Test API operation logging.

Verify that all CRUD operations and errors are properly logged.
"""

import logging
import tempfile
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


def test_project_create_is_logged(test_app: TestClient, log_file: Path) -> None:
    """Test that creating a project is logged."""
    response = test_app.post(
        "/api/projects", json={"name": "Test Project", "description": "Test description"}
    )
    assert response.status_code == 201

    project_id = response.json()["id"]

    # Check logs
    flush_logs()
    content = log_file.read_text()
    assert "[INFO]" in content
    assert "Created project: Test Project" in content
    assert f"(ID: {project_id})" in content


def test_project_not_found_is_logged(test_app: TestClient, log_file: Path) -> None:
    """Test that 404 errors are logged."""
    response = test_app.get("/api/projects/nonexistent-id")
    assert response.status_code == 404

    # Check logs
    flush_logs()
    content = log_file.read_text()
    assert "[WARNING]" in content
    assert "Project not found: nonexistent-id" in content


def test_project_delete_is_logged(test_app: TestClient, log_file: Path) -> None:
    """Test that deleting a project is logged."""
    # Create project first
    response = test_app.post(
        "/api/projects", json={"name": "Delete Me", "description": "Will be deleted"}
    )
    project_id = response.json()["id"]

    # Delete it
    response = test_app.delete(f"/api/projects/{project_id}")
    assert response.status_code == 204

    # Check logs
    flush_logs()
    content = log_file.read_text()
    assert f"Deleted project: {project_id}" in content
    assert "(cascaded to all related data)" in content


def test_duplicate_project_name_is_logged(test_app: TestClient, log_file: Path) -> None:
    """Test that duplicate name conflicts are logged."""
    # Create first project
    test_app.post("/api/projects", json={"name": "Duplicate", "description": "First"})

    # Try to create duplicate
    response = test_app.post(
        "/api/projects", json={"name": "Duplicate", "description": "Second"}
    )
    assert response.status_code == 409

    # Check logs
    flush_logs()
    content = log_file.read_text()
    assert "[WARNING]" in content
    assert "Failed to create project 'Duplicate': duplicate name" in content


def test_site_create_is_logged(test_app: TestClient, log_file: Path) -> None:
    """Test that creating a site is logged."""
    # Create project first
    project_response = test_app.post(
        "/api/projects", json={"name": "Parent Project", "description": "For site test"}
    )
    project_id = project_response.json()["id"]

    # Create site
    response = test_app.post(
        "/api/sites",
        json={
            "project_id": project_id,
            "name": "Test Site",
            "latitude": 10.5,
            "longitude": 20.5,
        },
    )
    assert response.status_code == 201

    site_id = response.json()["id"]

    # Check logs
    flush_logs()
    content = log_file.read_text()
    assert f"Created site: Test Site in project {project_id}" in content
    assert f"(ID: {site_id})" in content


def test_site_invalid_project_is_logged(test_app: TestClient, log_file: Path) -> None:
    """Test that foreign key errors are logged."""
    response = test_app.post(
        "/api/sites",
        json={
            "project_id": "nonexistent",
            "name": "Orphan Site",
            "latitude": 0.0,
            "longitude": 0.0,
        },
    )
    assert response.status_code == 400

    # Check logs
    flush_logs()
    content = log_file.read_text()
    assert "[WARNING]" in content
    assert "Failed to create site: project nonexistent not found" in content


def test_deployment_folder_scan_is_logged(
    test_app: TestClient, log_file: Path, tmp_path: Path
) -> None:
    """Test that folder scanning operations are logged."""
    # Create project and site
    project_response = test_app.post(
        "/api/projects", json={"name": "Scan Project", "description": "Test"}
    )
    project_id = project_response.json()["id"]

    site_response = test_app.post(
        "/api/sites",
        json={
            "project_id": project_id,
            "name": "Scan Site",
            "latitude": 0.0,
            "longitude": 0.0,
        },
    )
    site_id = site_response.json()["id"]

    # Create deployment with folder path
    test_folder = tmp_path / "test_deployment"
    test_folder.mkdir()

    deployment_response = test_app.post(
        "/api/deployments",
        json={
            "site_id": site_id,
            "start_date": "2024-01-01",
            "folder_path": str(test_folder),
        },
    )
    deployment_id = deployment_response.json()["id"]

    # Scan folder
    scan_response = test_app.post(f"/api/deployments/{deployment_id}/preview-folder")
    assert scan_response.status_code == 200

    # Check logs
    flush_logs()
    content = log_file.read_text()
    assert f"Scanning folder for deployment {deployment_id}" in content
    assert "Folder scan complete" in content


def test_deployment_folder_not_found_is_logged(
    test_app: TestClient, log_file: Path
) -> None:
    """Test that folder not found errors are logged."""
    # Create project, site, deployment
    project_response = test_app.post(
        "/api/projects", json={"name": "Error Project", "description": "Test"}
    )
    project_id = project_response.json()["id"]

    site_response = test_app.post(
        "/api/sites",
        json={
            "project_id": project_id,
            "name": "Error Site",
            "latitude": 0.0,
            "longitude": 0.0,
        },
    )
    site_id = site_response.json()["id"]

    deployment_response = test_app.post(
        "/api/deployments",
        json={
            "site_id": site_id,
            "start_date": "2024-01-01",
            "folder_path": "/nonexistent/path",
        },
    )
    deployment_id = deployment_response.json()["id"]

    # Try to scan nonexistent folder
    scan_response = test_app.post(f"/api/deployments/{deployment_id}/preview-folder")
    assert scan_response.status_code == 400

    # Check logs
    flush_logs()
    content = log_file.read_text()
    assert "[ERROR]" in content
    assert f"Folder not found for deployment {deployment_id}" in content
