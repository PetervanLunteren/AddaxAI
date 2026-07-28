"""Tests for the /api/ml endpoints."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_managers():
    """Patch _get_managers to return controllable mocks."""
    mock_manifest = MagicMock()
    mock_env = MagicMock()
    mock_storage = MagicMock()

    # Default: no models
    mock_manifest.get_detection_models.return_value = {}
    mock_manifest.get_classification_models.return_value = {}
    mock_manifest.get_embedding_models.return_value = {}

    with patch(
        "app.api.routers.ml_models._get_managers",
        return_value=(mock_manifest, mock_env, mock_storage),
    ):
        # Also patch module-level globals used by prepare endpoints
        with patch("app.api.routers.ml_models.manifest_manager", mock_manifest):
            with patch("app.api.routers.ml_models.model_storage", mock_storage):
                with patch("app.api.routers.ml_models.env_manager", mock_env):
                    yield mock_manifest, mock_env, mock_storage


def test_list_detection_models(client, mock_managers):
    mock_manifest, _, _ = mock_managers
    mock_manifest.get_detection_models.return_value = {}
    resp = client.get("/api/ml/models/detection")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_classification_models(client, mock_managers):
    mock_manifest, _, _ = mock_managers
    mock_manifest.get_classification_models.return_value = {}
    resp = client.get("/api/ml/models/classification")
    assert resp.status_code == 200
    # Always includes the "none" option
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["model_id"] == "none"


def test_list_embedding_models(client, mock_managers):
    mock_manifest, _, _ = mock_managers
    mock_manifest.get_embedding_models.return_value = {}
    resp = client.get("/api/ml/models/embedding")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["model_id"] == "none"


def test_get_model_status_ready(client, mock_managers):
    mock_manifest, mock_env, mock_storage = mock_managers
    manifest = MagicMock()
    manifest.model_id = "test-model"
    manifest.friendly_name = "Test Model"
    manifest.env = "test-env"
    mock_manifest.get_model.return_value = manifest
    mock_storage.check_weights_ready.return_value = True
    mock_storage.get_weights_size.return_value = 100.0
    mock_env.envs_dir = MagicMock()
    env_path = MagicMock()
    env_path.exists.return_value = True
    mock_env.envs_dir.__truediv__ = MagicMock(return_value=env_path)
    mock_env._validate_env.return_value = True

    resp = client.get("/api/ml/models/test-model/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_get_model_status_needs_weights(client, mock_managers):
    mock_manifest, mock_env, mock_storage = mock_managers
    manifest = MagicMock()
    manifest.model_id = "test-model"
    manifest.friendly_name = "Test Model"
    manifest.env = "test-env"
    mock_manifest.get_model.return_value = manifest
    mock_storage.check_weights_ready.return_value = False
    mock_storage.get_weights_size.return_value = None
    mock_env.envs_dir = MagicMock()
    env_path = MagicMock()
    env_path.exists.return_value = True
    mock_env.envs_dir.__truediv__ = MagicMock(return_value=env_path)
    mock_env._validate_env.return_value = True

    resp = client.get("/api/ml/models/test-model/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "needs_weights"


def test_get_model_status_not_found(client, mock_managers):
    mock_manifest, _, _ = mock_managers
    mock_manifest.get_model.side_effect = ValueError("Model not found")
    resp = client.get("/api/ml/models/nonexistent/status")
    assert resp.status_code == 404


def test_prepare_model(client, mock_managers):
    mock_manifest, _, _ = mock_managers
    manifest = MagicMock()
    manifest.model_id = "test-model"
    mock_manifest.get_model.return_value = manifest
    with patch("app.api.routers.ml_models.ws_manager"):
        resp = client.post("/api/ml/models/test-model/prepare")
    assert resp.status_code == 200
    assert resp.json()["task_id"] == "test-model"


def test_prepare_model_not_found(client, mock_managers):
    mock_manifest, _, _ = mock_managers
    mock_manifest.get_model.side_effect = ValueError("Model not found")
    resp = client.post("/api/ml/models/nonexistent/prepare")
    assert resp.status_code == 404
