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


# --- POST /api/ml/models/{id}/update ---------------------------------------


def test_update_model_returns_the_files_it_refreshed(client, mock_managers):
    _, _, mock_storage = mock_managers
    mock_storage.update_stale_files.return_value = ["inference.py", "taxonomy.csv"]

    resp = client.post("/api/ml/models/test-model/update")

    assert resp.status_code == 200
    body = resp.json()
    assert body["updated_files"] == ["inference.py", "taxonomy.csv"]
    assert body["model_id"] == "test-model"


def test_update_model_when_already_in_sync(client, mock_managers):
    """Nothing to do is a success, not an error."""
    _, _, mock_storage = mock_managers
    mock_storage.update_stale_files.return_value = []

    resp = client.post("/api/ml/models/test-model/update")

    assert resp.status_code == 200
    assert resp.json()["updated_files"] == []


def test_update_model_unknown_model(client, mock_managers):
    mock_manifest, _, _ = mock_managers
    mock_manifest.get_model.side_effect = ValueError("Model not found")
    resp = client.post("/api/ml/models/nonexistent/update")
    assert resp.status_code == 404


def test_update_model_not_installed(client, mock_managers):
    _, _, mock_storage = mock_managers
    mock_storage.update_stale_files.side_effect = FileNotFoundError("no weights")
    resp = client.post("/api/ml/models/test-model/update")
    assert resp.status_code == 409


def test_update_model_offline(client, mock_managers):
    _, _, mock_storage = mock_managers
    mock_storage.update_stale_files.side_effect = ConnectionError("offline")
    resp = client.post("/api/ml/models/test-model/update")
    assert resp.status_code == 503


def test_update_model_file_in_use(client, mock_managers):
    """Windows refusing to replace a file a running analysis holds open."""
    _, _, mock_storage = mock_managers
    mock_storage.update_stale_files.side_effect = PermissionError("in use")
    resp = client.post("/api/ml/models/test-model/update")
    assert resp.status_code == 409
    assert "running analysis" in resp.json()["detail"]


def test_update_model_download_failure(client, mock_managers):
    _, _, mock_storage = mock_managers
    mock_storage.update_stale_files.side_effect = RuntimeError("download failed")
    resp = client.post("/api/ml/models/test-model/update")
    assert resp.status_code == 500


def test_update_model_refuses_while_a_download_runs(client, mock_managers):
    """Two writers in the same model directory is never worth allowing."""
    from app.api.routers import ml_models

    _, _, mock_storage = mock_managers
    ml_models._active_prepares.add("test-model-weights")
    try:
        resp = client.post("/api/ml/models/test-model/update")
    finally:
        ml_models._active_prepares.discard("test-model-weights")

    assert resp.status_code == 409
    mock_storage.update_stale_files.assert_not_called()


def test_update_model_clears_the_startup_snapshot(client, mock_managers):
    """
    /api/ml/updates serves a snapshot taken at startup, so without this a
    window reload would offer an update that already happened.
    """
    _, _, mock_storage = mock_managers
    mock_storage.update_stale_files.return_value = ["inference.py"]
    client.app.state.model_updates = {
        "drifted_models": [
            {"model_id": "test-model", "friendly_name": "T", "emoji": "x"},
            {"model_id": "other-model", "friendly_name": "O", "emoji": "y"},
        ]
    }

    client.post("/api/ml/models/test-model/update")

    remaining = client.app.state.model_updates["drifted_models"]
    assert [m["model_id"] for m in remaining] == ["other-model"]


def test_redownload_endpoint_is_gone(client, mock_managers):
    """
    The old fire-and-forget contract is retired, not aliased, so a stale
    client fails loudly rather than silently doing nothing.

    Which 4xx it is depends on the environment, so this asserts the route
    is absent rather than pinning a status code. `main.create_app` only
    mounts the SPA catch-all (`GET /{full_path:path}`) when
    `frontend/dist` exists: with a built frontend the path matches that
    GET route and POST gives 405, without one there is no match at all
    and it is 404. This used to assert 405, which passed for anyone who
    had run a frontend build and failed in CI, where the backend job
    never builds it.
    """
    routes = {
        getattr(r, "path", None) for r in client.app.routes
    }
    assert "/api/ml/models/{model_id}/redownload" not in routes

    resp = client.post("/api/ml/models/test-model/redownload")
    assert resp.status_code in (404, 405)
