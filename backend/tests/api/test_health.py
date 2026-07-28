"""Tests for the /health endpoint."""


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_has_required_fields(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "version" in data
    assert "environment" in data


def test_health_environment_is_test(client):
    data = client.get("/health").json()
    assert data["environment"] == "test"
