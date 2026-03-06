"""Tests for the /api/logs endpoint."""


def test_forward_logs_success(client):
    resp = client.post("/api/logs", json={
        "logs": [
            {"timestamp": "2024-01-01T00:00:00Z", "level": "info", "message": "test msg"}
        ]
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "success"


def test_forward_logs_empty_list(client):
    resp = client.post("/api/logs", json={"logs": []})
    assert resp.status_code == 201


def test_forward_logs_all_levels(client):
    logs = [
        {"timestamp": "2024-01-01T00:00:00Z", "level": "info", "message": "info msg"},
        {"timestamp": "2024-01-01T00:00:01Z", "level": "warn", "message": "warn msg"},
        {"timestamp": "2024-01-01T00:00:02Z", "level": "error", "message": "error msg"},
    ]
    resp = client.post("/api/logs", json={"logs": logs})
    assert resp.status_code == 201
    assert "3" in resp.json()["message"]
