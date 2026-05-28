"""Tests for the /api/projects/{id}/observations endpoints."""

import json
from unittest.mock import patch

from app.api.schemas.observation import DetectionSummary, SearchResponse, SortResponse
from tests.conftest import make_project


def _ndjson_stream(*events: dict) -> list[bytes]:
    """Build an NDJSON byte-line list mimicking the subprocess stream."""
    return [(json.dumps(e) + "\n").encode("utf-8") for e in events]


def _read_result(resp) -> dict:
    """Parse the streamed NDJSON response and return the final result event."""
    last = None
    for line in resp.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if event["type"] == "result":
            last = event
    assert last is not None, "no result event in response"
    return last


def test_get_observation_stats(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/observations/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_detections"] == 0
    assert data["embedded_detections"] == 0


def test_sort_observations_success(client, db):
    p = make_project(db)
    mock_result = SortResponse(detections=[], total_detections=0).model_dump()
    events = _ndjson_stream(
        {"type": "progress", "phase": "load", "done": 0, "total": 0},
        {"type": "result", **mock_result},
    )
    with patch(
        "app.api.routers.observations.stream_sort_async",
        return_value=iter(events),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/observations/sort",
            json={"filters": {}, "sort": "similarity"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    final = _read_result(resp)
    assert final["total_detections"] == 0


def test_sort_observations_default_sort(client, db):
    """Body without `sort` defaults to similarity (matches schema)."""
    p = make_project(db)
    mock_result = SortResponse(detections=[], total_detections=0).model_dump()
    events = _ndjson_stream({"type": "result", **mock_result})
    with patch(
        "app.api.routers.observations.stream_sort_async",
        return_value=iter(events),
    ) as mock_sort:
        resp = client.post(
            f"/api/projects/{p.id}/observations/sort",
            json={"filters": {}},
        )
        # Force the streaming response to be consumed so the mock is invoked.
        list(resp.iter_lines())
    assert resp.status_code == 200
    # New signature: stream_sort_async(request, project_id, body, db).
    body = mock_sort.call_args.args[2]
    assert body.sort == "similarity"


def test_sort_observations_rejects_unknown_mode(client, db):
    p = make_project(db)
    resp = client.post(
        f"/api/projects/{p.id}/observations/sort",
        json={"filters": {}, "sort": "bogus"},
    )
    assert resp.status_code == 422


def test_sort_observations_error(client, db):
    p = make_project(db)
    with patch(
        "app.api.routers.observations.stream_sort_async",
        side_effect=FileNotFoundError("script not found"),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/observations/sort",
            json={"filters": {}, "sort": "similarity"},
        )
    assert resp.status_code == 503


def test_search_observations_success(client, db):
    p = make_project(db)
    mock_anchor = DetectionSummary(
        detection_id="det-1",
        file_id="file-1",
        label=None,
        label_confidence=None,
        confidence=0.9,
        category="animal",
        verified=False,
        classification_method=None,
        crop_url="/api/detections/det-1/crop",
    )
    mock_result = SearchResponse(
        anchor=mock_anchor, results=[], total_results=0, threshold_applied=0.0,
    ).model_dump()
    events = _ndjson_stream({"type": "result", **mock_result})
    with patch(
        "app.api.routers.observations.stream_search_async",
        return_value=iter(events),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/observations/search",
            json={"anchor_detection_id": "abc", "filters": {}},
        )
    assert resp.status_code == 200
    final = _read_result(resp)
    assert final["total_results"] == 0


def test_search_observations_error(client, db):
    p = make_project(db)
    with patch(
        "app.api.routers.observations.stream_search_async",
        side_effect=FileNotFoundError("script not found"),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/observations/search",
            json={"anchor_detection_id": "abc", "filters": {}},
        )
    assert resp.status_code == 503
