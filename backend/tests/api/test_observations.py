"""Tests for the /api/projects/{id}/observations endpoints."""

from unittest.mock import patch

from app.api.schemas.similarity import (
    DetectionSummary,
    SearchResponse,
    SortResponse,
)
from tests.conftest import make_project


def test_get_observation_stats(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/observations/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_detections"] == 0
    assert data["embedded_detections"] == 0


def test_sort_observations_success(client, db):
    p = make_project(db)
    mock_result = SortResponse(detections=[], total_detections=0)
    with patch(
        "app.api.routers.observations.sort_detections_service",
        return_value=mock_result,
    ):
        resp = client.post(
            f"/api/projects/{p.id}/observations/sort",
            json={"filters": {}, "reverse": False},
        )
    assert resp.status_code == 200


def test_sort_observations_error(client, db):
    p = make_project(db)
    with patch(
        "app.api.routers.observations.sort_detections_service",
        side_effect=FileNotFoundError("script not found"),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/observations/sort",
            json={"filters": {}, "reverse": False},
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
    )
    with patch(
        "app.api.routers.observations.search_similar_service",
        return_value=mock_result,
    ):
        resp = client.post(
            f"/api/projects/{p.id}/observations/search",
            json={"anchor_detection_id": "abc", "filters": {}},
        )
    assert resp.status_code == 200


def test_search_observations_error(client, db):
    p = make_project(db)
    with patch(
        "app.api.routers.observations.search_similar_service",
        side_effect=FileNotFoundError("script not found"),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/observations/search",
            json={"anchor_detection_id": "abc", "filters": {}},
        )
    assert resp.status_code == 503
