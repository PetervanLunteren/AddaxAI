"""Tests for the /api/projects/{id}/observations endpoints."""

import json
import uuid
from unittest.mock import patch

from app.api.schemas.label import DetectionSummary, SearchResponse, SortResponse
from app.models import DetectionEmbedding
from tests.conftest import make_deployment, make_detection, make_file, make_project


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
    resp = client.get(f"/api/projects/{p.id}/labels/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_detections"] == 0
    assert data["embedded_detections"] == 0


def test_observation_stats_count_only_this_project(client, db):
    """The three counts of one pass, each with its own rule, and nothing
    from another project's boxes or embeddings."""
    model = "DINOV2-VITS14"

    def box(project, *, confidence=0.9, verified=False, embed_model=None):
        dep = make_deployment(db, project_id=project.id)
        f = make_file(db, deployment_id=dep.id)
        d = make_detection(db, file_id=f.id, confidence=confidence, verified=verified)
        if embed_model:
            db.add(
                DetectionEmbedding(
                    id=str(uuid.uuid4()),
                    detection_id=d.id,
                    embedding_model_id=embed_model,
                    vector=b"\0" * 8,
                    dimension=4,
                    l2_norm=1.0,
                )
            )
        return d

    p = make_project(db, embedding_model_id=model, classification_gate=0.5)
    other = make_project(db, embedding_model_id=model, classification_gate=0.5)
    box(p, embed_model=model)  # counted, embedded
    box(p, verified=True, embed_model=model)  # counted, embedded, reviewed
    box(p, verified=True, embed_model="OTHER-MODEL")  # counted, reviewed, not embedded
    box(p)  # counted, missing an embedding
    box(p, confidence=0.1)  # below the gate: none of the three
    box(other, verified=True, embed_model=model)  # another project
    db.flush()

    data = client.get(f"/api/projects/{p.id}/labels/stats").json()
    assert data["total_detections"] == 4
    assert data["verified_detections"] == 2
    assert data["embedded_detections"] == 2
    assert data["missing_embeddings"] == 2


def test_sort_observations_success(client, db):
    p = make_project(db)
    mock_result = SortResponse(detections=[], total_detections=0).model_dump()
    events = _ndjson_stream(
        {"type": "progress", "phase": "load", "done": 0, "total": 0},
        {"type": "result", **mock_result},
    )
    with patch(
        "app.api.routers.labels.stream_sort_async",
        return_value=iter(events),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/labels/sort",
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
        "app.api.routers.labels.stream_sort_async",
        return_value=iter(events),
    ) as mock_sort:
        resp = client.post(
            f"/api/projects/{p.id}/labels/sort",
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
        f"/api/projects/{p.id}/labels/sort",
        json={"filters": {}, "sort": "bogus"},
    )
    assert resp.status_code == 422


def test_sort_observations_error(client, db):
    p = make_project(db)
    with patch(
        "app.api.routers.labels.stream_sort_async",
        side_effect=FileNotFoundError("script not found"),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/labels/sort",
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
        "app.api.routers.labels.stream_search_async",
        return_value=iter(events),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/labels/search",
            json={"anchor_detection_id": "abc", "filters": {}},
        )
    assert resp.status_code == 200
    final = _read_result(resp)
    assert final["total_results"] == 0


def test_search_observations_error(client, db):
    p = make_project(db)
    with patch(
        "app.api.routers.labels.stream_search_async",
        side_effect=FileNotFoundError("script not found"),
    ):
        resp = client.post(
            f"/api/projects/{p.id}/labels/search",
            json={"anchor_detection_id": "abc", "filters": {}},
        )
    assert resp.status_code == 503


def test_unprocessed_count_counts_unembedded_in_range(client, db):
    """The labels grid's "unprocessed detections" banner counts
    embeddable detections in a confidence range that have no embedding
    for the project's current embedding model. Data-driven, so it is
    correct whatever classification gate each deployment ran under."""
    import uuid as _uuid

    import numpy as np

    from app.models.detection_embedding import DetectionEmbedding
    from tests.conftest import make_deployment, make_detection, make_file

    p = make_project(db, embedding_model_id="DINOV2-VITB14")
    dep = make_deployment(db, project_id=p.id)
    f = make_file(db, deployment_id=dep.id, observation_type="animal")
    common = dict(
        file_id=f.id, category="animal",
        bbox_x=0.1, bbox_y=0.1, bbox_width=0.2, bbox_height=0.2,
    )
    # In range, unembedded -> counted.
    make_detection(db, confidence=0.05, **common)
    make_detection(db, confidence=0.08, **common)
    # In range but embedded -> not counted.
    d_embedded = make_detection(db, confidence=0.06, **common)
    db.add(DetectionEmbedding(
        id=str(_uuid.uuid4()),
        detection_id=d_embedded.id,
        embedding_model_id="DINOV2-VITB14",
        vector=np.zeros(4, dtype=np.float16).tobytes(),
        dimension=4,
        l2_norm=0.0,
    ))
    # Out of range -> not counted.
    make_detection(db, confidence=0.5, **common)
    db.flush()

    resp = client.get(
        f"/api/projects/{p.id}/labels/unprocessed-count"
        f"?min_confidence=0.01&max_confidence=0.1"
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


def test_unprocessed_count_zero_without_embedding_model(client, db):
    p = make_project(db)
    p.embedding_model_id = None
    db.flush()
    resp = client.get(
        f"/api/projects/{p.id}/labels/unprocessed-count"
        f"?min_confidence=0.01&max_confidence=1.0"
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


# ── cohorts: the scope reaches the worker ────────────────────────────────


def _cohort_params_sent(client, project_id: str, query: str) -> dict:
    """Call the cohorts endpoint and return the params the worker got."""
    sent: dict = {}

    async def fake_stream(_request, operation, _project_id, params):
        sent.update(operation=operation, **params)
        for line in _ndjson_stream({"type": "result", "cohorts": []}):
            yield line

    with patch(
        "app.services.label_service.stream_labels_subprocess_async", fake_stream
    ):
        resp = client.get(f"/api/projects/{project_id}/labels/cohorts{query}")
        assert resp.status_code == 200
        assert _read_result(resp)["cohorts"] == []
    return sent


def test_cohorts_send_the_site_and_date_scope_to_the_worker(client, db):
    p = make_project(db, counting_threshold=0.2)
    sent = _cohort_params_sent(
        client,
        p.id,
        "?site_ids=site-a,null&date_from=2024-06-01&date_to=2024-09-30",
    )
    assert sent["operation"] == "cohorts"
    assert sent["filters"] == {
        "site_ids": ["site-a", "null"],
        "date_from": "2024-06-01 00:00:00",
        "date_to": "2024-09-30 23:59:59.999999",
        "project_floor": 0.2,
    }


def test_cohorts_without_a_scope_send_only_the_floor(client, db):
    p = make_project(db, counting_threshold=0.2)
    sent = _cohort_params_sent(client, p.id, "")
    assert sent["filters"] == {"project_floor": 0.2}
    assert (sent["min_count"], sent["max_cohorts"]) == (8, 200)


def test_cohorts_reject_a_date_that_is_not_a_date(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/labels/cohorts?date_from=yesterday")
    assert resp.status_code == 422
