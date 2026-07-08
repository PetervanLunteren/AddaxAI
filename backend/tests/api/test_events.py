"""Tests for the /api/events endpoints."""

from datetime import datetime
from unittest.mock import patch

from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_project,
    make_site,
)


def test_generate_events(client, db):
    p = make_project(db)
    with patch(
        "app.api.routers.events.event_crud.generate_events_for_project",
        return_value=5,
    ):
        resp = client.post("/api/events/generate", json={"project_id": p.id})
    assert resp.status_code == 200
    assert resp.json()["event_count"] == 5


def test_generate_events_project_not_found(client):
    with patch(
        "app.api.routers.events.event_crud.generate_events_for_project",
        side_effect=ValueError("Project not found"),
    ):
        resp = client.post("/api/events/generate", json={"project_id": "nonexistent"})
    assert resp.status_code == 404


def test_list_events_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/events?project_id={p.id}")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_event_count(client, db):
    p = make_project(db)
    resp = client.get(f"/api/events/count?project_id={p.id}")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_get_verification_stats(client, db):
    p = make_project(db)
    resp = client.get(f"/api/events/verification-stats?project_id={p.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_files" in data
    assert "verified_files" in data


def test_get_event_not_found(client):
    resp = client.get("/api/events/nonexistent")
    assert resp.status_code == 404


def test_get_event_with_files(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = make_event_with_files(
        db,
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    resp = client.get(f"/api/events/{ev.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == ev.id
    assert len(data["files"]) == 1


def test_get_event_files_same_second_sort_alphabetically(client, db):
    # Burst shots often share one second-resolution EXIF timestamp; the
    # filmstrip must then fall back to the sequential camera filenames.
    from app.models.event import event_files as event_files_table
    from tests.conftest import make_file

    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = make_event_with_files(
        db,
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
        files_verified=[],
    )
    ts = datetime(2024, 1, 1, 12, 0, 7)
    # Insert out of alphabetical order so relationship order alone fails.
    for seq, name in enumerate(["IMG_0003.jpg", "IMG_0001.jpg", "IMG_0002.jpg"]):
        f = make_file(
            db,
            deployment_id=d.id,
            captured_at_local=ts,
            file_path=f"/cam01/{name}",
        )
        db.execute(
            event_files_table.insert().values(
                event_id=ev.id, file_id=f.id, sequence_number=seq
            )
        )
    db.commit()

    resp = client.get(f"/api/events/{ev.id}")
    assert resp.status_code == 200
    paths = [f["file_path"] for f in resp.json()["files"]]
    assert paths == [
        "/cam01/IMG_0001.jpg",
        "/cam01/IMG_0002.jpg",
        "/cam01/IMG_0003.jpg",
    ]


def test_get_adjacent_events(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = make_event_with_files(
        db,
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    resp = client.get(
        f"/api/events/{ev.id}/adjacent?project_id={p.id}"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "previous_id" in data
    assert "next_id" in data


def _event_with_detection(db, deployment_id, start):
    """Helper: event + one file + one visible detection so the project
    threshold filter doesn't drop the event."""
    ev = make_event_with_files(
        db,
        deployment_id=deployment_id,
        event_start_local=start,
    )
    make_detection(db, file_id=ev.files[0].id, confidence=0.9)
    db.commit()
    return ev


def test_events_filter_flagged_exists_on_any_file(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev_with_flag = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    ev_without = _event_with_detection(db, d.id, datetime(2024, 1, 2, 12, 0))
    flagged_file = ev_with_flag.files[0]
    client.patch(f"/api/files/{flagged_file.id}", json={"flagged": True})

    resp = client.get(f"/api/events?project_id={p.id}&flagged=flagged")
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()]
    assert ev_with_flag.id in ids
    assert ev_without.id not in ids


def test_events_filter_favorited_exists_on_any_file(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev_fav = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    ev_none = _event_with_detection(db, d.id, datetime(2024, 1, 2, 12, 0))
    fav_file = ev_fav.files[0]
    client.patch(f"/api/files/{fav_file.id}", json={"favorited": True})

    resp = client.get(f"/api/events?project_id={p.id}&favorited=favorited")
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()]
    assert ev_fav.id in ids
    assert ev_none.id not in ids


def test_events_filter_label_confidence_range(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev_high = make_event_with_files(
        db, deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    ev_low = make_event_with_files(
        db, deployment_id=d.id,
        event_start_local=datetime(2024, 1, 2, 12, 0),
    )
    make_detection(
        db, file_id=ev_high.files[0].id, confidence=0.9, label_confidence=0.9,
    )
    make_detection(
        db, file_id=ev_low.files[0].id, confidence=0.9, label_confidence=0.3,
    )
    db.commit()

    resp = client.get(
        f"/api/events?project_id={p.id}&min_label_confidence=0.5"
    )
    ids = [row["id"] for row in resp.json()]
    assert ev_high.id in ids
    assert ev_low.id not in ids


def test_events_filter_label_confidence_excludes_null(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    classified = make_event_with_files(
        db, deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    unclassified = make_event_with_files(
        db, deployment_id=d.id,
        event_start_local=datetime(2024, 1, 2, 12, 0),
    )
    make_detection(
        db, file_id=classified.files[0].id, confidence=0.9, label_confidence=0.3,
    )
    make_detection(
        db, file_id=unclassified.files[0].id, confidence=0.9, label_confidence=None,
    )
    db.commit()

    resp = client.get(
        f"/api/events?project_id={p.id}&min_label_confidence=0.0"
    )
    ids = [row["id"] for row in resp.json()]
    assert classified.id in ids
    assert unclassified.id not in ids


def test_events_filter_empty_show_only_and_hide(client, db):
    """Empty event = every file in it has observation_type='blank'."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev_animal = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    ev_blank = _event_with_detection(db, d.id, datetime(2024, 1, 2, 12, 0))
    # Mark all files in ev_blank as observation_type='blank'.
    for f in ev_blank.files:
        f.observation_type = "blank"
    db.commit()

    resp = client.get(f"/api/events?project_id={p.id}&empty=show_only")
    ids = [row["id"] for row in resp.json()]
    assert ev_blank.id in ids
    assert ev_animal.id not in ids

    resp = client.get(f"/api/events?project_id={p.id}&empty=hide")
    ids = [row["id"] for row in resp.json()]
    assert ev_animal.id in ids
    assert ev_blank.id not in ids


def test_event_summary_aggregates_any_file_flagged_and_favorited(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    f = ev.files[0]
    client.patch(f"/api/files/{f.id}", json={"flagged": True, "favorited": True})

    resp = client.get(f"/api/events?project_id={p.id}")
    assert resp.status_code == 200
    summary = next(row for row in resp.json() if row["id"] == ev.id)
    assert summary["any_file_flagged"] is True
    assert summary["any_file_favorited"] is True


def _setup_three_events(db):
    """Three events with distinct start times for sort tests."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    older = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    middle = _event_with_detection(db, d.id, datetime(2024, 2, 1, 12, 0))
    newer = _event_with_detection(db, d.id, datetime(2024, 3, 1, 12, 0))
    return p, older, middle, newer


def test_events_sort_oldest(client, db):
    p, older, middle, newer = _setup_three_events(db)

    resp = client.get(f"/api/events?project_id={p.id}&sort=oldest")
    ids = [row["id"] for row in resp.json()]
    assert ids == [older.id, middle.id, newer.id]


def test_events_sort_random_stable_with_seed(client, db):
    p, *_ = _setup_three_events(db)

    a = client.get(f"/api/events?project_id={p.id}&sort=random&seed=42").json()
    b = client.get(f"/api/events?project_id={p.id}&sort=random&seed=42").json()
    assert [r["id"] for r in a] == [r["id"] for r in b]

    # Three events have 6 permutations, so any specific seed pair has a
    # 1/6 chance of producing the same order. Probe seeds until we find
    # one that genuinely differs from seed 42, rather than asserting a
    # hard-coded pair and hoping for the best.
    base_ids = [r["id"] for r in a]
    for trial in range(43, 100):
        other = client.get(
            f"/api/events?project_id={p.id}&sort=random&seed={trial}",
        ).json()
        if [r["id"] for r in other] != base_ids:
            break
    else:
        raise AssertionError(
            "Seeds 43-99 all produced the same order as seed 42; "
            "seeded_hash UDF is likely broken.",
        )


def test_events_sort_cls_low_pushes_nulls_last(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    low = make_event_with_files(
        db, deployment_id=d.id, event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    high = make_event_with_files(
        db, deployment_id=d.id, event_start_local=datetime(2024, 2, 1, 12, 0),
    )
    null = make_event_with_files(
        db, deployment_id=d.id, event_start_local=datetime(2024, 3, 1, 12, 0),
    )
    make_detection(
        db, file_id=low.files[0].id, confidence=0.9, label_confidence=0.2,
    )
    make_detection(
        db, file_id=high.files[0].id, confidence=0.9, label_confidence=0.7,
    )
    make_detection(
        db, file_id=null.files[0].id, confidence=0.9, label_confidence=None,
    )
    db.commit()

    resp = client.get(f"/api/events?project_id={p.id}&sort=cls_low")
    ids = [row["id"] for row in resp.json()]
    assert ids == [low.id, high.id, null.id]


def test_events_adjacent_respects_sort(client, db):
    p, older, middle, newer = _setup_three_events(db)

    # Oldest-first: opening "older" → next is "middle".
    resp = client.get(
        f"/api/events/{older.id}/adjacent?project_id={p.id}&sort=oldest"
    ).json()
    assert resp["next_id"] == middle.id
    assert resp["previous_id"] is None

    # Newest-first (default): opening "older" → next is None.
    resp = client.get(
        f"/api/events/{older.id}/adjacent?project_id={p.id}"
    ).json()
    assert resp["next_id"] is None
    assert resp["previous_id"] == middle.id


def test_events_sort_invalid_value_returns_400(client, db):
    p = make_project(db)
    db.commit()

    resp = client.get(f"/api/events?project_id={p.id}&sort=bogus")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Collage file IDs (event card collage layout)
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from app.api.crud.event import _build_collage_file_ids  # noqa: E402


def _fake_file(file_id: str, confidences: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        id=file_id,
        detections=[SimpleNamespace(confidence=c) for c in confidences],
    )


def test_collage_uses_max_n_frames_first_in_order():
    max_n = [{"file_id": "a"}, {"file_id": "b"}]
    files = [_fake_file("a", [0.9]), _fake_file("b", [0.8]), _fake_file("c", [0.7])]
    assert _build_collage_file_ids(max_n, files) == ["a", "b", "c"]


def test_collage_caps_at_four_when_max_n_is_long():
    max_n = [{"file_id": x} for x in ("a", "b", "c", "d", "e")]
    assert _build_collage_file_ids(max_n, []) == ["a", "b", "c", "d"]


def test_collage_pads_by_top_detection_confidence():
    max_n = [{"file_id": "a"}]
    files = [
        _fake_file("a", [0.9]),
        _fake_file("b", [0.5]),
        _fake_file("c", [0.95]),
        _fake_file("d", [0.7]),
    ]
    # MaxN file first, then remaining files sorted by max(confidence) desc.
    assert _build_collage_file_ids(max_n, files) == ["a", "c", "d", "b"]


def test_collage_skips_files_already_in_max_n():
    max_n = [{"file_id": "a"}]
    files = [_fake_file("a", [0.9]), _fake_file("b", [0.8])]
    assert _build_collage_file_ids(max_n, files) == ["a", "b"]


def test_collage_no_max_n_uses_top_confidence_files():
    files = [
        _fake_file("a", [0.5]),
        _fake_file("b", [0.95]),
        _fake_file("c", [0.7]),
        _fake_file("d", [0.3]),
        _fake_file("e", [0.85]),
    ]
    assert _build_collage_file_ids([], files) == ["b", "e", "c", "a"]


def test_collage_empty_inputs_returns_empty_list():
    assert _build_collage_file_ids([], []) == []


def test_collage_handles_files_with_no_detections():
    files = [_fake_file("a", []), _fake_file("b", [0.9])]
    assert _build_collage_file_ids([], files) == ["b", "a"]


def test_event_summary_includes_collage_file_ids(client, db):
    """The events list endpoint exposes collage_file_ids on every summary."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))

    resp = client.get(f"/api/events?project_id={p.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    summary = body[0]
    assert summary["id"] == ev.id
    # No EventObservation rows in this fixture, so the only padding source
    # is the file by max detection confidence: one file → one collage tile.
    assert summary["collage_file_ids"] == [ev.files[0].id]


def test_event_verify_and_count_endpoints(client, db):
    """The Observations-page flow: read counts, verify, edit a count
    (which clears the sign-off), and add a species the AI missed."""
    from app.api.crud.event_observation import calculate_max_n_for_event
    from app.models.label_taxonomy import LabelTaxonomy

    p = make_project(db, detection_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = make_event_with_files(
        db, deployment_id=d.id, event_start_local=datetime(2024, 1, 1, 12)
    )
    det = make_detection(
        db, file_id=ev.files[0].id, category="animal", label="cow",
        confidence=0.9,
    )
    # Link a taxonomy so the count list resolves display names (regression
    # guard: the row item must read scientific_name / common_name).
    tax = LabelTaxonomy(
        name="cow", level="species", classification_model_id="",
        project_id=p.id, common_name="Cow", scientific_name="Bos taurus",
    )
    db.add(tax)
    db.flush()
    det.label_taxonomy_id = tax.id
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.commit()

    data = client.get(f"/api/events/{ev.id}").json()
    assert data["confirmed"] is False
    assert len(data["observations"]) == 1
    obs = data["observations"][0]
    assert obs["effective_count"] == 1
    assert obs["scientific_name"] == "Bos taurus"
    assert obs["common_name"] == "Cow"
    obs_id = obs["id"]

    data = client.patch(
        f"/api/events/{ev.id}/confirm", json={"confirmed": True}
    ).json()
    assert data["confirmed"] is True

    # Bumping a count clears the sign-off.
    data = client.patch(
        f"/api/events/{ev.id}/observations/{obs_id}", json={"count": 3}
    ).json()
    assert data["observations"][0]["effective_count"] == 3
    assert data["confirmed"] is False

    # Add a species the AI never detected.
    data = client.post(
        f"/api/events/{ev.id}/observations",
        json={"category": "animal", "label": "fox", "count": 2},
    ).json()
    by_label = {o["label"]: o for o in data["observations"]}
    assert by_label["fox"]["effective_count"] == 2

    # Reset to the AI proposal: the human-only fox row is gone, the cow
    # override is cleared back to its MaxN, and the sign-off is cleared.
    client.patch(f"/api/events/{ev.id}/confirm", json={"confirmed": True})
    data = client.post(f"/api/events/{ev.id}/observations/reset").json()
    assert data["confirmed"] is False
    assert len(data["observations"]) == 1
    assert data["observations"][0]["label"] == "cow"
    assert data["observations"][0]["effective_count"] == 1


def test_filter_options_reports_min_label_confidence(client, db):
    """The filter bars clamp the cls range slider at the lowest
    classification confidence that exists; the endpoint reports it."""
    from tests.conftest import make_deployment, make_detection, make_file, make_project

    p = make_project(db)
    dep = make_deployment(db, project_id=p.id)
    f = make_file(db, deployment_id=dep.id, observation_type="animal")
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.9,
        label="dog", label_confidence=0.42,
    )
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.8,
        label="cat", label_confidence=0.07,
    )

    resp = client.get(f"/api/events/filter-options?project_id={p.id}")
    assert resp.status_code == 200
    assert resp.json()["min_label_confidence"] == 0.07


def test_filter_options_min_label_confidence_null_when_unclassified(client, db):
    from tests.conftest import make_project

    p = make_project(db)
    resp = client.get(f"/api/events/filter-options?project_id={p.id}")
    assert resp.status_code == 200
    assert resp.json()["min_label_confidence"] is None
