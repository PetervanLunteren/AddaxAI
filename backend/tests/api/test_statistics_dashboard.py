"""The dashboard's scoped statistics.

Every label by default, like every label filter in the app; picked
labels narrow it. Wildlife only where a card asks for it (the Overview's
random animal).
Covers the scope shared by the activity clock, the trend, the summary
tiles and the demographics card, and the random photos.
"""

import uuid
from datetime import date, datetime

from app.api.crud import statistics as stats_crud
from app.models.event_observation import EventObservation
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def _taxon(db, name: str) -> LabelTaxonomy:
    row = LabelTaxonomy(
        id=str(uuid.uuid4()),
        classification_model_id="test-model",
        name=name,
        level="species",
        scientific_name=name.capitalize(),
        common_name=name.capitalize(),
    )
    db.add(row)
    db.flush()
    return row


def _observe(db, event, *, label, taxon=None, category="animal", max_n=1, **kw):
    obs = EventObservation(
        event_id=event.id,
        label=label,
        label_taxonomy_id=taxon.id if taxon else None,
        category=category,
        max_n=max_n,
        **kw,
    )
    db.add(obs)
    db.flush()
    return obs


def _project(db):
    """One site, one deployment, events in June and August 2024.

    June: 2 leopards. August: 3 lions (one cohort set to female adult
    resting) and 1 person, who is not wildlife but can be picked.
    """
    project = make_project(db, timezone="UTC")
    site = make_site(db, project_id=project.id, name="Waterhole")
    dep = make_deployment(
        db, site_id=site.id, start_date_local=date(2024, 6, 1)
    )
    leopard, lion = _taxon(db, "leopard"), _taxon(db, "lion")
    person = _taxon(db, "person")
    june = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 6, 10, 6, 0)
    )
    _observe(db, june, label="leopard", taxon=leopard, max_n=2)
    august = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 8, 20, 22, 0)
    )
    _observe(
        db, august, label="lion", taxon=lion, max_n=3,
        sex="female", life_stage="adult", behavior="resting",
    )
    _observe(db, august, label=None, taxon=person, category="person", max_n=1)
    return project, site, dep, leopard, lion, person


# ---------------------------------------------------------------------------
# Activity clock and trend
# ---------------------------------------------------------------------------


def test_trend_respects_the_date_window(db):
    """The window used to be accepted and ignored."""
    project, *_ = _project(db)

    points = stats_crud.get_detection_trend(
        db, project.id, date_from="2024-08-01", date_to="2024-08-31"
    )

    assert [(p.date, p.count) for p in points] == [("2024-08-20", 4)]


def test_activity_respects_the_date_window(db):
    project, *_ = _project(db)

    response = stats_crud.get_activity_pattern(
        db, project.id, date_from="2024-06-01", date_to="2024-06-30"
    )

    assert {h.hour: h.count for h in response.hours if h.count} == {6: 2}


def test_nothing_picked_means_every_label(db):
    project, *_ = _project(db)

    points = stats_crud.get_detection_trend(db, project.id)
    activity = stats_crud.get_activity_pattern(db, project.id)

    assert sum(p.count for p in points) == 6
    assert activity.total_observations == 6


def test_a_picked_person_is_counted(db):
    """A pick means exactly what was picked, people included."""
    project, *_, person = _project(db)

    activity = stats_crud.get_activity_pattern(
        db, project.id, label_taxonomy_ids=[person.id]
    )
    summary = stats_crud.get_dashboard_summary(
        db, project.id, label_taxonomy_ids=[person.id]
    )

    assert activity.total_observations == 1
    assert (summary.events, summary.observations) == (1, 1)


def test_trend_and_activity_narrow_to_the_taxon(db):
    project, _, _, leopard, _, _ = _project(db)

    points = stats_crud.get_detection_trend(
        db, project.id, label_taxonomy_ids=[leopard.id]
    )
    activity = stats_crud.get_activity_pattern(
        db, project.id, label_taxonomy_ids=[leopard.id]
    )

    assert [(p.date, p.count) for p in points] == [("2024-06-10", 2)]
    assert activity.total_observations == 2


def test_an_empty_taxon_list_is_refused_not_widened(client, db):
    project, *_ = _project(db)

    for path in (
        "activity-pattern", "detection-trend", "summary",
        "demographics",
    ):
        response = client.get(
            f"/api/statistics/{path}",
            params={"project_id": project.id, "label_taxonomy_ids": ""},
        )
        assert response.status_code == 422, path


# ---------------------------------------------------------------------------
# Wildlife summary
# ---------------------------------------------------------------------------


def test_summary_counts_every_label_by_default(db):
    project, *_ = _project(db)

    summary = stats_crud.get_dashboard_summary(db, project.id)

    assert summary.events == 2
    assert summary.observations == 6


def test_summary_follows_taxon_and_dates(db):
    project, _, _, leopard, lion, _ = _project(db)

    leopards = stats_crud.get_dashboard_summary(
        db, project.id, label_taxonomy_ids=[leopard.id]
    )
    june_lions = stats_crud.get_dashboard_summary(
        db, project.id, label_taxonomy_ids=[lion.id],
        date_from="2024-06-01", date_to="2024-06-30",
    )

    assert (leopards.events, leopards.observations) == (1, 2)
    assert (june_lions.events, june_lions.observations) == (0, 0)


def test_summary_uses_the_human_count(db):
    project, *_ = _project(db)
    lion_row = db.query(EventObservation).filter_by(label="lion").one()
    lion_row.human_count = 5
    db.flush()

    assert stats_crud.get_dashboard_summary(db, project.id).observations == 8


def test_trap_nights_are_not_floored_at_one(db):
    """A project without dated files has no effort, not one night: on the
    Overview tile (from the overview) and on Explore (from the summary)."""
    project = make_project(db)

    assert stats_crud.get_dashboard_summary(db, project.id).trap_nights == 0
    assert stats_crud.get_dashboard_overview(db, project.id).trap_nights == 0


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------


def test_demographics_split_observations_and_count_unknown(db):
    project, *_ = _project(db)

    result = stats_crud.get_demographics(db, project.id)

    assert result.observations == 6
    assert {(c.value, c.count) for c in result.sex} == {
        ("female", 3), (None, 3),
    }
    assert {(c.value, c.count) for c in result.life_stage} == {
        ("adult", 3), (None, 3),
    }
    assert {(c.value, c.count) for c in result.behavior} == {
        ("resting", 3), (None, 3),
    }


def test_demographics_count_a_human_only_cohort(db):
    """A split-off cohort has max_n 0 and only a human count."""
    project, _, dep, _, lion, _ = _project(db)
    event = db.query(EventObservation).filter_by(label="lion").one().event
    _observe(
        db, event, label="lion", taxon=lion, max_n=0,
        human_count=1, sex="male",
    )

    result = stats_crud.get_demographics(
        db, project.id, label_taxonomy_ids=[lion.id]
    )

    assert {(c.value, c.count) for c in result.sex} == {
        ("female", 3), ("male", 1),
    }


# ---------------------------------------------------------------------------
# Random animal photos
# ---------------------------------------------------------------------------


def _photo_project(db):
    project = make_project(db, counting_threshold=0.2)
    site = make_site(db, project_id=project.id, name="Ridge")
    dep = make_deployment(db, site_id=site.id)
    return project, dep


def _photo_ids(db, project, **kw):
    return {
        p.detection_id
        for p in stats_crud.get_animal_photos(db, project.id, 12, **kw)
    }


def test_photos_take_confident_named_real_detections(db):
    """Every label by default, people too; a rejected box never."""
    project, dep = _photo_project(db)
    f = make_file(db, deployment_id=dep.id)
    good = make_detection(
        db, file_id=f.id, confidence=0.9, label="lion", label_confidence=0.9
    )
    unnamed = make_detection(db, file_id=f.id, confidence=0.85)
    make_detection(db, file_id=f.id, confidence=0.7, label="lion",
                   label_confidence=0.9)  # box below the floor
    make_detection(db, file_id=f.id, confidence=0.9, label="lion",
                   label_confidence=0.3)  # name below the floor
    person = make_detection(db, file_id=f.id, category="person", confidence=0.99)
    make_detection(db, file_id=f.id, confidence=0.95, label="false detection",
                   label_confidence=0.95)
    make_detection(db, file_id=f.id, confidence=0.95, bbox_x=None)

    assert _photo_ids(db, project) == {good.id, unnamed.id, person.id}
    assert _photo_ids(db, project, wildlife_only=True) == {good.id, unnamed.id}


def test_photos_show_a_picked_person(db):
    """A pick means exactly what was picked; a rejected box never shows."""
    project, dep = _photo_project(db)
    person = _taxon(db, "person")
    f = make_file(db, deployment_id=dep.id)
    shown = make_detection(
        db, file_id=f.id, category="person", confidence=0.95,
        label_taxonomy_id=person.id,
    )
    make_detection(
        db, file_id=f.id, category="person", confidence=0.95,
        label="false detection", verified=True, label_taxonomy_id=person.id,
    )

    assert _photo_ids(db, project, label_taxonomy_ids=[person.id]) == {shown.id}


def test_a_verified_box_passes_whatever_its_scores(db):
    """A human verdict outranks the model, as everywhere else."""
    project, dep = _photo_project(db)
    f = make_file(db, deployment_id=dep.id)
    checked = make_detection(
        db, file_id=f.id, confidence=0.3, label="lion",
        label_confidence=0.2, verified=True,
    )

    assert _photo_ids(db, project) == {checked.id}


def test_a_rejected_box_never_shows(db):
    project, dep = _photo_project(db)
    f = make_file(db, deployment_id=dep.id)
    make_detection(
        db, file_id=f.id, confidence=0.95, label="false detection",
        verified=True,
    )

    assert _photo_ids(db, project) == set()


def test_videos_show_only_their_saved_best_frame(db):
    project, dep = _photo_project(db)
    saved = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        best_frame_number=4, best_frame_path="/fake/best.jpg",
    )
    on_best = make_detection(db, file_id=saved.id, frame_number=4)
    make_detection(db, file_id=saved.id, frame_number=9)
    unsaved = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        best_frame_number=2, best_frame_path=None,
    )
    make_detection(db, file_id=unsaved.id, frame_number=2)

    assert _photo_ids(db, project) == {on_best.id}


def test_photos_follow_taxon_and_limit(db):
    project, dep = _photo_project(db)
    lion = _taxon(db, "lion")
    f = make_file(db, deployment_id=dep.id)
    lions = {
        make_detection(
            db, file_id=f.id, label="lion", label_confidence=0.9,
            label_taxonomy_id=lion.id,
        ).id
        for _ in range(3)
    }
    make_detection(db, file_id=f.id, label="leopard", label_confidence=0.9)

    assert _photo_ids(db, project, label_taxonomy_ids=[lion.id]) == lions
    one = stats_crud.get_animal_photos(
        db, project.id, 1, label_taxonomy_ids=[lion.id]
    )
    assert len(one) == 1 and one[0].detection_id in lions


def test_photos_carry_site_and_camera_date(client, db):
    project, dep = _photo_project(db)
    f = make_file(
        db, deployment_id=dep.id, captured_at_local=datetime(2024, 3, 5, 23, 50)
    )
    make_detection(db, file_id=f.id, label="lion", label_confidence=0.9)

    response = client.get(
        "/api/statistics/animal-photos",
        params={"project_id": project.id, "limit": 1},
    )

    assert response.status_code == 200
    [photo] = response.json()
    assert photo["site_name"] == "Ridge"
    assert photo["captured_date"] == "2024-03-05"
    assert photo["file_id"] == f.id
