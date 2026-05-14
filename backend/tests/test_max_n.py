"""
Tests for MaxN (event observation) calculation.

MaxN is the maximum number of individuals of a species visible in any
single image within an event.
"""

import uuid
from datetime import datetime

from sqlalchemy import insert

from app.api.crud.event_observation import (
    calculate_max_n_for_event,
    get_event_ids_for_detections,
    recalculate_max_n_for_project,
)
from app.models.event import event_files
from app.models.event_observation import EventObservation
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def _make_event_with_detections(db, deployment_id, event_start_local, file_specs):
    """
    Create an event with files and detections.

    file_specs: list of dicts, each with:
        - detections: list of (label, category, confidence) tuples
    """
    eid = str(uuid.uuid4())
    from app.models.event import Event

    ev = Event(
        id=eid,
        deployment_id=deployment_id,
        event_start_local=event_start_local,
        event_end_local=event_start_local,
        file_count=len(file_specs),
    )
    db.add(ev)
    db.flush()

    created_files = []
    all_detections = []
    for seq, spec in enumerate(file_specs):
        f = make_file(
            db,
            deployment_id=deployment_id,
            captured_at_local=event_start_local,
        )
        db.execute(
            insert(event_files).values(
                event_id=eid,
                file_id=f.id,
                sequence_number=seq,
            )
        )
        created_files.append(f)
        for label, category, confidence in spec["detections"]:
            det = make_detection(
                db,
                file_id=f.id,
                category=category,
                confidence=confidence,
                label=label,
            )
            all_detections.append(det)
    db.flush()
    return ev, created_files, all_detections


def test_basic_max_n(db):
    """MaxN picks the peak count across images for a single species."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 5},   # image 1: 5 cows
        {"detections": [("cow", "animal", 0.9)] * 3},   # image 2: 3 cows
        {"detections": [("cow", "animal", 0.9)] * 7},   # image 3: 7 cows
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].label == "cow"
    assert obs[0].max_n == 7
    assert obs[0].max_n_file_id == files[2].id


def test_multi_species_max_n(db):
    """Each species gets its own MaxN within the same event."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 10 + [("bear", "animal", 0.9)] * 2},
        {"detections": [("cow", "animal", 0.9)] * 3 + [("bear", "animal", 0.9)] * 4},
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    obs_by_label = {o.label: o for o in obs}
    assert len(obs_by_label) == 2
    assert obs_by_label["cow"].max_n == 10
    assert obs_by_label["cow"].max_n_file_id == files[0].id
    assert obs_by_label["bear"].max_n == 4
    assert obs_by_label["bear"].max_n_file_id == files[1].id


def test_threshold_filtering(db):
    """Detections below threshold are excluded unless verified."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [
            ("cow", "animal", 0.9),   # above threshold
            ("cow", "animal", 0.3),   # below threshold, excluded
            ("cow", "animal", 0.2),   # below threshold, excluded
        ]},
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].max_n == 1  # only 1 passes threshold


def test_verified_below_threshold_included(db):
    """Verified detections below threshold are included in MaxN."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [
            ("cow", "animal", 0.9),
            ("cow", "animal", 0.3),
        ]},
    ])

    # Verify the low-confidence detection
    dets[1].verified = True
    db.flush()

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].max_n == 2  # both count now


def test_blank_event_no_observations(db):
    """Events with no detections get no EventObservation rows."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev = make_event_with_files(
        db,
        deployment_id=dep.id,
        event_start_local=datetime(2024, 1, 1, 12),
    )

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 0


def test_peak_file_id_correct(db):
    """max_n_file_id points to the file where MaxN was observed."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("deer", "animal", 0.9)] * 2},
        {"detections": [("deer", "animal", 0.9)] * 5},
        {"detections": [("deer", "animal", 0.9)] * 3},
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].max_n == 5
    assert obs[0].max_n_file_id == files[1].id


def test_recalculate_max_n_for_project(db):
    """Project-wide recalculation updates all events."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev1, _, _ = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 3},
    ])
    ev2, _, _ = _make_event_with_detections(db, dep.id, datetime(2024, 1, 2, 12), [
        {"detections": [("deer", "animal", 0.9)] * 2},
    ])
    db.commit()

    total = recalculate_max_n_for_project(db, project.id)
    db.commit()

    assert total == 2  # 2 observations total (cow in ev1, deer in ev2)

    # Verify stored values
    obs1 = db.query(EventObservation).filter(EventObservation.event_id == ev1.id).all()
    obs2 = db.query(EventObservation).filter(EventObservation.event_id == ev2.id).all()
    assert len(obs1) == 1
    assert obs1[0].max_n == 3
    assert len(obs2) == 1
    assert obs2[0].max_n == 2


def test_recalculation_replaces_old_values(db):
    """Recalculating replaces old EventObservation rows, not appends."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 5},
    ])

    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    # Recalculate again
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    obs = db.query(EventObservation).filter(EventObservation.event_id == ev.id).all()
    assert len(obs) == 1  # still 1, not 2
    assert obs[0].max_n == 5


def test_get_event_ids_for_detections(db):
    """Helper finds event IDs containing the given detection IDs."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)]},
    ])

    event_ids = get_event_ids_for_detections(db, [dets[0].id])
    assert ev.id in event_ids


def test_person_and_vehicle_max_n(db):
    """MaxN applies to person and vehicle categories too."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("person", "person", 0.9)] * 3 + [("vehicle", "vehicle", 0.9)] * 2},
        {"detections": [("person", "person", 0.9)] * 1 + [("vehicle", "vehicle", 0.9)] * 4},
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    obs_by_label = {o.label: o for o in obs}
    assert obs_by_label["person"].max_n == 3
    assert obs_by_label["vehicle"].max_n == 4


def test_max_n_counts_no_bbox_observations(db):
    """Event-level observations (no bbox, no bbox-anchor) still count
    toward MaxN. Two user-added observations of "deer" on the same
    video file should be MaxN=2 — they collapse into one (file, frame,
    label) group because both have frame_number=None."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12), [
            {"detections": []},  # one file, no AI detections
        ],
    )
    # Two user-added event-level observations on the same file.
    for _ in range(2):
        make_detection(
            db,
            file_id=files[0].id,
            category="animal",
            confidence=1.0,
            label="deer",
            bbox_x=None,
            bbox_y=None,
            bbox_width=None,
            bbox_height=None,
            classification_method="human",
        )
    db.flush()

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    obs_by_label = {o.label: o for o in obs}
    assert obs_by_label["deer"].max_n == 2


def test_max_n_groups_per_frame_for_videos(db):
    """Two video detections of "deer" on the same frame group into 2;
    two more on a different frame don't add up to 4. This is the core
    promise of adding `frame_number` to the GROUP BY — MaxN is the
    peak per-frame count, not the per-video total."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12), [
            {"detections": []},
        ],
    )
    for frame, count in [(10, 2), (20, 3)]:
        for _ in range(count):
            make_detection(
                db,
                file_id=files[0].id,
                category="animal",
                confidence=0.9,
                label="deer",
                bbox_x=0.1,
                bbox_y=0.1,
                bbox_width=0.2,
                bbox_height=0.2,
                frame_number=frame,
            )
    db.flush()

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    obs_by_label = {o.label: o for o in obs}
    assert obs_by_label["deer"].max_n == 3
