"""
Tests for MaxN (event observation) calculation.

MaxN is the maximum number of individuals of a species visible in any
single image within an event.
"""

import uuid
from datetime import datetime

from sqlalchemy import insert

from app.api.crud.event_observation import (
    add_human_species,
    calculate_max_n_for_event,
    get_event_ids_for_detections,
    list_event_observations,
    recalculate_max_n_for_project,
    relabel_observation,
    reset_event_to_ai,
    set_event_confirmed,
    set_human_count,
)
from app.models.event import Event, event_files
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
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
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


def test_relabel_observation_merges_into_existing(db):
    """Relabelling a row into a species already present sums the counts and
    hides the source row (count-level relabel, sum-on-merge)."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _, _ = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("bird", "animal", 0.9)] * 5 + [("deer", "animal", 0.9)]},
    ])
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    rows = {o.label: o for o in list_event_observations(db, ev.id)}
    bird_id = rows["bird"].id

    relabel_observation(db, bird_id, category="animal", label="deer")

    rows = {o.label: o for o in list_event_observations(db, ev.id)}
    # deer took bird's 5 on top of its own 1.
    assert rows["deer"].effective_count == 6
    # bird is hidden: AI boxes survive (max_n=5) but effective count is 0.
    assert rows["bird"].max_n == 5
    assert rows["bird"].effective_count == 0
    # The relabel un-signs the event.
    assert db.get(Event, ev.id).confirmed is False


def test_relabel_observation_to_new_species(db):
    """Relabelling into a species with no existing row creates a human-only
    row carrying the source count; the source row is hidden."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _, _ = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("bird", "animal", 0.9)] * 5},
    ])
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    bird_id = list_event_observations(db, ev.id)[0].id
    relabel_observation(db, bird_id, category="animal", label="chicken")

    rows = {o.label: o for o in list_event_observations(db, ev.id)}
    assert rows["chicken"].max_n == 0
    assert rows["chicken"].effective_count == 5
    assert rows["bird"].effective_count == 0


def test_threshold_filtering(db):
    """Detections below threshold are excluded unless verified."""
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_event_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)]},
    ])

    event_ids = get_event_ids_for_detections(db, [dets[0].id])
    assert ev.id in event_ids


def test_person_and_vehicle_max_n(db):
    """MaxN applies to person and vehicle categories too."""
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
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
    project = make_project(db, counting_threshold=0.5)
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


# ── Human count layer + event sign-off ─────────────────────────────


def test_human_count_overrides_and_survives_recompute(db):
    """A human count overrides the AI MaxN for the effective count and
    is preserved when MaxN is later recomputed."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, _ = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 2}],
    )
    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()
    assert obs[0].max_n == 2
    assert obs[0].effective_count == 2

    # Saw 3 more the AI missed across other frames: bump to 5.
    updated = set_human_count(db, obs[0].id, 5)
    assert updated.human_count == 5
    assert updated.effective_count == 5

    # A later recompute keeps the human count, not just the AI MaxN.
    recalc = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()
    assert len(recalc) == 1
    assert recalc[0].max_n == 2
    assert recalc[0].human_count == 5
    assert recalc[0].effective_count == 5


def test_add_human_species_creates_human_only_row_and_survives(db):
    """A species the AI never detected is stored as a human-only row
    (max_n=0, no frame) and survives a recompute."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, _ = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)]}],
    )
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    obs = add_human_species(db, ev.id, category="animal", count=1, label="fox")
    assert obs.max_n == 0
    assert obs.human_count == 1
    assert obs.max_n_file_id is None

    recalc = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()
    by_label = {o.label: o for o in recalc}
    assert "cow" in by_label
    assert "fox" in by_label
    assert by_label["fox"].max_n == 0
    assert by_label["fox"].human_count == 1


def test_recompute_clears_event_verified_when_counts_change(db):
    """The event sign-off survives a no-op recompute but clears when the
    species/count set actually changes."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, dets = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 3}],
    )
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    set_event_confirmed(db, ev.id, True)
    assert db.get(Event, ev.id).confirmed is True

    # No-op recompute (nothing changed) keeps the sign-off.
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()
    assert db.get(Event, ev.id).confirmed is True

    # Removing a detection lowers MaxN 3 -> 2; the sign-off clears.
    db.delete(dets[0])
    db.flush()
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()
    assert db.get(Event, ev.id).confirmed is False


def test_dashboard_total_observations_uses_effective_count(db):
    """The dashboard observation total sums the effective count
    (human override, else MaxN), not the raw AI MaxN."""
    from app.api.crud.statistics import get_dashboard_overview

    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev, _files, _ = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 2}],
    )
    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.commit()

    assert get_dashboard_overview(db, project.id).total_observations == 2

    set_human_count(db, obs[0].id, 5)
    db.commit()
    assert get_dashboard_overview(db, project.id).total_observations == 5


def test_remove_ai_species_via_zero_count_survives_recompute(db):
    """Removing an AI species in the Counts panel sets human_count=0, which
    survives a recompute (the durable representation of 'not present')."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, _ = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 2}],
    )
    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    # "Remove" the species: the human says none are actually present.
    set_human_count(db, obs[0].id, 0)

    recalc = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()
    assert len(recalc) == 1
    assert recalc[0].max_n == 2  # the boxes are still there
    assert recalc[0].human_count == 0  # but the human override survives
    assert recalc[0].effective_count == 0


def test_reset_event_to_ai_drops_human_edits(db):
    """reset_event_to_ai clears overrides on AI rows, deletes human-only
    rows, and clears the event sign-off."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, _ = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 2}],
    )
    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()
    set_human_count(db, obs[0].id, 9)  # override the AI row
    add_human_species(db, ev.id, category="animal", count=1, label="fox")
    set_event_confirmed(db, ev.id, True)

    event = reset_event_to_ai(db, ev.id)
    assert event is not None
    assert event.confirmed is False

    rows = (
        db.query(EventObservation)
        .filter(EventObservation.event_id == ev.id)
        .all()
    )
    assert len(rows) == 1  # the human-only fox row is gone
    assert rows[0].label == "cow"
    assert rows[0].human_count is None  # override cleared
    assert rows[0].effective_count == 2  # back to the AI MaxN


def test_list_event_observations_order_is_stable_under_count_edits(db):
    """Row order follows AI MaxN (then label), never the editable count, so
    bumping a count does not reshuffle the list under the user."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, _ = _make_event_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{
            "detections": [("cow", "animal", 0.9)] * 3
            + [("fox", "animal", 0.9)],
        }],
    )
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    before = [o.label for o in list_event_observations(db, ev.id)]
    assert before == ["cow", "fox"]  # cow (max_n 3) before fox (max_n 1)

    # Bump fox far above cow; order must NOT change.
    fox = next(o for o in list_event_observations(db, ev.id) if o.label == "fox")
    set_human_count(db, fox.id, 50)
    after = [o.label for o in list_event_observations(db, ev.id)]
    assert after == ["cow", "fox"]


# ── Video best-frame species gate ───────────────────────────────────────
#
# For videos, only species present on the best frame (or verified on some
# frame) may produce a count row; non-best-frame labels are per-frame
# classifier noise the user can never see or clean, so they must not spawn
# spurious species. Images are never gated.


def _make_event_with_frames(db, deployment_id, event_start_local, file_specs):
    """Create an event whose files can be videos with per-frame detections.

    file_specs: list of dicts:
        - file_type: "image" | "video" (default "image")
        - best_frame_number: int | None (videos)
        - detections: list of dicts with keys
            label, frame_number (opt), category (opt "animal"),
            confidence (opt 0.9), verified (opt False)
    Returns (event, files).
    """
    eid = str(uuid.uuid4())
    ev = Event(
        id=eid,
        deployment_id=deployment_id,
        event_start_local=event_start_local,
        event_end_local=event_start_local,
        file_count=len(file_specs),
    )
    db.add(ev)
    db.flush()

    files = []
    for seq, spec in enumerate(file_specs):
        fkw = {"file_type": spec.get("file_type", "image")}
        if "best_frame_number" in spec:
            fkw["best_frame_number"] = spec["best_frame_number"]
        f = make_file(
            db,
            deployment_id=deployment_id,
            captured_at_local=event_start_local,
            **fkw,
        )
        db.execute(
            insert(event_files).values(
                event_id=eid,
                file_id=f.id,
                sequence_number=seq,
            )
        )
        files.append(f)
        for d in spec["detections"]:
            make_detection(
                db,
                file_id=f.id,
                category=d.get("category", "animal"),
                confidence=d.get("confidence", 0.9),
                label=d["label"],
                frame_number=d.get("frame_number"),
                verified=d.get("verified", False),
            )
    db.flush()
    return ev, files


def test_video_non_best_frame_label_is_gated_out(db):
    """A label that only appears on a non-best frame is dropped."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_event_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 5,
            "detections": [
                {"label": "leopard", "frame_number": 5},     # best frame
                {"label": "carnivora", "frame_number": 12},  # non-best noise
            ],
        },
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert {o.label for o in obs} == {"leopard"}  # carnivora gated out


def test_video_best_frame_species_counts_peak_across_frames(db):
    """An allowed species still takes its peak count across all frames."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_event_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 5,
            "detections": [
                {"label": "leopard", "frame_number": 5},   # best frame: 1
                {"label": "leopard", "frame_number": 40},  # non-best: 3
                {"label": "leopard", "frame_number": 40},
                {"label": "leopard", "frame_number": 40},
            ],
        },
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].label == "leopard"
    assert obs[0].max_n == 3  # allowed via best frame, peak across all frames


def test_video_verified_non_best_frame_species_survives(db):
    """A human-verified species survives even off the best frame."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_event_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 5,
            "detections": [
                {"label": "leopard", "frame_number": 5},
                # not on best frame, but the human confirmed it:
                {"label": "serval", "frame_number": 12, "verified": True},
            ],
        },
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert {o.label for o in obs} == {"leopard", "serval"}


def test_image_multispecies_is_not_gated(db):
    """Images are never gated: every species in a photo is kept."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_event_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "image",
            "detections": [
                {"label": "cow"},
                {"label": "bear"},
            ],
        },
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    assert {o.label for o in obs} == {"cow", "bear"}


def test_multi_video_event_gates_per_file(db):
    """The gate is per video: a species allowed in one video is not
    rescued for another video where it's only a non-best-frame label."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_event_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {  # video A: leopard on its best frame
            "file_type": "video",
            "best_frame_number": 0,
            "detections": [{"label": "leopard", "frame_number": 0}],
        },
        {  # video B: cow on best frame, leopard only on a non-best frame
            "file_type": "video",
            "best_frame_number": 0,
            "detections": [
                {"label": "cow", "frame_number": 0},
                {"label": "leopard", "frame_number": 7},  # gated for B
            ],
        },
    ])

    obs = calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    by_label = {o.label: o.max_n for o in obs}
    assert set(by_label) == {"leopard", "cow"}
    # leopard counted only from video A (1); video B's frame-7 leopard gated.
    assert by_label["leopard"] == 1
    assert by_label["cow"] == 1


def test_event_card_chips_match_best_frame_gate(db):
    """Gallery-card chips exclude non-best-frame video noise, same as the
    count suggestion (the chips are built from raw detections, so they must
    be gated against the event's EventObservation rows)."""
    from app.api.crud.event import get_events_by_project

    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_event_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 5,
            "detections": [
                {"label": "leopard", "frame_number": 5},     # best frame
                {"label": "carnivora", "frame_number": 12},  # non-best noise
            ],
        },
    ])

    # Populate EventObservation (the gate source), as analysis would.
    calculate_max_n_for_event(db, ev.id, 0.5)
    db.flush()

    summaries = get_events_by_project(db, project.id, project_floor=0.5)
    assert len(summaries) == 1
    assert summaries[0]["labels"] == ["leopard"]  # carnivora chip gated out


# ---------------------------------------------------------------------------
# Preservation of the human layer across event regeneration (interval change)
# ---------------------------------------------------------------------------


def _file_with_dets(db, deployment_id, ts, dets):
    """One file at time `ts` with `dets` = [(label, category, confidence)]."""
    f = make_file(db, deployment_id=deployment_id, captured_at_local=ts)
    for label, category, confidence in dets:
        make_detection(
            db, file_id=f.id, category=category, confidence=confidence, label=label
        )
    return f


def _events(db, deployment_id):
    return (
        db.query(Event)
        .filter(Event.deployment_id == deployment_id)
        .order_by(Event.event_start_local)
        .all()
    )


def test_generate_preserves_confirmed_and_human_count_when_unchanged(db):
    """A reprocess that doesn't regroup an event (same interval) keeps its
    confirmation and manual count."""
    from app.api.crud.event import generate_events_for_project

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    dep = make_deployment(db, project_id=project.id)
    # A close cow pair (one event) and a far-off bear (a second event).
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 0, 0), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 0, 30), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 10, 0), [("bear", "animal", 0.9)])
    db.commit()

    generate_events_for_project(db, project.id)
    cow_event = _events(db, dep.id)[0]
    cow_obs = (
        db.query(EventObservation)
        .filter_by(event_id=cow_event.id, label="cow")
        .one()
    )
    cow_obs.human_count = 5
    cow_event.confirmed = True
    db.commit()

    # Regenerate at the same interval: grouping is identical.
    generate_events_for_project(db, project.id)

    cow_event2 = _events(db, dep.id)[0]
    assert cow_event2.confirmed is True
    cow_obs2 = (
        db.query(EventObservation)
        .filter_by(event_id=cow_event2.id, label="cow")
        .one()
    )
    assert cow_obs2.human_count == 5
    assert cow_obs2.effective_count == 5


def test_generate_interval_change_resets_merged_but_keeps_unaffected(db):
    """Widening the interval merges some events (reset) but leaves others
    untouched (preserved)."""
    from app.api.crud.event import generate_events_for_project

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    # dep1: cow pair + far bear -> merges into one at a wide interval.
    dep1 = make_deployment(db, project_id=project.id)
    _file_with_dets(db, dep1.id, datetime(2024, 1, 1, 12, 0, 0), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep1.id, datetime(2024, 1, 1, 12, 0, 30), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep1.id, datetime(2024, 1, 1, 12, 10, 0), [("bear", "animal", 0.9)])
    # dep2: a single deer file -> always its own event, unaffected.
    dep2 = make_deployment(db, project_id=project.id)
    _file_with_dets(db, dep2.id, datetime(2024, 1, 1, 12, 0, 0), [("deer", "animal", 0.9)])
    db.commit()

    generate_events_for_project(db, project.id)
    cow_event = _events(db, dep1.id)[0]
    deer_event = _events(db, dep2.id)[0]
    db.query(EventObservation).filter_by(event_id=cow_event.id, label="cow").one().human_count = 4
    db.query(EventObservation).filter_by(event_id=deer_event.id, label="deer").one().human_count = 2
    cow_event.confirmed = True
    deer_event.confirmed = True
    db.commit()

    # Widen the interval so dep1's two events merge.
    project.independence_interval = 3600
    db.commit()
    generate_events_for_project(db, project.id)

    dep1_events = _events(db, dep1.id)
    assert len(dep1_events) == 1  # merged
    merged = dep1_events[0]
    assert merged.confirmed is False  # regrouped -> reset
    cow_after = (
        db.query(EventObservation).filter_by(event_id=merged.id, label="cow").one()
    )
    assert cow_after.human_count is None  # reverted to AI count

    deer_after_event = _events(db, dep2.id)[0]
    assert deer_after_event.confirmed is True  # unchanged grouping -> preserved
    deer_after = (
        db.query(EventObservation)
        .filter_by(event_id=deer_after_event.id, label="deer")
        .one()
    )
    assert deer_after.human_count == 2


def test_generate_relabel_unconfirms_preserved_event(db):
    """Same interval, but a label change inside a preserved event (e.g.
    smoothing) still clears the confirmation, per the existing rule."""
    from app.api.crud.event import generate_events_for_project
    from app.models import Detection

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    dep = make_deployment(db, project_id=project.id)
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 0, 0), [("cow", "animal", 0.9)])
    db.commit()

    generate_events_for_project(db, project.id)
    event = _events(db, dep.id)[0]
    event.confirmed = True
    db.commit()

    # Relabel the detection (grouping unchanged, species set changes).
    db.query(Detection).update({"label": "elk"})
    db.commit()
    generate_events_for_project(db, project.id)

    event2 = _events(db, dep.id)[0]
    assert event2.confirmed is False


def test_count_regroup_impact(db):
    """Preview counts confirmed events / manual counts a regroup would reset,
    and is zero when the interval is unchanged."""
    from app.api.crud.event import count_regroup_impact, generate_events_for_project

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    dep1 = make_deployment(db, project_id=project.id)
    _file_with_dets(db, dep1.id, datetime(2024, 1, 1, 12, 0, 0), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep1.id, datetime(2024, 1, 1, 12, 0, 30), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep1.id, datetime(2024, 1, 1, 12, 10, 0), [("bear", "animal", 0.9)])
    dep2 = make_deployment(db, project_id=project.id)
    _file_with_dets(db, dep2.id, datetime(2024, 1, 1, 12, 0, 0), [("deer", "animal", 0.9)])
    db.commit()

    generate_events_for_project(db, project.id)
    cow_event = _events(db, dep1.id)[0]
    deer_event = _events(db, dep2.id)[0]
    db.query(EventObservation).filter_by(event_id=cow_event.id, label="cow").one().human_count = 4
    cow_event.confirmed = True
    deer_event.confirmed = True
    db.commit()

    # Widening merges dep1's cow event; dep2's deer event is unaffected.
    merging = count_regroup_impact(db, project.id, 3600)
    assert merging["total_confirmed"] == 2
    assert merging["confirmed_at_risk"] == 1
    assert merging["counts_at_risk"] == 1
    # The example points at the regrouped cow event: its confirmed count,
    # its time range, and that its files land in one merged event.
    example = merging["example"]
    assert example is not None
    assert example["observations"] == [{"label": "cow", "count": 4}]
    assert example["maps_to"] == 1
    assert example["time_range"] is not None

    # Same interval: nothing at risk, no example.
    unchanged = count_regroup_impact(db, project.id, 300)
    assert unchanged["confirmed_at_risk"] == 0
    assert unchanged["counts_at_risk"] == 0
    assert unchanged["example"] is None
