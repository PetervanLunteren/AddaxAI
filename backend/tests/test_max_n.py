"""
Tests for MaxN (sequence observation) calculation.

MaxN is the maximum number of individuals of a species visible in any
single image within a sequence.
"""

import uuid
from datetime import datetime, timedelta

from sqlalchemy import insert

from app.api.crud.sequence_observation import (
    add_human_species,
    calculate_max_n_for_sequence,
    delete_sequence_observation,
    get_sequence_ids_for_detections,
    list_sequence_observations,
    recalculate_max_n_for_project,
    relabel_observation,
    reset_sequence_to_ai,
    set_human_count,
    set_human_observations,
    set_observation_attributes,
    set_sequence_confirmed,
    set_sequence_notes,
    split_observation,
)
from app.models.sequence import Sequence, sequence_files
from app.models.sequence_observation import SequenceObservation
from app.models.track import Track
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_sequence_with_files,
    make_site,
    make_track,
)


def _make_sequence_with_detections(db, deployment_id, sequence_start_local, file_specs):
    """
    Create a sequence with files and detections.

    file_specs: list of dicts, each with:
        - detections: list of (label, category, confidence) tuples
    """
    eid = str(uuid.uuid4())
    from app.models.sequence import Sequence

    ev = Sequence(
        camera_key="",
        id=eid,
        deployment_id=deployment_id,
        sequence_start_local=sequence_start_local,
        sequence_end_local=sequence_start_local,
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
            captured_at_local=sequence_start_local,
        )
        db.execute(
            insert(sequence_files).values(
                sequence_id=eid,
                file_id=f.id,
                position=seq,
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

    ev, files, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 5},   # image 1: 5 cows
        {"detections": [("cow", "animal", 0.9)] * 3},   # image 2: 3 cows
        {"detections": [("cow", "animal", 0.9)] * 7},   # image 3: 7 cows
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].label == "cow"
    assert obs[0].max_n == 7
    assert obs[0].max_n_file_id == files[2].id


def test_multi_species_max_n(db):
    """Each species gets its own MaxN within the same sequence."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 10 + [("bear", "animal", 0.9)] * 2},
        {"detections": [("cow", "animal", 0.9)] * 3 + [("bear", "animal", 0.9)] * 4},
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
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

    ev, _, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("bird", "animal", 0.9)] * 5 + [("deer", "animal", 0.9)]},
    ])
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    rows = {o.label: o for o in list_sequence_observations(db, ev.id)}
    bird_id = rows["bird"].id

    relabel_observation(db, bird_id, category="animal", label="deer")

    rows = {o.label: o for o in list_sequence_observations(db, ev.id)}
    # deer took bird's 5 on top of its own 1.
    assert rows["deer"].effective_count == 6
    # bird is hidden: AI boxes survive (max_n=5) but effective count is 0.
    assert rows["bird"].max_n == 5
    assert rows["bird"].effective_count == 0
    # The relabel un-signs the sequence.
    assert db.get(Sequence, ev.id).confirmed is False


def test_relabel_observation_to_new_species(db):
    """Relabelling into a species with no existing row creates a human-only
    row carrying the source count; the source row is hidden."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("bird", "animal", 0.9)] * 5},
    ])
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    bird_id = list_sequence_observations(db, ev.id)[0].id
    relabel_observation(db, bird_id, category="animal", label="chicken")

    rows = {o.label: o for o in list_sequence_observations(db, ev.id)}
    assert rows["chicken"].max_n == 0
    assert rows["chicken"].effective_count == 5
    assert rows["bird"].effective_count == 0


def test_threshold_filtering(db):
    """Detections below threshold are excluded unless verified."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [
            ("cow", "animal", 0.9),   # above threshold
            ("cow", "animal", 0.3),   # below threshold, excluded
            ("cow", "animal", 0.2),   # below threshold, excluded
        ]},
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].max_n == 1  # only 1 passes threshold


def test_verified_below_threshold_included(db):
    """Verified detections below threshold are included in MaxN."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [
            ("cow", "animal", 0.9),
            ("cow", "animal", 0.3),
        ]},
    ])

    # Verify the low-confidence detection
    dets[1].verified = True
    db.flush()

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].max_n == 2  # both count now


def test_blank_sequence_no_observations(db):
    """Sequences with no detections get no SequenceObservation rows."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev = make_sequence_with_files(
        db,
        deployment_id=dep.id,
        sequence_start_local=datetime(2024, 1, 1, 12),
    )

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 0


def test_peak_file_id_correct(db):
    """max_n_file_id points to the file where MaxN was observed."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("deer", "animal", 0.9)] * 2},
        {"detections": [("deer", "animal", 0.9)] * 5},
        {"detections": [("deer", "animal", 0.9)] * 3},
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].max_n == 5
    assert obs[0].max_n_file_id == files[1].id


def test_recalculate_max_n_for_project(db):
    """Project-wide recalculation updates all sequences."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev1, _, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 3},
    ])
    ev2, _, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 2, 12), [
        {"detections": [("deer", "animal", 0.9)] * 2},
    ])
    db.commit()

    total = recalculate_max_n_for_project(db, project.id)
    db.commit()

    assert total == 2  # 2 observations total (cow in ev1, deer in ev2)

    # Verify stored values
    obs1 = db.query(SequenceObservation).filter(SequenceObservation.sequence_id == ev1.id).all()
    obs2 = db.query(SequenceObservation).filter(SequenceObservation.sequence_id == ev2.id).all()
    assert len(obs1) == 1
    assert obs1[0].max_n == 3
    assert len(obs2) == 1
    assert obs2[0].max_n == 2


def test_recalculation_replaces_old_values(db):
    """Recalculating replaces old SequenceObservation rows, not appends."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 5},
    ])

    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    # Recalculate again
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    obs = db.query(SequenceObservation).filter(SequenceObservation.sequence_id == ev.id).all()
    assert len(obs) == 1  # still 1, not 2
    assert obs[0].max_n == 5


def test_recalculation_drops_only_this_sequences_rows_from_the_session(db):
    """The old rows leave the session, another sequence's rows stay in it.

    The delete in `calculate_max_n_for_sequence` synchronizes the session by
    the ids SQLite returns ("fetch"), not by testing every loaded object.
    This pins that the result is still right: a stale row left in the
    session would be read back as a live observation.
    """
    from sqlalchemy import inspect

    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev, _, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)] * 2},
    ])
    other, _, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 2, 12), [
        {"detections": [("bear", "animal", 0.9)]},
    ])
    [old] = calculate_max_n_for_sequence(db, ev.id, 0.5)
    [kept] = calculate_max_n_for_sequence(db, other.id, 0.5)
    db.flush()

    [new] = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert inspect(old).was_deleted
    assert old not in db
    assert kept in db and not inspect(kept).was_deleted
    assert new.id != old.id and new.max_n == 2


def test_get_sequence_ids_for_detections(db):
    """Helper finds sequence IDs containing the given detection IDs."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, dets = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("cow", "animal", 0.9)]},
    ])

    sequence_ids = get_sequence_ids_for_detections(db, [dets[0].id])
    assert ev.id in sequence_ids


def test_person_and_vehicle_max_n(db):
    """MaxN applies to person and vehicle categories too."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_sequence_with_detections(db, dep.id, datetime(2024, 1, 1, 12), [
        {"detections": [("person", "person", 0.9)] * 3 + [("vehicle", "vehicle", 0.9)] * 2},
        {"detections": [("person", "person", 0.9)] * 1 + [("vehicle", "vehicle", 0.9)] * 4},
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    obs_by_label = {o.label: o for o in obs}
    assert obs_by_label["person"].max_n == 3
    assert obs_by_label["vehicle"].max_n == 4


def test_max_n_counts_no_bbox_observations(db):
    """Box-less observations (no bbox, no bbox-anchor) still count
    toward MaxN. Two user-added observations of "deer" on the same
    video file should be MaxN=2 — they collapse into one (file, frame,
    label) group because both have frame_number=None."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12), [
            {"detections": []},  # one file, no AI detections
        ],
    )
    # Two user-added box-less observations on the same file.
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

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
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

    ev, files, _ = _make_sequence_with_detections(
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

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    obs_by_label = {o.label: o for o in obs}
    assert obs_by_label["deer"].max_n == 3


def test_a_track_voted_to_a_rejection_counts_nothing(db):
    """The other half of the vote rule: a track whose every box carries a
    rejection label (the vote gave it "false detection") adds no species
    to the sequence, while the fox track beside it still counts."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev, files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12), [{"detections": []}],
    )
    clip = files[0]
    clip.file_type = "video"
    for key, label in ((1, "fox"), (2, "false detection")):
        track = make_track(db, file_id=clip.id, track_key=key, start_frame=10,
                           end_frame=20, representative_frame_number=10)
        for frame in (10, 20):
            make_detection(
                db, file_id=clip.id, category="animal", confidence=0.9, label=label,
                bbox_x=0.1, bbox_y=0.1, bbox_width=0.2, bbox_height=0.2,
                frame_number=frame, track_id=track.id,
            )
    db.flush()

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)

    assert {o.label: o.max_n for o in obs} == {"fox": 1}


# ── Human count layer + sequence sign-off ─────────────────────────────


def test_human_count_overrides_and_survives_recompute(db):
    """A human count overrides the AI MaxN for the effective count and
    is preserved when MaxN is later recomputed."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 2}],
    )
    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert obs[0].max_n == 2
    assert obs[0].effective_count == 2

    # Saw 3 more the AI missed across other frames: bump to 5.
    updated = set_human_count(db, obs[0].id, 5)
    assert updated.human_count == 5
    assert updated.effective_count == 5

    # A later recompute keeps the human count, not just the AI MaxN.
    recalc = calculate_max_n_for_sequence(db, ev.id, 0.5)
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

    ev, _files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)]}],
    )
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    obs = add_human_species(db, ev.id, category="animal", count=1, label="fox")
    assert obs.max_n == 0
    assert obs.human_count == 1
    assert obs.max_n_file_id is None

    recalc = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    by_label = {o.label: o for o in recalc}
    assert "cow" in by_label
    assert "fox" in by_label
    assert by_label["fox"].max_n == 0
    assert by_label["fox"].human_count == 1


def test_recompute_clears_sequence_verified_when_counts_change(db):
    """The sequence sign-off survives a no-op recompute but clears when the
    species/count set actually changes."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, dets = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 3}],
    )
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    set_sequence_confirmed(db, ev.id, True)
    assert db.get(Sequence, ev.id).confirmed is True

    # No-op recompute (nothing changed) keeps the sign-off.
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert db.get(Sequence, ev.id).confirmed is True

    # Removing a detection lowers MaxN 3 -> 2; the sign-off clears.
    db.delete(dets[0])
    db.flush()
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert db.get(Sequence, ev.id).confirmed is False


def test_dashboard_total_observations_uses_effective_count(db):
    """The dashboard observation total sums the effective count
    (human override, else MaxN), not the raw AI MaxN."""
    from app.api.crud.statistics import get_dashboard_overview

    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev, _files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 2}],
    )
    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
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

    ev, _files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 2}],
    )
    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    # "Remove" the species: the human says none are actually present.
    set_human_count(db, obs[0].id, 0)

    recalc = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert len(recalc) == 1
    assert recalc[0].max_n == 2  # the boxes are still there
    assert recalc[0].human_count == 0  # but the human override survives
    assert recalc[0].effective_count == 0


def test_reset_sequence_to_ai_drops_human_edits(db):
    """reset_sequence_to_ai clears overrides on AI rows, deletes human-only
    rows, and clears the sequence sign-off."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 2}],
    )
    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    set_human_count(db, obs[0].id, 9)  # override the AI row
    add_human_species(db, ev.id, category="animal", count=1, label="fox")
    set_sequence_confirmed(db, ev.id, True)

    sequence = reset_sequence_to_ai(db, ev.id)
    assert sequence is not None
    assert sequence.confirmed is False

    rows = (
        db.query(SequenceObservation)
        .filter(SequenceObservation.sequence_id == ev.id)
        .all()
    )
    assert len(rows) == 1  # the human-only fox row is gone
    assert rows[0].label == "cow"
    assert rows[0].human_count is None  # override cleared
    assert rows[0].effective_count == 2  # back to the AI MaxN


def test_list_sequence_observations_order_is_stable_under_count_edits(db):
    """Row order follows AI MaxN (then label), never the editable count, so
    bumping a count does not reshuffle the list under the user."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{
            "detections": [("cow", "animal", 0.9)] * 3
            + [("fox", "animal", 0.9)],
        }],
    )
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    before = [o.label for o in list_sequence_observations(db, ev.id)]
    assert before == ["cow", "fox"]  # cow (max_n 3) before fox (max_n 1)

    # Bump fox far above cow; order must NOT change.
    fox = next(o for o in list_sequence_observations(db, ev.id) if o.label == "fox")
    set_human_count(db, fox.id, 50)
    after = [o.label for o in list_sequence_observations(db, ev.id)]
    assert after == ["cow", "fox"]


# ── Video species gate: one card per track ──────────────────────────────
#
# Every video box belongs to a track, and a track has one card: its box on
# the representative frame. A video species may produce a count row only
# when one of its boxes is a card; the other per-frame labels are
# classifier noise the user can never see or clean, so they must not
# spawn spurious species. Once admitted, a species takes its peak count
# across every frame. Images are never gated.


def _make_sequence_with_frames(db, deployment_id, sequence_start_local, file_specs):
    """Create a sequence whose files can be videos with per-frame detections.

    file_specs: list of dicts:
        - file_type: "image" | "video" (default "image")
        - best_frame_number: int | None (videos, the cover picture only)
        - detections: list of dicts with keys
            label, frame_number (opt), category (opt "animal"),
            confidence (opt 0.9), verified (opt False),
            track (opt): boxes of one video sharing a track name share
                one track, whose card is on the lowest of their frames;
                a video box without it gets a one-frame track of its
                own, so it is a card,
            untracked (opt False): no track at all, so never a card.
    Returns (sequence, files).
    """
    eid = str(uuid.uuid4())
    ev = Sequence(
        camera_key="",
        id=eid,
        deployment_id=deployment_id,
        sequence_start_local=sequence_start_local,
        sequence_end_local=sequence_start_local,
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
            captured_at_local=sequence_start_local,
            **fkw,
        )
        db.execute(
            insert(sequence_files).values(
                sequence_id=eid,
                file_id=f.id,
                position=seq,
            )
        )
        files.append(f)
        is_video = fkw["file_type"] == "video"
        shared: dict[str, Track] = {}
        next_key = 1
        for d in spec["detections"]:
            track_id = None
            if is_video and not d.get("untracked", False):
                name = d.get("track")
                if name is not None and name in shared:
                    track_id = shared[name].id
                else:
                    frames = [
                        x["frame_number"]
                        for x in spec["detections"]
                        if name is not None and x.get("track") == name
                    ] or [d["frame_number"]]
                    track = make_track(
                        db,
                        file_id=f.id,
                        track_key=next_key,
                        start_frame=min(frames),
                        end_frame=max(frames),
                        frame_count=len(set(frames)),
                        representative_frame_number=min(frames),
                        max_confidence=d.get("confidence", 0.9),
                        crop_path=None,
                    )
                    next_key += 1
                    if name is not None:
                        shared[name] = track
                    track_id = track.id
            make_detection(
                db,
                file_id=f.id,
                category=d.get("category", "animal"),
                confidence=d.get("confidence", 0.9),
                label=d["label"],
                frame_number=d.get("frame_number"),
                verified=d.get("verified", False),
                track_id=track_id,
            )
    db.flush()
    return ev, files


def test_video_species_without_a_card_spawns_no_row(db):
    """A species whose boxes are never a card is dropped: a per-frame
    label on another frame of a track, and an untracked box, even one on
    the cover frame."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_sequence_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 12,
            "detections": [
                {"label": "leopard", "frame_number": 5, "track": "a"},    # card
                {"label": "carnivora", "frame_number": 12, "track": "a"}, # same track, cover
                {"label": "serval", "frame_number": 12, "untracked": True},
            ],
        },
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert {o.label for o in obs} == {"leopard"}


def test_admitted_video_species_counts_peak_across_frames(db):
    """An admitted species takes its peak count across every frame, from
    every box it has: the card, the same track's other boxes, and boxes
    with no card of their own."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_sequence_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 5,
            "detections": [
                {"label": "leopard", "frame_number": 5, "track": "a"},   # card: 1
                {"label": "leopard", "frame_number": 40, "track": "a"},  # frame 40: 3
                {"label": "leopard", "frame_number": 40, "untracked": True},
                {"label": "leopard", "frame_number": 40, "untracked": True},
            ],
        },
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].label == "leopard"
    assert obs[0].max_n == 3
    assert obs[0].max_n_frame_number == 40


def test_equal_counts_keep_the_first_frame(db):
    """Two frames with the same count: MaxN is taken at its first
    occurrence, so the earlier frame is the one recorded."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_sequence_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 0,
            "detections": [
                {"label": "leopard", "frame_number": 10, "track": "a"},
                {"label": "leopard", "frame_number": 10, "track": "b"},
                {"label": "leopard", "frame_number": 20, "track": "a"},
                {"label": "leopard", "frame_number": 20, "track": "b"},
            ],
        },
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert len(obs) == 1
    assert obs[0].max_n == 2
    assert obs[0].max_n_frame_number == 10


def test_video_verified_box_off_its_card_does_not_admit_a_species(db):
    """A verdict is given on the card and reaches the whole track, so a
    verified box on another frame is not a card and admits nothing: not
    when it sits on a track, not when it is untracked."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_sequence_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 5,
            "detections": [
                {"label": "leopard", "frame_number": 5, "track": "a"},
                {"label": "serval", "frame_number": 12, "track": "a", "verified": True},
                {"label": "caracal", "frame_number": 12, "untracked": True, "verified": True},
            ],
        },
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert {o.label for o in obs} == {"leopard"}


def test_image_multispecies_is_not_gated(db):
    """Images are never gated: every species in a photo is kept."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_sequence_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "image",
            "detections": [
                {"label": "cow"},
                {"label": "bear"},
            ],
        },
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert {o.label for o in obs} == {"cow", "bear"}


def test_multi_video_sequence_gates_per_file(db):
    """The gate is per video: a species with a card in one video is not
    rescued for another video where it has no card."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_sequence_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {  # video A: leopard is a card
            "file_type": "video",
            "best_frame_number": 0,
            "detections": [{"label": "leopard", "frame_number": 0}],
        },
        {  # video B: cow is a card, leopard is the cow track's later label
            "file_type": "video",
            "best_frame_number": 0,
            "detections": [
                {"label": "cow", "frame_number": 0, "track": "a"},
                {"label": "leopard", "frame_number": 7, "track": "a"},  # gated for B
            ],
        },
    ])

    obs = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    by_label = {o.label: o.max_n for o in obs}
    assert set(by_label) == {"leopard", "cow"}
    # leopard counted only from video A (1); video B's frame-7 leopard gated.
    assert by_label["leopard"] == 1
    assert by_label["cow"] == 1


def test_sequence_card_chips_match_the_card_gate(db):
    """Gallery-card chips exclude video labels that have no card, same as
    the count suggestion (the chips are built from raw detections, so they
    must be gated against the sequence's SequenceObservation rows)."""
    from app.api.crud.sequence import get_sequences_by_project

    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, _ = _make_sequence_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 5,
            "detections": [
                {"label": "leopard", "frame_number": 5, "track": "a"},    # card
                {"label": "carnivora", "frame_number": 12, "track": "a"}, # noise
            ],
        },
    ])

    # Populate SequenceObservation (the gate source), as analysis would.
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    summaries = get_sequences_by_project(db, project.id, project_floor=0.5)
    assert len(summaries) == 1
    assert summaries[0]["labels"] == ["leopard"]  # carnivora chip gated out


# ---------------------------------------------------------------------------
# Preservation of the human layer across sequence regeneration (interval change)
# ---------------------------------------------------------------------------


def _file_with_dets(db, deployment_id, ts, dets):
    """One file at time `ts` with `dets` = [(label, category, confidence)]."""
    f = make_file(db, deployment_id=deployment_id, captured_at_local=ts)
    for label, category, confidence in dets:
        make_detection(
            db, file_id=f.id, category=category, confidence=confidence, label=label
        )
    return f


def _sequences(db, deployment_id):
    return (
        db.query(Sequence)
        .filter(Sequence.deployment_id == deployment_id)
        .order_by(Sequence.sequence_start_local)
        .all()
    )


def test_generate_preserves_confirmed_and_human_count_when_unchanged(db):
    """A reprocess that doesn't regroup a sequence (same interval) keeps its
    confirmation and manual count."""
    from app.api.crud.sequence import generate_sequences_for_project

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    dep = make_deployment(db, project_id=project.id)
    # A close cow pair (one sequence) and a far-off bear (a second sequence).
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 0, 0), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 0, 30), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 10, 0), [("bear", "animal", 0.9)])
    db.commit()

    generate_sequences_for_project(db, project.id)
    cow_sequence = _sequences(db, dep.id)[0]
    cow_obs = (
        db.query(SequenceObservation)
        .filter_by(sequence_id=cow_sequence.id, label="cow")
        .one()
    )
    cow_obs.human_count = 5
    cow_sequence.confirmed = True
    db.commit()

    # Regenerate at the same interval: grouping is identical.
    generate_sequences_for_project(db, project.id)

    cow_event2 = _sequences(db, dep.id)[0]
    assert cow_event2.confirmed is True
    cow_obs2 = (
        db.query(SequenceObservation)
        .filter_by(sequence_id=cow_event2.id, label="cow")
        .one()
    )
    assert cow_obs2.human_count == 5
    assert cow_obs2.effective_count == 5


def test_the_interval_regroups_nothing(db):
    """The independence interval is applied by the statistics, so a
    regeneration after changing it gives the same sequences and keeps
    every confirmation and count."""
    from app.api.crud.sequence import generate_sequences_for_project

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    dep = make_deployment(db, project_id=project.id)
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 0, 0), [("cow", "animal", 0.9)])
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 10, 0), [("bear", "animal", 0.9)])
    db.commit()

    generate_sequences_for_project(db, project.id)
    cow_sequence = _sequences(db, dep.id)[0]
    db.query(SequenceObservation).filter_by(
        sequence_id=cow_sequence.id, label="cow"
    ).one().human_count = 4
    cow_sequence.confirmed = True
    db.commit()

    project.independence_interval = 3600
    db.commit()
    generate_sequences_for_project(db, project.id)

    after = _sequences(db, dep.id)
    assert len(after) == 2
    assert after[0].confirmed is True
    cow_after = (
        db.query(SequenceObservation).filter_by(sequence_id=after[0].id, label="cow").one()
    )
    assert cow_after.human_count == 4

def test_generate_relabel_unconfirms_preserved_sequence(db):
    """Same interval, but a label change inside a preserved sequence (e.g.
    smoothing) still clears the confirmation, per the existing rule."""
    from app.api.crud.sequence import generate_sequences_for_project
    from app.models import Detection

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    dep = make_deployment(db, project_id=project.id)
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 0, 0), [("cow", "animal", 0.9)])
    db.commit()

    generate_sequences_for_project(db, project.id)
    sequence = _sequences(db, dep.id)[0]
    sequence.confirmed = True
    db.commit()

    # Relabel the detection (grouping unchanged, species set changes).
    db.query(Detection).update({"label": "elk"})
    db.commit()
    generate_sequences_for_project(db, project.id)

    event2 = _sequences(db, dep.id)[0]
    assert event2.confirmed is False


def test_update_paired_cameras_regroups_only_that_deployment(db):
    """Editing the flag regenerates the deployment's sequences at once,
    keeps confirmations on the other deployment, and clears the
    postprocessing hash so the reprocess banner appears."""
    from app.api.crud.deployment import update_deployment
    from app.api.crud.sequence import generate_sequences_for_project
    from app.api.schemas.deployment import DeploymentUpdate

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    project.postprocessing_settings_hash = "stamped"
    station = make_deployment(db, project_id=project.id, folder_path="/data/station")
    other = make_deployment(db, project_id=project.id, folder_path="/data/other")
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    for cam, offset in (("cam_a", 0), ("cam_b", 1)):
        ts = t0 + timedelta(seconds=offset)
        f = _file_with_dets(db, station.id, ts, [("cow", "animal", 0.9)])
        f.file_path = f"/data/station/{cam}/{f.id}.jpg"
    _file_with_dets(db, other.id, t0, [("deer", "animal", 0.9)])
    db.commit()

    generate_sequences_for_project(db, project.id)
    assert len(_sequences(db, station.id)) == 2
    deer_sequence = _sequences(db, other.id)[0]
    deer_sequence.confirmed = True
    db.commit()

    update_deployment(db, station.id, DeploymentUpdate(paired_cameras=True))

    assert len(_sequences(db, station.id)) == 1
    assert _sequences(db, station.id)[0].file_count == 2
    assert _sequences(db, other.id)[0].confirmed is True
    assert project.postprocessing_settings_hash is None

    # Saving the same value again is a no-op for sequences and the hash.
    project.postprocessing_settings_hash = "stamped"
    db.commit()
    update_deployment(db, station.id, DeploymentUpdate(paired_cameras=True))
    assert len(_sequences(db, station.id)) == 1
    assert project.postprocessing_settings_hash == "stamped"

    # Turning it off splits the sequence again.
    update_deployment(db, station.id, DeploymentUpdate(paired_cameras=False))
    assert len(_sequences(db, station.id)) == 2
    assert project.postprocessing_settings_hash is None


# ── Cohorts: sex, life stage and behaviour per row; notes per sequence ─────
#
# A row is one cohort. The AI seeds one row per species; a person labels it
# in place or splits it. The rebuild carries all of that (DEVELOPERS.md,
# "Observation cohorts").


def _cow_sequence(db, n_cows: int = 6):
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev, _files, dets = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * n_cows}],
    )
    (cow,) = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    return ev, cow, dets


def test_demographics_on_the_ai_row_survive_a_recompute(db):
    """"These 6 cows are all male" is one dropdown on the AI row, and a
    Labels-page relabel (a rebuild) must not lose it."""
    ev, cow, _ = _cow_sequence(db)
    set_observation_attributes(db, cow.id, sex="male", life_stage="adult")

    (recalc,) = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert recalc.max_n == 6
    assert recalc.human_count is None
    assert (recalc.sex, recalc.life_stage, recalc.behavior) == ("male", "adult", None)


def test_split_makes_two_cohorts_and_both_survive_a_recompute(db):
    """Split: source minus one, new human-only row at one with nothing set.
    After a rebuild the AI row is still the seed and the cohort is still
    there, in the same order."""
    ev, cow, _ = _cow_sequence(db)
    new = split_observation(db, cow.id)
    assert new.max_n == 0
    assert new.human_count == 1
    assert (new.sex, new.life_stage, new.behavior) == (None, None, None)
    assert db.get(SequenceObservation, cow.id).effective_count == 5

    set_observation_attributes(db, cow.id, sex="male")
    set_observation_attributes(db, new.id, sex="female", life_stage="juvenile")

    rows = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert [(r.max_n, r.effective_count, r.sex, r.life_stage) for r in rows] == [
        (6, 5, "male", None),
        (0, 1, "female", "juvenile"),
    ]
    assert [r.label for r in list_sequence_observations(db, ev.id)] == ["cow", "cow"]


def test_same_as_last_rows_survive_a_recompute(db):
    """set_human_observations writes ordinary human rows, so a rebuild
    carries them like any other cohort: the seeded AI row keeps its
    override and the demographic cohorts stay as the person left them."""
    ev, _cow, _ = _cow_sequence(db)
    set_human_observations(db, ev.id, [
        {"category": "animal", "label": "cow", "count": 2},
        {
            "category": "animal", "label": "cow", "count": 1,
            "sex": "male", "life_stage": "adult",
        },
        {"category": "animal", "label": "badger", "count": 1, "sex": "female"},
    ])
    assert db.get(Sequence, ev.id).confirmed is False

    rows = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert [
        (r.label, r.max_n, r.effective_count, r.sex, r.life_stage) for r in rows
    ] == [
        ("cow", 6, 2, None, None),
        ("cow", 0, 1, "male", "adult"),
        ("badger", 0, 1, "female", None),
    ]


def test_split_at_one_gives_two_rows_of_one(db):
    """Connect's floor: splitting a row of one never deletes it."""
    ev, cow, _ = _cow_sequence(db, n_cows=1)
    new = split_observation(db, cow.id)
    assert db.get(SequenceObservation, cow.id).effective_count == 1
    assert new.effective_count == 1


def test_a_cohort_survives_even_when_the_ai_now_detects_its_species(db):
    """A split-off cohort is not the plain human-only row the AI merges
    into: it stays its own row when the species gets an AI row."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev, files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12), [{"detections": []}]
    )
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    plain = add_human_species(db, ev.id, category="animal", count=2, label="fox")
    juvenile = add_human_species(
        db, ev.id, category="animal", count=1, label="fox", life_stage="juvenile"
    )
    assert plain.id != juvenile.id

    # The AI now finds a fox: the plain row merges into the AI row (count
    # 2 kept), the juvenile cohort stays.
    make_detection(db, file_id=files[0].id, category="animal", confidence=0.9, label="fox")
    db.flush()
    rows = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert [(r.max_n, r.effective_count, r.life_stage) for r in rows] == [
        (1, 2, None),
        (0, 1, "juvenile"),
    ]


def test_add_species_merges_only_into_the_row_with_the_same_demographics(db):
    ev, cow, _ = _cow_sequence(db)
    set_observation_attributes(db, cow.id, sex="male")
    female = add_human_species(db, ev.id, category="animal", count=2, label="cow", sex="female")
    assert female.id != cow.id
    # Adding the same cohort again changes nothing: its count stays 2.
    again = add_human_species(db, ev.id, category="animal", count=3, label="cow", sex="female")
    assert again.id == female.id
    assert again.effective_count == 2


def test_relabel_keeps_the_cohorts_demographics(db):
    """Relabelling a female cohort of cows to deer gives a female deer row,
    summing only into a female deer row that already exists."""
    ev, cow, _ = _cow_sequence(db)
    add_human_species(db, ev.id, category="animal", count=1, label="deer", sex="female")
    female_cow = add_human_species(
        db, ev.id, category="animal", count=2, label="cow", sex="female"
    )
    target = relabel_observation(db, female_cow.id, category="animal", label="deer")
    assert (target.label, target.sex, target.effective_count) == ("deer", "female", 3)
    deer_rows = [o for o in list_sequence_observations(db, ev.id) if o.label == "deer"]
    assert len(deer_rows) == 1

    male = add_human_species(db, ev.id, category="animal", count=1, label="cow", sex="male")
    target = relabel_observation(db, male.id, category="animal", label="deer")
    assert (target.sex, target.effective_count) == ("male", 1)


def test_demographic_edits_unconfirm_but_notes_do_not(db):
    ev, cow, _ = _cow_sequence(db)
    set_sequence_confirmed(db, ev.id, True)

    set_sequence_notes(db, ev.id, "  grazing by the fence ")
    assert db.get(Sequence, ev.id).confirmed is True
    assert db.get(Sequence, ev.id).notes == "grazing by the fence"
    set_sequence_notes(db, ev.id, "   ")
    assert db.get(Sequence, ev.id).notes is None

    set_observation_attributes(db, cow.id, behavior="foraging")
    assert db.get(Sequence, ev.id).confirmed is False

    # A no-op recompute keeps the sign-off with demographics in place.
    set_sequence_confirmed(db, ev.id, True)
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert db.get(Sequence, ev.id).confirmed is True


def test_set_attributes_leaves_unnamed_fields_alone_and_clears_with_none(db):
    ev, cow, _ = _cow_sequence(db)
    set_observation_attributes(db, cow.id, sex="male", life_stage="adult")
    row = set_observation_attributes(db, cow.id, sex=None)
    assert (row.sex, row.life_stage) == (None, "adult")


def test_reset_to_ai_clears_demographics_and_deletes_cohorts(db):
    ev, cow, _ = _cow_sequence(db)
    set_observation_attributes(db, cow.id, sex="male")
    split_observation(db, cow.id)
    set_sequence_notes(db, ev.id, "kept")
    reset_sequence_to_ai(db, ev.id)
    rows = list_sequence_observations(db, ev.id)
    assert len(rows) == 1
    assert (rows[0].human_count, rows[0].sex) == (None, None)
    # Reset is about the counts; a note is the person's and stays.
    assert db.get(Sequence, ev.id).notes == "kept"


def test_cohorts_sit_under_their_species_in_creation_order(db):
    """Order: species by AI MaxN, then within a species the AI row first
    and the cohorts as they were made. Editing a dropdown does not move a
    row."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev, _files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 3 + [("fox", "animal", 0.9)]}],
    )
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    fox = next(o for o in list_sequence_observations(db, ev.id) if o.label == "fox")
    fox_cohort = split_observation(db, fox.id)
    cow = next(o for o in list_sequence_observations(db, ev.id) if o.label == "cow")
    cow_a = split_observation(db, cow.id)
    cow_b = split_observation(db, cow.id)
    add_human_species(db, ev.id, category="animal", count=1, label="badger")

    def order():
        return [(o.label, o.id) for o in list_sequence_observations(db, ev.id)]

    expected = [
        ("cow", cow.id), ("cow", cow_a.id), ("cow", cow_b.id),
        ("fox", fox.id), ("fox", fox_cohort.id),
    ]
    assert order()[:5] == expected
    assert order()[5][0] == "badger"

    set_observation_attributes(db, cow_a.id, sex="female")
    set_observation_attributes(db, cow.id, life_stage="juvenile")
    assert order()[:5] == expected

    # A rebuild reinserts the rows in that same order.
    calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert [o.label for o in list_sequence_observations(db, ev.id)] == [
        "cow", "cow", "cow", "fox", "fox", "badger",
    ]


def test_generate_sequences_carries_cohorts_when_the_file_set_is_unchanged(db):
    from app.api.crud.sequence import generate_sequences_for_project

    project = make_project(db, counting_threshold=0.5, independence_interval=1800)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev, _files, _ = _make_sequence_with_detections(
        db, dep.id, datetime(2024, 1, 1, 12),
        [{"detections": [("cow", "animal", 0.9)] * 4}],
    )
    (cow,) = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    set_observation_attributes(db, cow.id, sex="male")
    cohort = split_observation(db, cow.id)
    set_observation_attributes(db, cohort.id, sex="female")
    set_sequence_notes(db, ev.id, "with calf")
    set_sequence_confirmed(db, ev.id, True)

    generate_sequences_for_project(db, project.id)
    (new_ev,) = db.query(Sequence).filter(Sequence.deployment_id == dep.id).all()
    rows = list_sequence_observations(db, new_ev.id)
    assert new_ev.confirmed is True
    assert new_ev.notes == "with calf"
    assert [(r.effective_count, r.sex) for r in rows] == [
        (3, "male"),
        (1, "female"),
    ]


def test_notes_survive_a_merge_and_a_split(db):
    """Counts are dropped when sequences regroup; notes never are. A merge
    joins the notes in time order, a split copies the note to every
    child. Turning paired cameras on and off merges and splits."""
    from app.api.crud.sequence import generate_sequences_for_project

    project = make_project(db, counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id, folder_path="/data/pair")
    for cam, secs in (("cam_a", 0), ("cam_b", 30)):
        f = _file_with_dets(
            db, dep.id, datetime(2024, 1, 1, 12, 0, secs), [("cow", "animal", 0.9)]
        )
        f.file_path = f"/data/pair/{cam}/{f.id}.jpg"
    db.commit()
    generate_sequences_for_project(db, project.id)
    first, second = _sequences(db, dep.id)
    set_sequence_notes(db, first.id, "camera knocked")
    set_sequence_notes(db, second.id, "check interval")

    # Merge: one sequence, both notes, earliest first, one per line.
    dep.paired_cameras = True
    db.commit()
    generate_sequences_for_project(db, project.id)
    (merged,) = _sequences(db, dep.id)
    assert merged.notes == "camera knocked\ncheck interval"
    assert merged.confirmed is False

    # Split it again: both children carry the joined note.
    dep.paired_cameras = False
    db.commit()
    generate_sequences_for_project(db, project.id)
    assert [e.notes for e in _sequences(db, dep.id)] == [
        "camera knocked\ncheck interval",
        "camera knocked\ncheck interval",
    ]

def test_demographics_survive_when_the_ai_stops_seeing_the_species(db):
    """An AI row a person labelled keeps that label as a human-only row
    when the species disappears from the detections (a relabel, a higher
    threshold), the same way a count override does."""
    ev, cow, dets = _cow_sequence(db, n_cows=1)
    set_observation_attributes(db, cow.id, sex="male")
    for d in dets:
        d.label = "bear"
    db.flush()
    rows = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()
    assert [(r.label, r.max_n, r.effective_count, r.sex) for r in rows] == [
        ("bear", 1, 1, None),
        ("cow", 0, 1, "male"),
    ]


def test_max_n_frames_carry_the_species_total(db):
    """The card chip reads `max_n_frames[].effective_count`; after a split
    it must be the sum of the cohorts, not the frame row's own count."""
    from app.api.crud.sequence_observation import get_max_n_frames

    ev, cow, _ = _cow_sequence(db, n_cows=2)
    split_observation(db, cow.id)
    (frame,) = get_max_n_frames(db, ev.id)
    assert (frame["label"], frame["max_n"], frame["effective_count"]) == ("cow", 2, 2)


def test_observation_edits_are_scoped_to_the_sequence_in_the_path(db):
    ev, cow, _ = _cow_sequence(db)
    other = make_sequence_with_files(
        db, deployment_id=ev.deployment_id, sequence_start_local=datetime(2024, 2, 1, 12)
    )
    assert set_human_count(db, cow.id, 5, sequence_id=other.id) is None
    assert split_observation(db, cow.id, sequence_id=other.id) is None
    assert set_observation_attributes(db, cow.id, sex="male", sequence_id=other.id) is None
    assert delete_sequence_observation(db, cow.id, sequence_id=other.id) is None
    assert relabel_observation(
        db, cow.id, category="animal", label="deer", sequence_id=other.id
    ) is None
    assert db.get(SequenceObservation, cow.id).effective_count == 6
    assert set_human_count(db, cow.id, 5, sequence_id=ev.id).effective_count == 5


def test_update_camera_offsets_shifts_one_subfolder_and_regroups(db):
    """A per-camera delta moves only that subfolder's files (exact prefix,
    so a sibling folder sharing the prefix is untouched), regenerates the
    deployment's sequences and clears the postprocessing hash. Removing the
    key shifts back."""
    from app.api.crud.deployment import update_deployment
    from app.api.crud.sequence import generate_sequences_for_project
    from app.api.schemas.deployment import DeploymentUpdate
    from app.models import File

    project = make_project(db, counting_threshold=0.5, independence_interval=300)
    project.postprocessing_settings_hash = "stamped"
    station = make_deployment(
        db, project_id=project.id, folder_path="/data/station", paired_cameras=True
    )
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    paths = {}
    # cam_2 fires 301 s after cam_1 (one past the interval), cam_2b is a
    # sibling whose name shares the prefix "cam_2" and must not move.
    for cam, secs in (("cam_1", 0), ("cam_2", 301), ("cam_2b", 0)):
        f = _file_with_dets(db, station.id, t0 + timedelta(seconds=secs), [("cow", "animal", 0.9)])
        f.file_path = f"/data/station/{cam}/{f.id}.jpg"
        paths[cam] = f.file_path
    db.commit()

    generate_sequences_for_project(db, project.id)
    assert len(_sequences(db, station.id)) == 2

    update_deployment(db, station.id, DeploymentUpdate(camera_offsets={"cam_2": -301}))

    by_cam = {
        cam: db.query(File).filter(File.file_path == p).one().captured_at_local
        for cam, p in paths.items()
    }
    assert by_cam["cam_2"] == t0
    assert by_cam["cam_1"] == t0
    assert by_cam["cam_2b"] == t0
    assert len(_sequences(db, station.id)) == 1
    assert _sequences(db, station.id)[0].file_count == 3
    assert station.camera_offsets == {"cam_2": -301}
    assert project.postprocessing_settings_hash is None

    # Same value again: no regroup, hash untouched.
    project.postprocessing_settings_hash = "stamped"
    db.commit()
    update_deployment(db, station.id, DeploymentUpdate(camera_offsets={"cam_2": -301}))
    assert project.postprocessing_settings_hash == "stamped"

    # Dropping the key shifts cam_2 back and splits the sequence again.
    update_deployment(db, station.id, DeploymentUpdate(camera_offsets={}))
    assert db.query(File).filter(File.file_path == paths["cam_2"]).one().captured_at_local == (
        t0 + timedelta(seconds=301)
    )
    assert len(_sequences(db, station.id)) == 2
    assert project.postprocessing_settings_hash is None


def test_video_species_on_a_tracks_representative_frame_is_allowed(db):
    """A tracked video has one card per track, on the track's
    representative frame, so a species that appears there is reviewable
    and spawns a row even when it never touches the cover frame. Its MaxN
    is still the peak across every frame, boxes without a card included."""
    project = make_project(db, counting_threshold=0.5)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev, files = _make_sequence_with_frames(db, dep.id, datetime(2024, 1, 1, 12), [
        {
            "file_type": "video",
            "best_frame_number": 5,
            "detections": [
                {"label": "leopard", "frame_number": 5},                  # cover, a card
                {"label": "hammerhead", "frame_number": 300, "track": "h"},  # the track's card
                {"label": "hammerhead", "frame_number": 330, "track": "h"},
                {"label": "hammerhead", "frame_number": 330, "untracked": True},
                {"label": "carnivora", "frame_number": 12, "untracked": True},  # noise
            ],
        },
    ])

    rows = calculate_max_n_for_sequence(db, ev.id, 0.5)
    db.flush()

    assert {o.label: o.max_n for o in rows} == {"leopard": 1, "hammerhead": 2}
    # The frame each MaxN was counted on rides along for the Counts page.
    assert {o.label: o.max_n_frame_number for o in rows} == {"leopard": 5, "hammerhead": 330}


def _sequence_over(db, deployment_id, video):
    """One sequence holding one video, the shape both gate tests need."""
    from app.models.sequence import Sequence

    ev = Sequence(
        camera_key="",
        id=str(uuid.uuid4()),
        deployment_id=deployment_id,
        sequence_start_local=datetime(2024, 1, 1, 12, 0),
        sequence_end_local=datetime(2024, 1, 1, 12, 0),
        file_count=1,
    )
    db.add(ev)
    db.flush()
    db.execute(
        insert(sequence_files).values(
            sequence_id=ev.id, file_id=video.id, position=0
        )
    )
    return ev


def test_a_species_a_person_named_on_a_frame_gets_a_row(db):
    """Relabelling the frames of a merged track that followed a different
    animal is the documented way to put it right. The card keeps the old
    species, so on the card test alone the new one had no row anywhere:
    the verdict lived in the database and in no count."""
    from app.api.crud.sequence_observation import calculate_max_n_for_sequence
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
        make_project,
        make_site,
        make_track,
    )

    project = make_project(db, counting_threshold=0.2)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path="/fake/merged.mp4", frame_rate=30.0, best_frame_number=60,
    )
    track = make_track(
        db, file_id=video.id, track_key=1, start_frame=30, end_frame=90,
        representative_frame_number=60,
    )
    # The card, and two frames a person relabelled by hand.
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                   category="elasmobranch", label="elasmobranch", confidence=0.9)
    for frame in (30, 90):
        make_detection(
            db, file_id=video.id, frame_number=frame, track_id=track.id,
            category="elasmobranch", label="blacktip", confidence=0.9,
            classification_method="human", verified=True,
        )
    sequence = _sequence_over(db, dep.id, video)
    db.commit()

    rows = calculate_max_n_for_sequence(db, sequence.id, project.counting_threshold)

    by_label = {r.label: r.max_n for r in rows}
    assert by_label.get("blacktip") == 1
    assert by_label.get("elasmobranch") == 1


def test_a_per_frame_classifier_label_still_gets_no_row(db):
    """The other half of the gate: an unverified per-frame reading the
    person has never seen must not spawn a species row."""
    from app.api.crud.sequence_observation import calculate_max_n_for_sequence
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
        make_project,
        make_site,
        make_track,
    )

    project = make_project(db, counting_threshold=0.2)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path="/fake/noisy.mp4", frame_rate=30.0, best_frame_number=60,
    )
    track = make_track(
        db, file_id=video.id, track_key=1, start_frame=30, end_frame=90,
        representative_frame_number=60,
    )
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                   category="animal", label="red deer", confidence=0.9)
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   category="animal", label="roe deer", confidence=0.9,
                   classification_method="model")
    sequence = _sequence_over(db, dep.id, video)
    db.commit()

    rows = calculate_max_n_for_sequence(db, sequence.id, project.counting_threshold)

    assert {r.label for r in rows} == {"red deer"}


# ---------------------------------------------------------------------------
# A group somebody worked on is never cut
# ---------------------------------------------------------------------------


def _legacy_group(db, minutes=(0, 10, 20), label="cow"):
    """One deployment whose files sit 10 minutes apart in ONE stored group,
    as the 30-minute rule made them before sequences existed. Returns
    (project, deployment, group, files)."""
    from app.api.crud.sequence import _create_sequence

    project = make_project(db, counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    files = [
        _file_with_dets(db, dep.id, t0 + timedelta(minutes=m), [(label, "animal", 0.9)])
        for m in minutes
    ]
    group = _create_sequence(db, dep, files)
    calculate_max_n_for_sequence(db, group.id, 0.5)
    db.commit()
    return project, dep, group, files


def _cow_row(db, sequence_id):
    return db.query(SequenceObservation).filter_by(sequence_id=sequence_id, label="cow").one()


def test_a_confirmed_old_group_stays_whole(db):
    from app.api.crud.sequence import generate_sequences_for_project

    project, dep, group, _ = _legacy_group(db)
    _cow_row(db, group.id).human_count = 3
    group.confirmed = True
    db.commit()

    generate_sequences_for_project(db, project.id)

    (kept,) = _sequences(db, dep.id)
    assert kept.file_count == 3
    assert kept.confirmed is True
    assert _cow_row(db, kept.id).human_count == 3


def test_a_count_or_a_demographic_alone_keeps_the_group_whole(db):
    """Not confirmed, but a person set a count, or only a sex."""
    from app.api.crud.sequence import generate_sequences_for_project

    project, dep, group, _ = _legacy_group(db)
    _cow_row(db, group.id).human_count = 2
    db.commit()
    generate_sequences_for_project(db, project.id)
    (kept,) = _sequences(db, dep.id)
    assert _cow_row(db, kept.id).human_count == 2

    project, dep, group, _ = _legacy_group(db)
    set_observation_attributes(db, _cow_row(db, group.id).id, sex="female")
    db.commit()
    generate_sequences_for_project(db, project.id)
    (kept,) = _sequences(db, dep.id)
    assert _cow_row(db, kept.id).sex == "female"


def test_an_untouched_old_group_is_cut_into_sequences(db):
    """The AI's numbers only: nothing to keep, so the new rule applies.
    A note does not protect the group; it is copied to every part."""
    from app.api.crud.sequence import generate_sequences_for_project

    project, dep, group, _ = _legacy_group(db)
    set_sequence_notes(db, group.id, "foggy")
    db.commit()

    generate_sequences_for_project(db, project.id)

    parts = _sequences(db, dep.id)
    assert [p.file_count for p in parts] == [1, 1, 1]
    assert [p.notes for p in parts] == ["foggy"] * 3
    assert not any(p.confirmed for p in parts)


def test_a_worked_on_group_that_gains_a_file_resets_as_before(db):
    """A new file inside the group's time span is not a split: the files
    changed, so the count no longer describes them and resets."""
    from app.api.crud.sequence import generate_sequences_for_project

    project, dep, group, _ = _legacy_group(db, minutes=(0, 1, 10))
    group.confirmed = True
    db.commit()
    _file_with_dets(db, dep.id, datetime(2024, 1, 1, 12, 11, 0), [("cow", "animal", 0.9)])
    db.commit()

    generate_sequences_for_project(db, project.id)

    parts = _sequences(db, dep.id)
    assert [p.file_count for p in parts] == [2, 2]
    assert not any(p.confirmed for p in parts)


def test_an_analysis_regroups_only_its_own_deployment(db):
    """Adding a folder to a project must not cut the old groups of the
    other deployments: the end of an analysis regroups its deployment only."""
    from app.api.crud.sequence import generate_sequences_for_deployment

    project, old_dep, group, _ = _legacy_group(db)
    new_dep = make_deployment(db, project_id=project.id)
    t0 = datetime(2024, 2, 1, 12, 0, 0)
    for m in (0, 10):
        _file_with_dets(db, new_dep.id, t0 + timedelta(minutes=m), [("deer", "animal", 0.9)])
    db.commit()

    generate_sequences_for_deployment(db, new_dep)
    db.commit()

    assert [s.id for s in _sequences(db, old_dep.id)] == [group.id]
    assert [s.file_count for s in _sequences(db, new_dep.id)] == [1, 1]
