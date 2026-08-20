"""Marking a detection false takes it out of what a file is about.

"Mark false" on the Labels page writes the label ``false detection`` and
deliberately leaves the detector's ``category`` alone, because the
category is the detector's own word and is never translated. It also
verifies the box, and a verified box always passes the threshold.

Those two facts together used to make a rejected box the file's subject:
the file exported ``observation_type = animal`` with
``classification_label = false detection`` beside it, and the Counts page
grew an observation called "false detection" with a MaxN of 1. A thing
the user had just declared not real became a species with a count.

The AI's own non-label calls never reach the database at all; the ingest
skip drops them (DEVELOPERS.md, "Non-label detection skip"). These tests
pin the same rule at read time, which is the only place it can be applied
when a human reaches that verdict later.

The row itself is kept, not deleted. A human looked at that box and
judged it, which is worth recording, keeps undo working, and keeps
``detections.csv`` honest about what the detector found and what was
rejected.
"""

from app.api.crud.event_observation import calculate_max_n_for_event
from app.api.crud.export import (
    build_detection_rows,
    build_files_rows,
    get_scoped_detection_rows,
)
from app.models import Event, EventObservation, File
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
)


def _scaffold(db, threshold=0.2):
    project = make_project(db, counting_threshold=threshold)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    file = make_file(db, deployment_id=deployment.id)
    return project, file


def _mark_false(client, detection_ids):
    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": detection_ids, "label": "false detection"},
    )
    assert resp.status_code == 200, resp.text


def test_a_file_whose_only_box_was_marked_false_is_blank(client, db):
    project, f = _scaffold(db)
    det = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
    db.commit()

    _mark_false(client, [det.id])

    db.expire_all()
    assert db.get(File, f.id).observation_type == "blank"


def test_the_files_export_carries_no_species_for_it(client, db):
    project, f = _scaffold(db)
    det = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
    db.commit()

    _mark_false(client, [det.id])
    db.expire_all()

    headers, rows = build_files_rows(db, db.merge(project))
    row = dict(zip(headers, rows[0], strict=False))
    assert row["observation_type"] == "blank"
    # The species block comes off the deciding box. There is no deciding
    # box now, so naming one would be inventing a species for a file the
    # user said holds nothing.
    assert row["classification_label"] == ""
    assert row["scientific_name"] == ""


def test_a_real_animal_beside_it_still_decides_the_file(client, db):
    """The rejected box drops out; it does not drag the file with it.

    The falsed box is the stronger of the two, so this also pins that the
    skip happens before the strongest-wins comparison rather than after.
    """
    project, f = _scaffold(db)
    strong = make_detection(db, file_id=f.id, category="animal", confidence=0.95)
    real = make_detection(
        db, file_id=f.id, category="animal", confidence=0.4, label="deer"
    )
    db.commit()

    _mark_false(client, [strong.id])
    db.expire_all()

    assert db.get(File, f.id).observation_type == "animal"
    headers, rows = build_files_rows(db, db.merge(project))
    assert dict(zip(headers, rows[0], strict=False))["classification_label"] == "deer"
    assert real.id  # the survivor is the one that named it


def test_the_rejected_box_is_still_in_the_detections_export(client, db):
    """The record survives. This is the reason the row is not deleted."""
    project, f = _scaffold(db)
    det = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
    db.commit()

    _mark_false(client, [det.id])
    db.expire_all()

    project = db.merge(project)
    headers, rows = build_detection_rows(
        db, project, get_scoped_detection_rows(db, project)
    )
    assert len(rows) == 1
    row = dict(zip(headers, rows[0], strict=False))
    assert row["classification_label"] == "false detection"
    assert row["is_verified"] == "TRUE"


def test_it_never_becomes_a_counted_observation(client, db):
    """The Counts page groups by COALESCE(label, category) in its own
    query, so it needs the rule applied separately from the file one."""
    project, f = _scaffold(db)
    det = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
    event = Event(
        id="ev-false",
        deployment_id=f.deployment_id,
        event_start_local=f.captured_at_local,
        event_end_local=f.captured_at_local,
    )
    db.add(event)
    db.flush()
    event.files.append(f)
    db.commit()

    _mark_false(client, [det.id])
    calculate_max_n_for_event(db, event.id, 0.2)
    db.commit()

    rows = (
        db.query(EventObservation)
        .filter(EventObservation.event_id == event.id)
        .all()
    )
    assert [r.label for r in rows] == []


def test_every_non_label_class_behaves_the_same(client, db):
    """`bait`, `blank`, `empty`, `none` and `vide` say the same thing as
    `false detection`, and the ingest skip already treats them as one
    set. Read time must not pick favourites."""
    for word in ("bait", "blank", "empty", "none", "vide"):
        project, f = _scaffold(db)
        det = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
        db.commit()

        resp = client.post(
            "/api/detections/bulk-relabel",
            json={"detection_ids": [det.id], "label": word},
        )
        assert resp.status_code == 200, resp.text

        db.expire_all()
        assert db.get(File, f.id).observation_type == "blank", word


def test_the_empties_tab_and_the_export_agree_about_it(client, db):
    """`observation_type` and tab membership answer the same question, so
    they must not answer it differently.

    The file-level rule skipped non-label detections, but the Empties
    query did not: a falsed box is verified, and verified always passes
    its threshold-or-verified clause. So the file exported
    `observation_type = blank` while the tab, its progress bar and the
    dashboard counter all still called it a detection to check. On a real
    project that read 231 blank rows in files.csv against 229 in the tab.
    """
    project, f = _scaffold(db)
    det = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
    db.commit()
    _mark_false(client, [det.id])

    empties = client.get(f"/api/projects/{project.id}/labels/empties",
                         params={"verification": "all", "limit": 200})
    assert empties.status_code == 200, empties.text
    assert f.id in {i["id"] for i in empties.json()["items"]}

    progress = client.get(f"/api/projects/{project.id}/labels/progress")
    body = progress.json()
    assert body["empty_labels"] == 1
    assert body["crop_labels"] == 0


def test_an_unknown_verification_value_is_refused(client, db):
    """Every other parameter on this endpoint validates. A typo used to
    return 200 with silently unfiltered data."""
    project, _ = _scaffold(db)
    resp = client.get(f"/api/projects/{project.id}/labels/empties",
                      params={"verification": "nonsense"})
    assert resp.status_code == 422


def test_a_malformed_date_is_refused_rather_than_crashing(client, db):
    """`datetime.fromisoformat` raises ValueError, which reached the user
    as a 500 on a query they could fix themselves."""
    project, _ = _scaffold(db)
    for endpoint in ("empties", "progress"):
        resp = client.get(f"/api/projects/{project.id}/labels/{endpoint}",
                          params={"date_from": "notadate"})
        assert resp.status_code == 422, f"{endpoint}: {resp.status_code}"


def test_progress_follows_the_confidence_slider(client, db):
    """The chips on the tab switch read from this, and they sit beside
    grids that follow the slider. Pinned to the project threshold they
    contradicted the grid: "Empties 220" above "68 files"."""
    project, f = _scaffold(db)
    make_detection(db, file_id=f.id, category="animal", confidence=0.05)
    db.commit()

    at_rest = client.get(f"/api/projects/{project.id}/labels/progress").json()
    assert (at_rest["empty_labels"], at_rest["crop_labels"]) == (1, 0)

    lowered = client.get(f"/api/projects/{project.id}/labels/progress",
                         params={"min_confidence": 0.01}).json()
    assert (lowered["empty_labels"], lowered["crop_labels"]) == (0, 1)
