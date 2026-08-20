"""Tests for the Labels page's empties endpoint.

An "empty" photo is one where nothing passed: no detection at or above
the grid's current floor, and none verified. It is the same rule as
``derive_observation_type`` computed live at the floor rather than read
from the stored column, so it follows the confidence slider.

The property these tests exist to protect is the one users are told:
every photo is either in the crop grid or in the empties list, never
both and never neither. ``test_every_photo_is_in_exactly_one_of_the_two``
is the one that pins it.
"""

import uuid
from datetime import datetime

from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
)


def _project_with_files(db, n_files=0, threshold=0.2):
    p = make_project(db, counting_threshold=threshold)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    files = [make_file(db, deployment_id=d.id) for _ in range(n_files)]
    return p, d, files


def _empties(client, project_id, **params):
    resp = client.get(
        f"/api/projects/{project_id}/labels/empties", params=params
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_confident_box_is_not_empty(client, db):
    p, _d, (f,) = _project_with_files(db, 1)
    make_detection(db, file_id=f.id, confidence=0.9)
    db.commit()

    assert _empties(client, p.id)["total"] == 0


def test_a_weak_box_is_empty(client, db):
    """The whole point: a box below the threshold does not rescue a
    photo from the empties list, because the user cannot see it."""
    p, _d, (f,) = _project_with_files(db, 1)
    make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    data = _empties(client, p.id)
    assert data["total"] == 1
    assert data["items"][0]["id"] == f.id


def test_a_photo_with_no_boxes_at_all_is_empty(client, db):
    """18% of the empty photos in a real dataset have no detection rows.
    They are the population no crop grid can ever show, so the endpoint
    must not join them away."""
    p, _d, (f,) = _project_with_files(db, 1)
    db.commit()

    assert _empties(client, p.id)["items"][0]["id"] == f.id


def test_lowering_the_slider_empties_the_list(client, db):
    """Dragging the confidence slider down is what makes a photo stop
    being empty: its weak box starts passing. Measured on real data the
    list goes 229 -> 71 between the threshold and the bottom."""
    p, _d, (f,) = _project_with_files(db, 1)
    make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    assert _empties(client, p.id)["total"] == 1
    assert _empties(client, p.id, min_confidence=0.01)["total"] == 0


def test_raising_the_slider_does_not_grow_the_list(client, db):
    """`effective_floor` never goes above the project threshold. A user
    narrowing to a high range is filtering what they look at, not
    redefining what counts as found."""
    p, _d, (f,) = _project_with_files(db, 1)
    make_detection(db, file_id=f.id, confidence=0.3)
    db.commit()

    assert _empties(client, p.id, min_confidence=0.9)["total"] == 0


def test_a_verified_weak_box_is_not_empty(client, db):
    """The threshold-or-verified override. A human decision outranks the
    score, here as everywhere else."""
    p, _d, (f,) = _project_with_files(db, 1)
    make_detection(db, file_id=f.id, confidence=0.01, verified=True)
    db.commit()

    assert _empties(client, p.id)["total"] == 0


def test_a_box_on_another_video_frame_does_not_rescue_a_file(client, db):
    """A video is only its best frame. A confident box on a frame that
    was never written to disk has no card, no crop and no thumbnail, so
    it cannot answer for the clip."""
    p, d, _ = _project_with_files(db, 0)
    v = make_file(
        db,
        deployment_id=d.id,
        file_type="video",
        file_format="mp4",
        best_frame_number=10,
    )
    make_detection(db, file_id=v.id, confidence=0.9, frame_number=99)
    db.commit()

    assert _empties(client, p.id)["total"] == 1

    make_detection(db, file_id=v.id, confidence=0.9, frame_number=10)
    db.commit()
    assert _empties(client, p.id)["total"] == 0


def test_every_photo_is_in_exactly_one_of_the_two(client, db):
    """The mental model the UI promises: every photo is either in the
    crop grid or in the empties list. Break the floor rule on either
    side and a photo lands in both or in neither."""
    from sqlalchemy import or_

    from app.models import Detection, File

    p, _d, files = _project_with_files(db, 6)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    make_detection(db, file_id=files[1].id, confidence=0.25)
    make_detection(db, file_id=files[2].id, confidence=0.05)
    make_detection(db, file_id=files[3].id, confidence=0.01, verified=True)
    make_detection(db, file_id=files[4].id, confidence=0.19)
    # files[5] gets nothing at all.
    db.commit()

    empty_ids = {
        item["id"] for item in _empties(client, p.id, limit=200)["items"]
    }
    in_crop_grid = {
        fid
        for (fid,) in db.query(Detection.file_id)
        .join(File, File.id == Detection.file_id)
        .filter(
            or_(Detection.confidence >= 0.2, Detection.verified == True)  # noqa: E712
        )
        .distinct()
    }

    all_ids = {f.id for f in files}
    assert empty_ids | in_crop_grid == all_ids, "a photo is in neither"
    assert not (empty_ids & in_crop_grid), "a photo is in both"


def test_the_checked_filter_hides_what_is_done(client, db):
    p, _d, (a, b) = _project_with_files(db, 2)
    a.verified = True
    db.commit()

    assert _empties(client, p.id, verification="unverified")["total"] == 1
    assert _empties(client, p.id, verification="verified")["total"] == 1
    assert _empties(client, p.id)["total"] == 2


def test_path_is_the_default_sort(client, db):
    """Folder order groups one camera's photos together. Capture-time
    order interleaves cameras, which is miserable when the job is
    scanning the same scene over and over."""
    p, d, _ = _project_with_files(db, 0)
    later = make_file(
        db,
        deployment_id=d.id,
        file_path="/cam-b/IMG_0001.jpg",
        captured_at_local=datetime(2024, 1, 2, 12, 0),
    )
    earlier = make_file(
        db,
        deployment_id=d.id,
        file_path="/cam-a/IMG_0009.jpg",
        captured_at_local=datetime(2024, 1, 1, 12, 0),
    )
    db.commit()

    by_path = [i["id"] for i in _empties(client, p.id)["items"]]
    assert by_path == [earlier.id, later.id]

    by_newest = [
        i["id"] for i in _empties(client, p.id, sort="newest")["items"]
    ]
    assert by_newest == [later.id, earlier.id]


def test_random_is_stable_for_a_seed_and_needs_one(client, db):
    """Sampling is the only workable strategy on a big project, and a
    seed is what stops the sample reshuffling under the user between
    pages."""
    p, _d, _files = _project_with_files(db, 8)
    db.commit()

    first = [i["id"] for i in _empties(client, p.id, sort="random", seed=7)["items"]]
    again = [i["id"] for i in _empties(client, p.id, sort="random", seed=7)["items"]]
    assert first == again

    resp = client.get(
        f"/api/projects/{p.id}/labels/empties", params={"sort": "random"}
    )
    assert resp.status_code == 400


def test_total_is_the_uncapped_count(client, db):
    p, _d, _files = _project_with_files(db, 5)
    db.commit()

    data = _empties(client, p.id, limit=2)
    assert data["total"] == 5
    assert len(data["items"]) == 2


def test_the_floor_is_echoed_so_the_page_can_name_it(client, db):
    p, _d, _files = _project_with_files(db, 1, threshold=0.35)
    db.commit()

    assert _empties(client, p.id)["floor"] == 0.35
    assert _empties(client, p.id, min_confidence=0.02)["floor"] == 0.02


def test_other_projects_are_not_included(client, db):
    p, _d, _files = _project_with_files(db, 2)
    other, _od, _of = _project_with_files(db, 3)
    db.commit()

    assert _empties(client, p.id)["total"] == 2
    assert _empties(client, other.id)["total"] == 3


def test_a_deployment_with_no_site_is_included(client, db):
    """Folder runs never create sites, so a site join would return
    nothing at all in the mode this feature most needs to work in."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id)
    make_file(db, deployment_id=d.id)
    db.commit()

    assert _empties(client, p.id)["total"] == 1


def test_unknown_project_is_404(client, db):
    resp = client.get(
        f"/api/projects/{uuid.uuid4()}/labels/empties"
    )
    assert resp.status_code == 404


# ── One progress bar for the whole Labels page ──────────────────────


def _progress(client, project_id, **params):
    resp = client.get(
        f"/api/projects/{project_id}/labels/progress", params=params
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_label_is_a_passing_box_or_an_empty_file(client, db):
    """The unit: every card a person has to look at, across both tabs.
    Three passing boxes on two files, plus one file with nothing, is
    four labels."""
    p, _d, files = _project_with_files(db, 3)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    make_detection(db, file_id=files[1].id, confidence=0.9)
    # files[2] has nothing: one "nothing here" label.
    db.commit()

    data = _progress(client, p.id)
    assert data["total_labels"] == 4
    assert data["verified_labels"] == 0
    # The halves are reported separately so each tab can point at the
    # other: three boxes in Crops, one empty file in Empties.
    assert data["crop_labels"] == 3
    assert data["empty_labels"] == 1


def test_the_total_is_the_cards_across_both_tabs(client, db):
    """The property the single bar rests on: crops + empties, with no
    overlap and nothing missed, because a file either has a passing box
    or it does not."""
    p, _d, files = _project_with_files(db, 5)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    make_detection(db, file_id=files[0].id, confidence=0.4)
    make_detection(db, file_id=files[1].id, confidence=0.25)
    make_detection(db, file_id=files[2].id, confidence=0.05)
    make_detection(db, file_id=files[3].id, confidence=0.01, verified=True)
    # files[4] has nothing at all.
    db.commit()

    from app.models import Detection, File

    empties = _empties(client, p.id, limit=200)["total"]
    crops = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .filter(
            (Detection.confidence >= 0.2) | (Detection.verified == True)  # noqa: E712
        )
        .count()
    )
    assert _progress(client, p.id)["total_labels"] == crops + empties


def test_a_below_threshold_box_is_not_a_label(client, db):
    """It has no card in either tab: too weak for Crops, and its file
    shows in Empties as a single label, not one per hidden box."""
    p, _d, (f,) = _project_with_files(db, 1)
    make_detection(db, file_id=f.id, confidence=0.05)
    make_detection(db, file_id=f.id, confidence=0.03)
    db.commit()

    assert _progress(client, p.id)["total_labels"] == 1


def test_verifying_a_box_moves_the_bar(client, db):
    """One box, one label, so the bar ticks per box. Counting files
    instead made a two-animal photo look like no progress at all."""
    p, _d, (f,) = _project_with_files(db, 1)
    d1 = make_detection(db, file_id=f.id, confidence=0.9)
    make_detection(db, file_id=f.id, confidence=0.9)
    db.commit()

    client.patch(f"/api/detections/{d1.id}/verify", json={"verified": True})
    data = _progress(client, p.id)
    assert (data["total_labels"], data["verified_labels"]) == (2, 1)
    assert (data["crop_labels"], data["crop_labels_verified"]) == (2, 1)


def test_confirming_an_empty_file_moves_the_bar(client, db):
    """The other half. Without this the bar could read 100% while every
    empty file was untouched."""
    p, _d, files = _project_with_files(db, 2)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    db.commit()

    assert _progress(client, p.id)["verified_labels"] == 0
    client.patch(f"/api/files/{files[1].id}", json={"verified": True})
    data = _progress(client, p.id)
    assert (data["total_labels"], data["verified_labels"]) == (2, 1)
    assert (data["empty_labels"], data["empty_labels_verified"]) == (1, 1)


def test_progress_ignores_the_verified_filter(client, db):
    """There is deliberately no verification parameter: a bar whose
    denominator moves with the thing it measures can only read 0% or
    100%."""
    p, _d, files = _project_with_files(db, 3)
    db.commit()
    client.patch(f"/api/files/{files[0].id}", json={"verified": True})

    assert (
        _progress(client, p.id, verification="unverified")["total_labels"]
        == 3
    )


def test_progress_follows_the_site_filter(client, db):
    from tests.conftest import make_file

    p = make_project(db)
    s1 = make_site(db, project_id=p.id)
    s2 = make_site(db, project_id=p.id)
    d1 = make_deployment(db, site_id=s1.id)
    d2 = make_deployment(db, site_id=s2.id)
    make_file(db, deployment_id=d1.id)
    make_file(db, deployment_id=d2.id)
    make_file(db, deployment_id=d2.id)
    db.commit()

    assert (
        _progress(client, p.id, site_ids=s2.id)["total_labels"] == 2
    )


def test_progress_unknown_project_is_404(client, db):
    resp = client.get(f"/api/projects/{uuid.uuid4()}/labels/progress")
    assert resp.status_code == 404
