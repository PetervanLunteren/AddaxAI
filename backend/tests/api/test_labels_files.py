"""Tests for the Labels page's files endpoint and its progress counts.

The Files tab lists every file. Its ``empty`` filter narrows by whether
anything on the file's visible surface passes: ``show_only`` keeps the
files where nothing does, ``hide`` the files where something does. That
is the same rule as ``derive_observation_type``, judged at the project's
counting threshold and at nothing else: the confidence slider is clamped
there on Files, so a transient control can never redefine "empty".

The property these tests protect: the two halves of the ``empty`` filter
partition the project, and ``show_only`` is exactly the set of files with
no card in the Detections tab. ``test_every_photo_is_in_exactly_one_half``
pins it.
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


def _files(client, project_id, **params):
    resp = client.get(f"/api/projects/{project_id}/labels/files", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _empties(client, project_id, **params):
    return _files(client, project_id, empty="show_only", **params)


# ── The empty filter ──────────────────────────────────────────────────


def test_a_confident_box_is_not_empty(client, db):
    p, _d, (f,) = _project_with_files(db, 1)
    make_detection(db, file_id=f.id, confidence=0.9)
    db.commit()

    assert _empties(client, p.id)["total"] == 0


def test_a_weak_box_is_empty(client, db):
    """The whole point: a box below the threshold does not rescue a
    photo from the empty half, because the user cannot see it."""
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


def test_the_slider_never_redefines_empty(client, db):
    """"Empty" is defined by the project threshold, full stop. The Files
    slider is clamped there, so a below-floor min (a stale URL) changes
    nothing, and a raised min is a box filter that can only shrink the
    list. One control, one meaning."""
    p, _d, (f,) = _project_with_files(db, 1)
    make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    assert _empties(client, p.id)["total"] == 1
    assert _empties(client, p.id, min_confidence=0.01)["total"] == 1


def test_raising_the_slider_does_not_grow_the_list(client, db):
    """A user narrowing to a high range is filtering what they look at,
    not redefining what counts as found: an empty file has no box in
    any range, so the combination yields nothing."""
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


def test_every_photo_is_in_exactly_one_half(client, db):
    """`show_only` is exactly the set of files with no card in the
    Detections tab, and `hide` is the rest. Break the floor rule on
    either side and a photo lands in both or in neither."""
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
    with_boxes = {
        item["id"]
        for item in _files(client, p.id, empty="hide", limit=200)["items"]
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
    assert with_boxes == in_crop_grid
    assert empty_ids | with_boxes == all_ids, "a photo is in neither"
    assert not (empty_ids & with_boxes), "a photo is in both"


def test_all_is_the_default_and_lists_every_file(client, db):
    """The Files tab opens on everything. The empty filter is a lens on
    the same list, not a separate one."""
    p, _d, files = _project_with_files(db, 3)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    make_detection(db, file_id=files[1].id, confidence=0.05)
    db.commit()

    assert _files(client, p.id)["total"] == 3
    assert _files(client, p.id, empty="all")["total"] == 3
    assert _files(client, p.id, empty="hide")["total"] == 1
    assert _files(client, p.id, empty="show_only")["total"] == 2


def test_an_unknown_empty_value_is_refused(client, db):
    p, _d, _files_ = _project_with_files(db, 1)
    db.commit()

    resp = client.get(
        f"/api/projects/{p.id}/labels/files", params={"empty": "maybe"}
    )
    assert resp.status_code == 422


# ── Listing ───────────────────────────────────────────────────────────


def test_the_checked_filter_hides_what_is_done(client, db):
    p, _d, (a, b) = _project_with_files(db, 2)
    a.verified = True
    db.commit()

    assert _files(client, p.id, verification="unverified")["total"] == 1
    assert _files(client, p.id, verification="verified")["total"] == 1
    assert _files(client, p.id)["total"] == 2


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

    by_path = [i["id"] for i in _files(client, p.id)["items"]]
    assert by_path == [earlier.id, later.id]

    by_newest = [i["id"] for i in _files(client, p.id, sort="newest")["items"]]
    assert by_newest == [later.id, earlier.id]


def test_random_is_stable_for_a_seed_and_needs_one(client, db):
    """Sampling is the only workable strategy on a big project, and a
    seed is what stops the sample reshuffling under the user between
    pages."""
    p, _d, _files_ = _project_with_files(db, 8)
    db.commit()

    first = [i["id"] for i in _files(client, p.id, sort="random", seed=7)["items"]]
    again = [i["id"] for i in _files(client, p.id, sort="random", seed=7)["items"]]
    assert first == again

    resp = client.get(
        f"/api/projects/{p.id}/labels/files", params={"sort": "random"}
    )
    assert resp.status_code == 400


def test_flagged_and_liked_filters_select_files(client, db):
    """The Counts triage filters on Files: flag and heart live on the
    file row itself, so the filter is a plain column match and "all"
    (or absent) means no clause."""
    p, d, _ = _project_with_files(db, 0)
    plain = make_file(db, deployment_id=d.id)
    marked = make_file(db, deployment_id=d.id)
    marked.flagged = True
    liked = make_file(db, deployment_id=d.id)
    liked.favorited = True
    db.commit()

    got = _files(client, p.id, flagged="flagged")
    assert [i["id"] for i in got["items"]] == [marked.id]
    got = _files(client, p.id, favorited="favorited")
    assert [i["id"] for i in got["items"]] == [liked.id]
    got = _files(client, p.id, flagged="not_flagged")
    assert {i["id"] for i in got["items"]} == {plain.id, liked.id}
    assert _files(client, p.id, flagged="all")["total"] == 3
    assert _files(client, p.id)["total"] == 3


def test_find_reports_the_files_position_in_the_ordering(client, db):
    """`find` answers "where does this file sit in the list", under the
    same filters and sort as the page itself. It is what lets the
    Detections modal's "Open in files view" open the viewer at the
    file's real position, so next and previous continue through the
    list instead of a one-file dead end."""
    p, d, _ = _project_with_files(db, 0)
    for name in ("cam-b/IMG_2.jpg", "cam-a/IMG_1.jpg", "cam-c/IMG_3.jpg"):
        make_file(db, deployment_id=d.id, file_path=f"/{name}")
    db.commit()

    listed = [i["id"] for i in _files(client, p.id)["items"]]
    for want, fid in enumerate(listed):
        assert _files(client, p.id, find=fid)["find_index"] == want


def test_find_follows_the_sort(client, db):
    """The position is under the requested order, not path order, and a
    seeded random order answers consistently with its own listing."""
    p, d, _ = _project_with_files(db, 0)
    old = make_file(
        db,
        deployment_id=d.id,
        file_path="/a.jpg",
        captured_at_local=datetime(2024, 1, 1, 12, 0),
    )
    new = make_file(
        db,
        deployment_id=d.id,
        file_path="/b.jpg",
        captured_at_local=datetime(2024, 1, 2, 12, 0),
    )
    db.commit()

    assert _files(client, p.id, sort="newest", find=old.id)["find_index"] == 1
    assert _files(client, p.id, sort="newest", find=new.id)["find_index"] == 0

    shuffled = [
        i["id"]
        for i in _files(client, p.id, sort="random", seed=7)["items"]
    ]
    got = _files(client, p.id, sort="random", seed=7, find=old.id)
    assert got["find_index"] == shuffled.index(old.id)


def test_find_outside_the_filters_is_null(client, db):
    """A file the current view does not hold has no position; the
    frontend reads null and widens the view instead of guessing."""
    p, _d, (f,) = _project_with_files(db, 1)
    f.verified = True
    db.commit()

    data = _files(client, p.id, verification="unverified", find=f.id)
    assert data["find_index"] is None
    # And not asked for means not answered.
    assert _files(client, p.id)["find_index"] is None


def test_total_is_the_uncapped_count(client, db):
    p, _d, _files_ = _project_with_files(db, 5)
    db.commit()

    data = _files(client, p.id, limit=2)
    assert data["total"] == 5
    assert len(data["items"]) == 2


def test_the_floor_is_echoed_so_the_page_can_name_it(client, db):
    """Always the project threshold: the slider never moves the Files
    floor, so the page can quote the setting without caveats."""
    p, _d, _files_ = _project_with_files(db, 1, threshold=0.35)
    db.commit()

    assert _files(client, p.id)["floor"] == 0.35
    assert _files(client, p.id, min_confidence=0.02)["floor"] == 0.35


def test_other_projects_are_not_included(client, db):
    p, _d, _files_ = _project_with_files(db, 2)
    other, _od, _of = _project_with_files(db, 3)
    db.commit()

    assert _files(client, p.id)["total"] == 2
    assert _files(client, other.id)["total"] == 3


def test_a_deployment_with_no_site_is_included(client, db):
    """Folder runs never create sites, so a site join would return
    nothing at all in the mode this feature most needs to work in."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id)
    make_file(db, deployment_id=d.id)
    db.commit()

    assert _files(client, p.id)["total"] == 1


def test_unknown_project_is_404(client, db):
    resp = client.get(f"/api/projects/{uuid.uuid4()}/labels/files")
    assert resp.status_code == 404


# ── One progress bar for the whole Labels page ──────────────────────


def _progress(client, project_id, **params):
    resp = client.get(
        f"/api/projects/{project_id}/labels/progress", params=params
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_label_is_a_passing_box_or_an_empty_file(client, db):
    """The unit: every call a person has to make. Three passing boxes on
    two files, plus one file with nothing, is four labels."""
    p, _d, files = _project_with_files(db, 3)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    make_detection(db, file_id=files[1].id, confidence=0.9)
    # files[2] has nothing: one "nothing here" label.
    db.commit()

    data = _progress(client, p.id)
    assert data["total_labels"] == 4
    assert data["verified_labels"] == 0
    assert data["crop_labels"] == 3
    assert data["empty_labels"] == 1


def test_the_total_is_boxes_plus_empty_files(client, db):
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
    """It has no card in Detections, and its file is one "nothing here"
    label, not one per hidden box."""
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


def test_files_counts_feed_the_files_chip(client, db):
    """The Files tab's chip counts files, whatever is on them. A file with
    two boxes is two labels for the bar and one file for the chip, and it
    is only signed off once every box is."""
    p, _d, files = _project_with_files(db, 2)
    d1 = make_detection(db, file_id=files[0].id, confidence=0.9)
    make_detection(db, file_id=files[0].id, confidence=0.9)
    db.commit()

    data = _progress(client, p.id)
    assert (data["files"], data["files_verified"]) == (2, 0)
    assert data["total_labels"] == 3

    client.patch(f"/api/detections/{d1.id}/verify", json={"verified": True})
    assert _progress(client, p.id)["files_verified"] == 0

    client.patch(f"/api/files/{files[0].id}", json={"verified": True})
    assert _progress(client, p.id)["files_verified"] == 1


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
    p = make_project(db)
    s1 = make_site(db, project_id=p.id)
    s2 = make_site(db, project_id=p.id)
    d1 = make_deployment(db, site_id=s1.id)
    d2 = make_deployment(db, site_id=s2.id)
    make_file(db, deployment_id=d1.id)
    make_file(db, deployment_id=d2.id)
    make_file(db, deployment_id=d2.id)
    db.commit()

    assert _progress(client, p.id, site_ids=s2.id)["total_labels"] == 2
    assert _progress(client, p.id, site_ids=s2.id)["files"] == 2


def test_progress_unknown_project_is_404(client, db):
    resp = client.get(f"/api/projects/{uuid.uuid4()}/labels/progress")
    assert resp.status_code == 404


# ── The labels filter ─────────────────────────────────────────────────


def test_labels_filter_keeps_files_containing_the_species(client, db):
    """Same semantics as the events label filter: a file matches when at
    least one VISIBLE box carries the picked taxonomy id. A weak
    unverified box of the species does not rescue a file (the user
    cannot see it), and a rejected box never matches
    (`is_a_real_detection` inside the passing scope)."""
    from app.models.label_taxonomy import LabelTaxonomy

    p, d, (with_species, weak_only, other) = _project_with_files(db, 3)
    tax = LabelTaxonomy(
        classification_model_id="m", name="horse", level="species"
    )
    db.add(tax)
    db.flush()

    make_detection(
        db, file_id=with_species.id, confidence=0.9,
        label="horse", label_taxonomy_id=tax.id,
    )
    make_detection(
        db, file_id=weak_only.id, confidence=0.05,
        label="horse", label_taxonomy_id=tax.id,
    )
    make_detection(db, file_id=other.id, confidence=0.9, label="deer")
    db.commit()

    got = _files(client, p.id, labels=tax.id)
    assert [i["id"] for i in got["items"]] == [with_species.id]

    # A rejected box of the species is not the species.
    make_detection(
        db, file_id=other.id, confidence=0.9,
        label="false detection", label_taxonomy_id=tax.id, verified=True,
    )
    db.commit()
    got = _files(client, p.id, labels=tax.id)
    assert [i["id"] for i in got["items"]] == [with_species.id]


def test_confidence_range_selects_files_by_a_single_matching_box(client, db):
    """The Detections rules lifted to files: a file shows when ONE box
    satisfies every box filter together. dog@0.9 + cat@0.3 must not
    match "dogs below 40%" off two different boxes, and a ceiling or a
    raised minimum drops files with no box in the range. Boxless files
    only appear while no box filter is active."""
    from app.models.label_taxonomy import LabelTaxonomy

    p, d, (strong, weakish, boxless) = _project_with_files(db, 3)
    dog = LabelTaxonomy(
        classification_model_id="m", name="dog", level="species"
    )
    db.add(dog)
    db.flush()

    make_detection(
        db, file_id=strong.id, confidence=0.9,
        label="dog", label_taxonomy_id=dog.id,
    )
    make_detection(db, file_id=strong.id, confidence=0.3, label="cat")
    make_detection(db, file_id=weakish.id, confidence=0.3, label="cat")
    db.commit()

    # Raised minimum: only the file with a strong box.
    got = _files(client, p.id, min_confidence=0.5)
    assert [i["id"] for i in got["items"]] == [strong.id]

    # Ceiling: both box-holding files have a box <= 0.4; boxless drops.
    got = _files(client, p.id, max_confidence=0.4)
    assert {i["id"] for i in got["items"]} == {strong.id, weakish.id}

    # One box must match everything: there is no dog at or below 0.4.
    got = _files(client, p.id, labels=dog.id, max_confidence=0.4)
    assert got["total"] == 0
    got = _files(client, p.id, labels=dog.id, min_confidence=0.5)
    assert [i["id"] for i in got["items"]] == [strong.id]

    # A below-floor min (stale URL; the slider is clamped) is a no-op:
    # every file stays listed, the boxless one included.
    got = _files(client, p.id, min_confidence=0.05)
    assert got["total"] == 3


def test_classification_range_selects_by_a_single_box_too(client, db):
    """Label-confidence bounds join the same one-box-matches-all rule,
    and a box the classifier never named (NULL score) can never satisfy
    a bound, mirroring the Detections grid."""
    p, d, (sure, unsure, unnamed) = _project_with_files(db, 3)
    make_detection(
        db, file_id=sure.id, confidence=0.9,
        label="dog", label_confidence=0.95,
    )
    make_detection(
        db, file_id=unsure.id, confidence=0.9,
        label="dog", label_confidence=0.4,
    )
    make_detection(db, file_id=unnamed.id, confidence=0.9, label=None)
    db.commit()

    got = _files(client, p.id, min_label_confidence=0.8)
    assert [i["id"] for i in got["items"]] == [sure.id]

    got = _files(client, p.id, max_label_confidence=0.5)
    assert [i["id"] for i in got["items"]] == [unsure.id]

    # No bounds: the unnamed box's file is simply a file like any other.
    assert _files(client, p.id)["total"] == 3


def test_same_second_ties_break_by_file_name(client, db):
    """A burst shot within one second must come back in shooting order.
    The tie used to fall through to the row id, which is ingest order
    and shuffled such bursts."""
    p, d, _ = _project_with_files(db, 0)
    t = datetime(2024, 5, 1, 12, 0, 0)
    # Created out of name order on purpose: ingest order must not win.
    b = make_file(db, deployment_id=d.id, file_path="/x/IMG_0002.jpg",
                  captured_at_local=t)
    a = make_file(db, deployment_id=d.id, file_path="/x/IMG_0001.jpg",
                  captured_at_local=t)
    c = make_file(db, deployment_id=d.id, file_path="/x/IMG_0003.jpg",
                  captured_at_local=t)
    db.commit()

    got = _files(client, p.id, sort="oldest")
    assert [i["id"] for i in got["items"]] == [a.id, b.id, c.id]
    got = _files(client, p.id, sort="newest")
    assert [i["id"] for i in got["items"]] == [a.id, b.id, c.id]


def test_event_sort_keeps_bursts_together(client, db):
    """Events newest first, each burst contiguous and in shooting order,
    files outside any event last. Items carry their event id so the grid
    can draw dividers."""
    from sqlalchemy import insert

    from app.models.event import Event
    from app.models.event import event_files as ef
    from tests.conftest import make_event_with_files  # noqa: F401

    p, d, _ = _project_with_files(db, 0)

    def burst(start, names):
        files = [
            make_file(db, deployment_id=d.id, file_path=n,
                      captured_at_local=start)
            for n in names
        ]
        ev = Event(deployment_id=d.id, event_start_local=start,
                   event_end_local=start)
        db.add(ev)
        db.flush()
        for f in files:
            db.execute(insert(ef).values(event_id=ev.id, file_id=f.id))
        return ev, files

    old_ev, old_files = burst(
        datetime(2024, 1, 1, 8, 0), ["/x/A_2.jpg", "/x/A_1.jpg"]
    )
    new_ev, new_files = burst(
        datetime(2024, 6, 1, 8, 0), ["/x/B_1.jpg"]
    )
    loose = make_file(db, deployment_id=d.id, file_path="/x/C_1.jpg",
                      captured_at_local=datetime(2024, 3, 1, 8, 0))
    db.commit()

    got = _files(client, p.id, sort="events")
    ids = [i["id"] for i in got["items"]]
    assert ids == [
        new_files[0].id, old_files[1].id, old_files[0].id, loose.id
    ]
    assert got["items"][0]["event_id"] == new_ev.id
    assert got["items"][1]["event_id"] == old_ev.id
    assert got["items"][3]["event_id"] is None

    # `find` under the same sort: the correlated event subqueries must
    # work as window order keys too, or the handoff position would
    # disagree with the listing.
    for want, fid in enumerate(ids):
        found = _files(client, p.id, sort="events", find=fid)
        assert found["find_index"] == want
