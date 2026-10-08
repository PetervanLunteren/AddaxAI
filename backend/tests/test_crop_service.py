"""Crop thumbnails never come from a frame the detection isn't on.

A video's detections sit on every sampled frame, but only the best frame
is written to disk as a JPEG. Cropping that one frame at a bbox belonging
to a different moment produces a confident picture of the wrong place, and
of nothing at all once the animal has walked out of that corner. On a
30-second clip of a person walking, 31 of 32 grid tiles were leaf litter.

It stayed hidden for so long because a slow subject makes it look right:
successive bboxes overlap, so the crops still contain the animal, just
shifted. Only the moving-subject case exposes it.

The `_COMMON_JOINS` gate in similarity_script keeps these detections out of
the grid; this is the second half, so any other caller of `crop_url`
(detail modal, context cards, similarity results) gets an honest "no
thumbnail" rather than a misleading one.
"""

from PIL import Image

from app.services.crop_service import _resolve_image_path, get_or_create_crop
from tests.conftest import make_deployment, make_detection, make_file, make_project


def _jpeg(tmp_path, name="frame000024.jpg", size=(640, 480)):
    path = tmp_path / name
    Image.new("RGB", size, (120, 120, 120)).save(path, "JPEG")
    return path


def test_image_detection_resolves_to_the_file_itself(db, tmp_path):
    photo = _jpeg(tmp_path, "photo.jpg")
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(db, deployment_id=dep.id, file_path=str(photo))
    d = make_detection(db, file_id=f.id)

    assert _resolve_image_path(f, d) == photo


def test_video_detection_on_the_best_frame_resolves(db, tmp_path):
    best = _jpeg(tmp_path)
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(
        db,
        deployment_id=dep.id,
        file_type="video",
        file_format="mp4",
        file_path="/fake/clip.mp4",
        best_frame_number=24,
        best_frame_path=str(best),
    )
    d = make_detection(db, file_id=f.id, frame_number=24)

    assert _resolve_image_path(f, d) == best


def test_video_detection_off_the_best_frame_resolves_to_nothing(db, tmp_path):
    """The frame this detection belongs to was never written to disk, so
    there is no honest image to return. Answering with the best frame is
    what produced the wrong-place crops."""
    best = _jpeg(tmp_path)
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(
        db,
        deployment_id=dep.id,
        file_type="video",
        file_format="mp4",
        file_path="/fake/clip.mp4",
        best_frame_number=24,
        best_frame_path=str(best),
    )
    d = make_detection(db, file_id=f.id, frame_number=144)

    assert _resolve_image_path(f, d) is None


def test_video_detection_with_no_frame_number_resolves_to_nothing(db, tmp_path):
    """A video detection that never recorded its frame cannot be shown to
    sit on the best frame, so it is not croppable either."""
    best = _jpeg(tmp_path)
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(
        db,
        deployment_id=dep.id,
        file_type="video",
        file_format="mp4",
        file_path="/fake/clip.mp4",
        best_frame_number=24,
        best_frame_path=str(best),
    )
    d = make_detection(db, file_id=f.id, frame_number=None)

    assert _resolve_image_path(f, d) is None


def test_get_or_create_crop_returns_none_off_the_best_frame(db, tmp_path):
    """End to end: the endpoint turns None into a 404, and the grid tile
    renders its no-thumbnail state instead of a picture of the wrong
    thing."""
    best = _jpeg(tmp_path)
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(
        db,
        deployment_id=dep.id,
        file_type="video",
        file_format="mp4",
        file_path="/fake/clip.mp4",
        best_frame_number=24,
        best_frame_path=str(best),
    )
    on_best = make_detection(db, file_id=f.id, frame_number=24)
    off_best = make_detection(db, file_id=f.id, frame_number=144)
    db.flush()

    assert get_or_create_crop(on_best.id, 200, db) is not None
    assert get_or_create_crop(off_best.id, 200, db) is None


# ---------------------------------------------------------------------------
# Shaped crops: the window takes the frame's shape and slides inside the
# photo, so the dashboard's wide hero fills with the scene, not with blur.
# ---------------------------------------------------------------------------

import io  # noqa: E402

from app.services.crop_service import compute_expanded_crop_region  # noqa: E402

W, H = 1000, 600


def _contains_box(region, bbox):
    left, top, right, bottom = region
    bx, by, bw, bh = bbox
    return (
        left <= bx * W and right >= (bx + bw) * W
        and top <= by * H and bottom >= (by + bh) * H
    )


def test_a_square_crop_is_unchanged_and_still_centred():
    """The Labels grid overlay is computed from a centred square; the
    shaped path must not have moved it, even off the photo's edge."""
    region = compute_expanded_crop_region(0.0, 0.4, 0.1, 0.2, W, H)

    # box 100x120 px, padding 12, side 144, centre (50, 300)
    assert region == (-22, 228, 122, 372)


def test_a_shaped_crop_has_the_asked_shape_and_holds_the_whole_box():
    bbox = (0.4, 0.3, 0.1, 0.3)
    left, top, right, bottom = compute_expanded_crop_region(
        *bbox, W, H, aspect=1.6
    )

    assert abs((right - left) / (bottom - top) - 1.6) < 0.02
    assert _contains_box((left, top, right, bottom), bbox)


def test_a_box_at_the_edge_slides_the_window_into_the_photo():
    """No blur where the photo has pixels: the frame pans inward."""
    bbox = (0.0, 0.4, 0.1, 0.2)
    region = compute_expanded_crop_region(*bbox, W, H, aspect=1.6)

    left, top, right, bottom = region
    assert left == 0 and top >= 0 and right <= W and bottom <= H
    assert _contains_box(region, bbox)


def test_a_window_larger_than_the_photo_stays_centred():
    """A tall box in a wide frame needs more width than the photo has;
    nothing can slide, so it falls back to the square's blurred edges."""
    bbox = (0.45, 0.0, 0.1, 1.0)
    left, top, right, bottom = compute_expanded_crop_region(
        *bbox, W, H, aspect=2.0
    )

    assert right - left > W
    assert abs((left + right) / 2 - 500) <= 1
    assert _contains_box((left, top, right, bottom), bbox)


def test_a_shaped_crop_comes_out_at_that_shape(db, tmp_path):
    photo = _jpeg(tmp_path, "wide.jpg", size=(1000, 600))
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(db, deployment_id=dep.id, file_path=str(photo))
    d = make_detection(db, file_id=f.id, bbox_x=0.0, bbox_y=0.4,
                       bbox_width=0.1, bbox_height=0.2)

    square = Image.open(io.BytesIO(get_or_create_crop(d.id, 200, db)))
    wide = Image.open(io.BytesIO(get_or_create_crop(d.id, 320, db, aspect=1.6)))

    assert square.size == (200, 200)
    assert wide.size == (320, 200)
