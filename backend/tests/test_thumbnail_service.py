"""A project's card picture is picked from its own pictures.

At startup for every project, and since 2026-11-04 also when an analysis
finishes for a project that has none, so a new project's card is not a
grey placeholder until the next restart.
"""

from pathlib import Path

from PIL import Image

from app.services.thumbnail_service import ensure_project_thumbnail, thumbnails_dir
from tests.conftest import (
    make_deployment,
    make_file,
    make_project,
    make_site,
    make_video_box,
)


def _clip_with_a_cover(db, tmp_path: Path):
    cover = tmp_path / "frame000060.jpg"
    Image.new("RGB", (64, 48), (20, 80, 120)).save(cover)
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path=str(tmp_path / "clip.mp4"), best_frame_number=60,
        best_frame_path=str(cover),
    )
    make_video_box(db, file_id=video.id, frame_number=60, category="animal", confidence=0.9)
    db.commit()
    return project


def test_a_project_without_a_picture_gets_one_of_its_own(db, tmp_path):
    project = _clip_with_a_cover(db, tmp_path)
    assert project.thumbnail_path is None

    ensure_project_thumbnail(db, project)

    assert project.thumbnail_path == str(thumbnails_dir() / f"{project.id}.jpg")
    assert Path(project.thumbnail_path).is_file()


def test_a_video_project_whose_cover_is_gone_still_gets_a_picture(
    db, tmp_path, monkeypatch
):
    """After importing a bundle without photos onto videos that were
    never analysed here, the stored cover is missing. The card picture
    comes from the decoded cover frame instead of staying grey."""
    import io

    from app.services import video_frame_service

    def fake_decode(path, frame_number, rate):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (20, 200, 20)).save(buf, "JPEG")
        return buf.getvalue()

    monkeypatch.setattr(video_frame_service, "decode_frame_jpeg", fake_decode)
    project = _clip_with_a_cover(db, tmp_path)
    (tmp_path / "frame000060.jpg").unlink()

    ensure_project_thumbnail(db, project)

    assert project.thumbnail_path is not None
    with Image.open(project.thumbnail_path) as card:
        assert card.getpixel((32, 24))[1] > 150


def test_a_project_that_has_a_picture_keeps_it(db, tmp_path):
    """An uploaded picture is the person's choice; the end of a run never
    replaces it."""
    project = _clip_with_a_cover(db, tmp_path)
    project.thumbnail_path = "/somewhere/project-images/chosen.jpg"
    db.commit()

    ensure_project_thumbnail(db, project)

    assert project.thumbnail_path == "/somewhere/project-images/chosen.jpg"


def test_a_project_without_detections_stays_without(db):
    project = make_project(db)
    db.commit()

    ensure_project_thumbnail(db, project)

    assert project.thumbnail_path is None
