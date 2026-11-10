"""The project cover photo comes from a real detection.

`_auto_select_for_project` picks one of the ten most confident animal
boxes. A rejected box (label "false detection" or any other non-label
class) is not an animal the project found, however confident the
detector was about the stump, so it is not a candidate.
"""

from pathlib import Path

from PIL import Image

from app.services.thumbnail_service import _auto_select_for_project
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
)


def _jpeg(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path, format="JPEG")
    return path


def test_a_rejected_box_never_becomes_the_cover(db, tmp_path):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id, folder_path=str(tmp_path / "src"))

    stump = _jpeg(tmp_path / "src" / "stump.jpg", (0, 0, 0))
    wolf = _jpeg(tmp_path / "src" / "wolf.jpg", (255, 255, 255))
    f_stump = make_file(db, deployment_id=dep.id, file_path=str(stump))
    f_wolf = make_file(db, deployment_id=dep.id, file_path=str(wolf))
    # The stump outranks the wolf on detector confidence, and would have
    # been the pick before rejected boxes were excluded.
    make_detection(db, file_id=f_stump.id, confidence=0.99, label="false detection")
    make_detection(db, file_id=f_wolf.id, confidence=0.5, label="wolf")
    db.commit()

    for _ in range(5):
        _auto_select_for_project(db, project, tmp_path / "thumbs")
        db.refresh(project)
        with Image.open(project.thumbnail_path) as im:
            assert im.getpixel((0, 0)) == (255, 255, 255)


def test_only_rejected_boxes_means_no_cover(db, tmp_path):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id, folder_path=str(tmp_path / "src"))
    stump = _jpeg(tmp_path / "src" / "stump.jpg", (0, 0, 0))
    f_stump = make_file(db, deployment_id=dep.id, file_path=str(stump))
    make_detection(db, file_id=f_stump.id, confidence=0.99, label="false detection")
    db.commit()

    _auto_select_for_project(db, project, tmp_path / "thumbs")
    db.refresh(project)
    assert project.thumbnail_path is None
