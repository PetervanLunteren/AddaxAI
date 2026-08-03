"""observation_type stays consistent with the project threshold.

A file whose only detection is below the project detection threshold has
no trusted content and must read as "blank" (the verify grid hides those
boxes). Recompute must run at ingestion (covered by the pipeline tests),
on detection changes, and when the threshold itself changes.
"""

from app.api.crud.file import (
    recalculate_observation_type,
    recalculate_observation_types_for_project,
)
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def test_sub_threshold_only_file_is_blank(db):
    project = make_project(db, name="obs-sub", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, observation_type="animal")
    make_detection(db, file_id=f.id, category="animal", confidence=0.33)

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "blank"


def test_passing_detection_keeps_animal(db):
    project = make_project(db, name="obs-pass", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, observation_type="blank")
    make_detection(db, file_id=f.id, category="animal", confidence=0.9)

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "animal"


def test_verified_sub_threshold_counts(db):
    project = make_project(db, name="obs-verified", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, observation_type="blank")
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.2, verified=True
    )

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "animal"


def test_threshold_change_flips_files_project_wide(db):
    project = make_project(db, name="obs-thresh", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, observation_type="animal")
    make_detection(db, file_id=f.id, category="animal", confidence=0.4)

    # At threshold 0.5 the 0.4 box does not pass → blank.
    changed = recalculate_observation_types_for_project(db, project.id)
    db.refresh(f)
    assert changed == 1
    assert f.observation_type == "blank"

    # Lower the threshold below the box → it passes again → animal.
    project.counting_threshold = 0.3
    db.commit()
    changed = recalculate_observation_types_for_project(db, project.id)
    db.refresh(f)
    assert changed == 1
    assert f.observation_type == "animal"


# ── Videos are only their best frame ─────────────────────────────────
#
# A video is written to disk as one frame, so a box on any other frame
# has no picture the user can open, no card in the Labels grid and no
# MaxN count. The summary follows the same rule, or it would name
# something the user cannot find and therefore cannot correct.


def _video(db, deployment_id, best_frame_number, stored="animal"):
    return make_file(
        db,
        deployment_id=deployment_id,
        file_type="video",
        file_format="mp4",
        best_frame_number=best_frame_number,
        observation_type=stored,
    )


def test_video_off_best_frame_detection_is_blank(db):
    project = make_project(db, name="vid-off", counting_threshold=0.2)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3)
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.9, frame_number=7
    )

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "blank"


def test_video_best_frame_detection_decides(db):
    project = make_project(db, name="vid-on", counting_threshold=0.2)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3, stored="blank")
    # Stronger, but on a frame nobody can open.
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.95, frame_number=7
    )
    make_detection(
        db, file_id=f.id, category="person", confidence=0.60, frame_number=3
    )

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "person"


def test_video_verified_detection_counts_on_any_frame(db):
    """The escape hatch: a human decision must never end up out of reach."""
    project = make_project(db, name="vid-verified", counting_threshold=0.2)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3, stored="blank")
    make_detection(
        db,
        file_id=f.id,
        category="animal",
        confidence=0.10,
        frame_number=7,
        verified=True,
    )

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "animal"


def test_video_without_a_best_frame_is_blank(db):
    """Frame extraction can fail. Such a video has no picture at all, so
    it has no visible surface either. The rule must not widen when data
    is missing."""
    project = make_project(db, name="vid-noframe", counting_threshold=0.2)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=None)
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.9, frame_number=4
    )

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "blank"


def test_project_threshold_change_gates_videos_too(db):
    """The project-wide recompute takes the other lane (a joined query
    rather than a per-file one), so it needs its own pin."""
    project = make_project(db, name="vid-project", counting_threshold=0.2)
    dep = make_deployment(db, project_id=project.id)
    video = _video(db, dep.id, best_frame_number=3)
    make_detection(
        db, file_id=video.id, category="animal", confidence=0.9, frame_number=7
    )
    image = make_file(db, deployment_id=dep.id, observation_type="blank")
    make_detection(db, file_id=image.id, category="animal", confidence=0.9)
    db.commit()

    recalculate_observation_types_for_project(db, project.id)
    db.refresh(video)
    db.refresh(image)

    assert video.observation_type == "blank"
    # Images have no frames and must be untouched by the gate.
    assert image.observation_type == "animal"
