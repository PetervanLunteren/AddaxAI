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
    project = make_project(db, name="obs-sub", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, observation_type="animal")
    make_detection(db, file_id=f.id, category="animal", confidence=0.33)

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "blank"


def test_passing_detection_keeps_animal(db):
    project = make_project(db, name="obs-pass", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, observation_type="blank")
    make_detection(db, file_id=f.id, category="animal", confidence=0.9)

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "animal"


def test_verified_sub_threshold_counts(db):
    project = make_project(db, name="obs-verified", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, observation_type="blank")
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.2, verified=True
    )

    recalculate_observation_type(db, f.id)
    db.refresh(f)

    assert f.observation_type == "animal"


def test_threshold_change_flips_files_project_wide(db):
    project = make_project(db, name="obs-thresh", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, observation_type="animal")
    make_detection(db, file_id=f.id, category="animal", confidence=0.4)

    # At threshold 0.5 the 0.4 box does not pass → blank.
    changed = recalculate_observation_types_for_project(db, project.id)
    db.refresh(f)
    assert changed == 1
    assert f.observation_type == "blank"

    # Lower the threshold below the box → it passes again → animal.
    project.detection_threshold = 0.3
    db.commit()
    changed = recalculate_observation_types_for_project(db, project.id)
    db.refresh(f)
    assert changed == 1
    assert f.observation_type == "animal"
