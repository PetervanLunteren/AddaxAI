"""Which detector categories are wildlife.

One rule, in one place: person and vehicle are not wildlife, every other
category is. AddaxAI was written when a detector emitted only animal,
person or vehicle, so a great many places asked `category == "animal"`
when they meant this, and a shark was then neither wildlife nor a person.
"""


def test_person_and_vehicle_are_the_only_categories_that_are_not_wildlife():
    """One rule for every detector: AddaxAI was written when a detector
    said only animal, person or vehicle, and a shark was then neither
    wildlife nor a person, so it fell out of every wildlife statistic."""
    from app.ml.label_exclusion import is_wildlife

    assert is_wildlife("animal")
    assert is_wildlife("elasmobranch")
    assert is_wildlife("fish")
    assert not is_wildlife("person")
    assert not is_wildlife("vehicle")
    assert not is_wildlife(None)


def test_the_two_wildlife_lanes_agree(db):
    """The query predicate and the in-memory one must never disagree, the
    same parity the visibility rule keeps."""
    from app.ml.label_exclusion import is_wildlife, is_wildlife_category
    from app.models import Detection
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
        make_project,
        make_site,
    )

    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=dep.id)
    categories = ["animal", "person", "vehicle", "elasmobranch", "fish"]
    for cat in categories:
        make_detection(db, file_id=f.id, category=cat)
    db.commit()

    in_sql = {
        d.category
        for d in db.query(Detection)
        .filter(is_wildlife_category(Detection.category))
        .all()
    }
    in_python = {c for c in categories if is_wildlife(c)}
    assert in_sql == in_python == {"animal", "elasmobranch", "fish"}
