"""The shared file predicates behind the Labels page and the media export.

One definition of "inclusive on whole days" and of what a missing
capture time does under a date filter.
"""

from datetime import date, datetime

from sqlalchemy import select

from app.api.crud.file_filters import file_flag_clauses, tri_state
from app.models import File
from tests.conftest import make_deployment, make_file, make_project


def _ids(db, clauses) -> set[str]:
    return set(db.execute(select(File.id).where(*clauses)).scalars())


def test_date_range_is_inclusive_on_whole_days(db):
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    first = make_file(db, deployment_id=dep.id, captured_at_local=datetime(2024, 3, 1, 0, 0))
    last = make_file(db, deployment_id=dep.id, captured_at_local=datetime(2024, 3, 2, 23, 59))
    after = make_file(db, deployment_id=dep.id, captured_at_local=datetime(2024, 3, 3, 0, 0))

    got = _ids(db, file_flag_clauses(date_from=date(2024, 3, 1), date_to=date(2024, 3, 2)))

    assert got == {first.id, last.id}
    assert after.id not in got


def test_a_datetime_bound_is_used_as_given_and_its_end_day_is_whole(db):
    """The Labels page passes parsed datetimes: the start is exact, the
    end still covers its whole day."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    early = make_file(db, deployment_id=dep.id, captured_at_local=datetime(2024, 3, 1, 8, 0))
    later = make_file(db, deployment_id=dep.id, captured_at_local=datetime(2024, 3, 1, 9, 0))
    end_day = make_file(db, deployment_id=dep.id, captured_at_local=datetime(2024, 3, 2, 22, 0))

    got = _ids(
        db,
        file_flag_clauses(
            date_from=datetime(2024, 3, 1, 8, 30), date_to=datetime(2024, 3, 2, 1, 0)
        ),
    )

    assert got == {later.id, end_day.id}
    assert early.id not in got


def test_a_file_without_capture_time_drops_out_only_under_a_date_filter(db):
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    undated = make_file(db, deployment_id=dep.id)
    undated.captured_at_local = None
    db.flush()

    assert undated.id in _ids(db, file_flag_clauses())
    assert undated.id not in _ids(db, file_flag_clauses(date_from=date(2000, 1, 1)))
    assert undated.id not in _ids(db, file_flag_clauses(date_to=date(2100, 1, 1)))


def test_marks_filter_both_ways_and_none_means_no_clause(db):
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    liked = make_file(db, deployment_id=dep.id, favorited=True, verified=True)
    plain = make_file(db, deployment_id=dep.id)

    assert _ids(db, file_flag_clauses(favorited=True)) == {liked.id}
    assert _ids(db, file_flag_clauses(favorited=False)) == {plain.id}
    assert _ids(db, file_flag_clauses(verified=True)) == {liked.id}
    assert _ids(db, file_flag_clauses()) == {liked.id, plain.id}
    assert file_flag_clauses() == []


def test_tri_state_maps_the_labels_page_words():
    assert tri_state("verified", "verified", "unverified") is True
    assert tri_state("unverified", "verified", "unverified") is False
    assert tri_state("all", "verified", "unverified") is None
    assert tri_state(None, "verified", "unverified") is None
