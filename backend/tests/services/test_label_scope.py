"""Suggestions run inside the scope and ignore the selection.

Scope is where someone works: sites and dates. Selection is what they
look at: label, verification, confidence, flagged, liked. The
suggestions vote keeps the first and drops the second, because a label
or Verified filter removes voters by the very thing they vote on. With
every filter applied (until 2026-10-29) "Verified" emptied the review
into a false "All suggestions reviewed"; with none applied (until
2026-11-18) a team could not review one site each.

The pill (cohorts) and the review (suggestions sort) must select the
same pool, or their numbers disagree. Both get their filters from
`LabelFilters.scope_only`, and the last test pins that they match.
"""

from datetime import datetime

from app.api.schemas.label import LabelFilters, SortRequest
from app.services.label_service import _build_cohort_params, _build_sort_params
from tests.conftest import make_project

_EVERY_FILTER = LabelFilters(
    labels=["canis"],
    site_ids=["site-a", "null"],
    date_from=datetime(2024, 6, 1),
    date_to=datetime(2024, 9, 30),
    min_confidence=0.5,
    max_confidence=0.9,
    min_label_confidence=0.4,
    max_label_confidence=0.8,
    category="animal",
    verified=False,
    flagged="flagged",
    favorited="favorited",
)

_SCOPE_AS_SENT = {
    "site_ids": ["site-a", "null"],
    "date_from": "2024-06-01 00:00:00",
    "date_to": "2024-09-30 23:59:59.999999",
}


def test_scope_only_keeps_sites_and_dates_and_nothing_else():
    scope = _EVERY_FILTER.scope_only()
    assert scope == LabelFilters(
        site_ids=["site-a", "null"],
        date_from=datetime(2024, 6, 1),
        date_to=datetime(2024, 9, 30),
    )


def test_the_suggestions_sort_drops_the_selection_a_client_sends(db):
    """Enforced on the server: a client that sends every filter still
    gets the vote over the whole scope."""
    p = make_project(db, counting_threshold=0.2)
    params = _build_sort_params(
        p.id, SortRequest(filters=_EVERY_FILTER, sort="suggestions"), db
    )
    assert params["filters"] == {**_SCOPE_AS_SENT, "project_floor": 0.2}


def test_a_user_min_does_not_lower_the_floor_for_suggestions(db):
    """The confidence slider is selection, so it cannot dig the vote
    below the project threshold the way it does for the grid."""
    p = make_project(db, counting_threshold=0.2)
    params = _build_sort_params(
        p.id,
        SortRequest(filters=LabelFilters(min_confidence=0.05), sort="suggestions"),
        db,
    )
    assert params["filters"] == {"project_floor": 0.2}


def test_the_other_sorts_keep_every_filter(db):
    p = make_project(db, counting_threshold=0.2)
    for sort in ("similarity", "events"):
        params = _build_sort_params(
            p.id, SortRequest(filters=_EVERY_FILTER, sort=sort), db
        )
        assert params["filters"] == {
            **_SCOPE_AS_SENT,
            "labels": ["canis"],
            "min_confidence": 0.5,
            "max_confidence": 0.9,
            "min_label_confidence": 0.4,
            "max_label_confidence": 0.8,
            "project_floor": 0.2,
            "category": "animal",
            "verified": False,
            "flagged": "flagged",
            "favorited": "favorited",
        }


def test_the_pill_and_the_review_select_the_same_pool(db):
    p = make_project(db, counting_threshold=0.2)
    review = _build_sort_params(
        p.id, SortRequest(filters=_EVERY_FILTER, sort="suggestions"), db
    )
    pill = _build_cohort_params(p.id, 8, 200, _EVERY_FILTER, db)
    assert pill["filters"] == review["filters"]
    assert (pill["min_count"], pill["max_cohorts"]) == (8, 200)


def test_without_a_scope_the_pill_sends_only_the_floor(db):
    p = make_project(db, counting_threshold=0.2)
    pill = _build_cohort_params(p.id, 8, 200, LabelFilters(), db)
    assert pill["filters"] == {"project_floor": 0.2}
