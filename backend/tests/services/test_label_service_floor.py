"""The labels sort floor follows a user min below the project threshold.

The grid's range slider is unclamped: digging below the threshold means
"show me the low-confidence tail", so `_apply_project_threshold` lowers
`project_floor` to the user's min instead of silently cutting the
results at the threshold.
"""

from app.api.schemas.label import LabelFilters
from app.services.label_service import _apply_project_threshold
from tests.conftest import make_project


def test_floor_is_project_threshold_without_user_min(db):
    p = make_project(db, counting_threshold=0.2)
    out = _apply_project_threshold(LabelFilters(), p.id, db)
    assert out.project_floor == 0.2


def test_user_min_below_threshold_lowers_the_floor(db):
    p = make_project(db, counting_threshold=0.2)
    out = _apply_project_threshold(
        LabelFilters(min_confidence=0.05), p.id, db
    )
    assert out.project_floor == 0.05


def test_user_min_above_threshold_keeps_the_floor(db):
    """A narrower user range must not RAISE the floor: the floor's
    verified-override still lets verified low-confidence detections
    through it (the literal user min then excludes them, per the
    user-filters-are-literal rule)."""
    p = make_project(db, counting_threshold=0.2)
    out = _apply_project_threshold(
        LabelFilters(min_confidence=0.6), p.id, db
    )
    assert out.project_floor == 0.2
