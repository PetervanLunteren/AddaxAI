"""The delete-cascade configuration rule, expressed as a test.

One rule, no exceptions:

    every relationship with `delete-orphan` whose child foreign key declares
    `ON DELETE CASCADE` must set `passive_deletes=True`

Without it SQLAlchemy loads the whole child collection into memory just to
delete it row by row, which is what made re-running a large folder run take
hours. This test exists so the next person adding a fast-growing child table
cannot reintroduce that by accident.
"""

import app.models  # noqa: F401  # populates the registry
from app.db.base import Base


def _child_ondelete_actions(prop) -> set[str]:
    """ON DELETE actions on the child-side foreign key of a relationship."""
    return {
        (fk.ondelete or "").upper()
        for col in prop.remote_side
        for fk in col.foreign_keys
    }


def test_delete_cascades_are_passive() -> None:
    offenders = []
    checked = 0

    for mapper in Base.registry.mappers:
        for prop in mapper.relationships:
            if "delete-orphan" not in prop.cascade:
                continue
            if "CASCADE" not in _child_ondelete_actions(prop):
                continue
            checked += 1
            if not prop.passive_deletes:
                offenders.append(f"{mapper.class_.__name__}.{prop.key}")

    assert not offenders, (
        f"These relationships cascade deletes in Python instead of letting "
        f"the database do it: {sorted(offenders)}. Add passive_deletes=True "
        f"(not 'all'), which is safe because the FK already declares "
        f"ON DELETE CASCADE."
    )
    # Guard the guard: if the walk stops finding relationships, the rule is
    # silently unenforced.
    assert checked == 8, (
        f"expected 8 delete-orphan cascades backed by ON DELETE CASCADE, "
        f"found {checked}. If you added or removed one, update this count."
    )
