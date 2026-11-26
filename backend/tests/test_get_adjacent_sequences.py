"""
Tests for get_adjacent_sequences — the optimized SQL-based navigation query.

Sequences are ordered by start_time DESC (newest first).
"Previous" = newer sequence (earlier in the list), "Next" = older sequence (later in the list).
"""

from datetime import datetime

from app.api.crud.sequence import get_adjacent_sequences
from tests.conftest import (
    make_deployment,
    make_project,
    make_sequence_with_files,
    make_site,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_project(db):
    """Create a Project → Site → Deployment chain and return (project, deployment)."""
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    return project, deployment


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_sequence(db):
    """One sequence: no previous, no next, index=0, total=1."""
    project, dep = _setup_project(db)
    ev = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 12, 0)
    )

    result = get_adjacent_sequences(db, ev.id, project.id)

    assert result["previous_id"] is None
    assert result["next_id"] is None
    assert result["next_unconfirmed_id"] is None
    assert result["current_index"] == 0
    assert result["total_count"] == 1


def test_three_sequences_middle(db):
    """Three sequences, query the middle one — should have both previous and next."""
    project, dep = _setup_project(db)

    # Newest first in DESC order: ev_new > ev_mid > ev_old
    ev_old = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 10, 0)
    )
    ev_mid = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 12, 0)
    )
    ev_new = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 14, 0)
    )

    result = get_adjacent_sequences(db, ev_mid.id, project.id)

    assert result["previous_id"] == ev_new.id  # newer = previous in DESC
    assert result["next_id"] == ev_old.id  # older = next in DESC
    assert result["current_index"] == 1
    assert result["total_count"] == 3


def test_first_sequence_no_previous(db):
    """Newest sequence: no previous, next points to the second one."""
    project, dep = _setup_project(db)

    ev_old = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 10, 0)
    )
    ev_new = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 14, 0)
    )

    result = get_adjacent_sequences(db, ev_new.id, project.id)

    assert result["previous_id"] is None
    assert result["next_id"] == ev_old.id
    assert result["current_index"] == 0
    assert result["total_count"] == 2


def test_last_sequence_no_next(db):
    """Oldest sequence: previous points to the newer one, no next."""
    project, dep = _setup_project(db)

    ev_old = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 10, 0)
    )
    ev_new = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 14, 0)
    )

    result = get_adjacent_sequences(db, ev_old.id, project.id)

    assert result["previous_id"] == ev_new.id
    assert result["next_id"] is None
    assert result["current_index"] == 1
    assert result["total_count"] == 2


def test_next_unconfirmed_skips_confirmed(db):
    """
    Three sequences: A (newest), B (middle, confirmed), C (oldest, unconfirmed).
    Query A → next_unconfirmed should be C, skipping confirmed B. Keys on
    Sequence.confirmed, not file verification.
    """
    project, dep = _setup_project(db)

    ev_c = make_sequence_with_files(
        db, deployment_id=dep.id,
        sequence_start_local=datetime(2024, 6, 1, 10, 0),
    )
    ev_b = make_sequence_with_files(
        db, deployment_id=dep.id,
        sequence_start_local=datetime(2024, 6, 1, 12, 0),
    )
    ev_b.confirmed = True
    db.commit()
    ev_a = make_sequence_with_files(
        db, deployment_id=dep.id,
        sequence_start_local=datetime(2024, 6, 1, 14, 0),
    )

    result = get_adjacent_sequences(db, ev_a.id, project.id)

    assert result["next_unconfirmed_id"] == ev_c.id


def test_next_unconfirmed_none_remaining(db):
    """All sequences after current are confirmed → next_unconfirmed_id=None."""
    project, dep = _setup_project(db)

    ev_old = make_sequence_with_files(
        db, deployment_id=dep.id,
        sequence_start_local=datetime(2024, 6, 1, 10, 0),
    )
    ev_old.confirmed = True
    db.commit()
    ev_new = make_sequence_with_files(
        db, deployment_id=dep.id,
        sequence_start_local=datetime(2024, 6, 1, 14, 0),
    )

    result = get_adjacent_sequences(db, ev_new.id, project.id)

    assert result["next_unconfirmed_id"] is None


def test_nonexistent_sequence(db):
    """Query with a bogus sequence_id returns zeros/nulls."""
    project, dep = _setup_project(db)
    make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=datetime(2024, 6, 1, 12, 0)
    )

    result = get_adjacent_sequences(db, "nonexistent-id", project.id)

    assert result["previous_id"] is None
    assert result["next_id"] is None
    assert result["next_unconfirmed_id"] is None
    assert result["current_index"] == 0
    assert result["total_count"] == 0


def test_same_start_time_tiebreak(db):
    """
    Two sequences with identical start_time but different IDs.
    The one with the higher ID should appear as "previous" (newer in DESC order)
    when querying the one with the lower ID.
    """
    project, dep = _setup_project(db)

    ts = datetime(2024, 6, 1, 12, 0)
    # Use fixed IDs so we control the sort order
    ev_lo = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=ts, sequence_id="aaa-lo"
    )
    ev_hi = make_sequence_with_files(
        db, deployment_id=dep.id, sequence_start_local=ts, sequence_id="zzz-hi"
    )

    # DESC order: ev_hi (same time, higher id) comes first → is "previous" to ev_lo
    result = get_adjacent_sequences(db, ev_lo.id, project.id)
    assert result["previous_id"] == ev_hi.id
    assert result["next_id"] is None
    assert result["current_index"] == 1

    # From ev_hi's perspective, ev_lo is "next"
    result2 = get_adjacent_sequences(db, ev_hi.id, project.id)
    assert result2["previous_id"] is None
    assert result2["next_id"] == ev_lo.id
    assert result2["current_index"] == 0
