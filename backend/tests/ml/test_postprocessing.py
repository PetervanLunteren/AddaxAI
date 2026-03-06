"""Tests for app.ml.postprocessing utilities."""

from datetime import datetime
from unittest.mock import MagicMock

from app.ml.postprocessing import (
    build_sequence_information,
    compute_postprocessing_settings_hash,
)
from tests.conftest import make_deployment, make_file, make_project, make_site


def _make_project_mock(**overrides):
    defaults = dict(
        event_smoothing=True,
        taxonomic_rollup=True,
        taxonomic_rollup_threshold=0.65,
        independence_interval=1800,
        excluded_classes=[],
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def test_hash_deterministic():
    p = _make_project_mock()
    h1 = compute_postprocessing_settings_hash(p)
    h2 = compute_postprocessing_settings_hash(p)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_hash_changes_on_smoothing_toggle():
    p1 = _make_project_mock(event_smoothing=True)
    p2 = _make_project_mock(event_smoothing=False)
    assert compute_postprocessing_settings_hash(p1) != compute_postprocessing_settings_hash(p2)


def test_hash_changes_on_interval_change():
    p1 = _make_project_mock(independence_interval=1800)
    p2 = _make_project_mock(independence_interval=3600)
    assert compute_postprocessing_settings_hash(p1) != compute_postprocessing_settings_hash(p2)


def test_hash_excluded_classes_order_independent():
    p1 = _make_project_mock(excluded_classes=["a", "b"])
    p2 = _make_project_mock(excluded_classes=["b", "a"])
    assert compute_postprocessing_settings_hash(p1) == compute_postprocessing_settings_hash(p2)


def test_build_sequence_empty_deployment(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    result = build_sequence_information(d.id, 1800, db)
    assert result == []


def test_build_sequence_single_file(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/fake/folder")
    make_file(db, deployment_id=d.id, timestamp=datetime(2024, 1, 1, 12, 0))
    result = build_sequence_information(d.id, 1800, db)
    assert len(result) == 1
    assert "seq_id" in result[0]
    assert "file_name" in result[0]


def test_build_sequence_groups_within_interval(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/fake/folder")
    make_file(db, deployment_id=d.id, timestamp=datetime(2024, 1, 1, 12, 0, 0))
    make_file(db, deployment_id=d.id, timestamp=datetime(2024, 1, 1, 12, 10, 0))
    result = build_sequence_information(d.id, 1800, db)
    assert len(result) == 2
    # Both should have same seq_id (10 min gap < 30 min interval)
    assert result[0]["seq_id"] == result[1]["seq_id"]


def test_build_sequence_splits_on_gap(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/fake/folder")
    make_file(db, deployment_id=d.id, timestamp=datetime(2024, 1, 1, 12, 0, 0))
    make_file(db, deployment_id=d.id, timestamp=datetime(2024, 1, 1, 13, 0, 0))
    result = build_sequence_information(d.id, 1800, db)
    assert len(result) == 2
    # 60 min gap > 30 min interval → different seq_id
    assert result[0]["seq_id"] != result[1]["seq_id"]
