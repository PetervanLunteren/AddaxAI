"""Tests for the JSON-twin event clusterer used by the Timelapse runner."""

from datetime import datetime, timedelta

from app.services.event_clustering import cluster_entries_into_events


def _t(minutes: int) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0) + timedelta(minutes=minutes)


def test_single_folder_single_cluster():
    entries = [
        ("a/img1.jpg", _t(0)),
        ("a/img2.jpg", _t(1)),
        ("a/img3.jpg", _t(2)),
    ]
    clusters = cluster_entries_into_events(entries, independence_interval=600)
    assert len(clusters) == 1
    assert [e[0] for e in clusters[0]] == ["a/img1.jpg", "a/img2.jpg", "a/img3.jpg"]


def test_time_gap_splits_cluster():
    entries = [
        ("a/img1.jpg", _t(0)),
        ("a/img2.jpg", _t(1)),
        ("a/img3.jpg", _t(30)),
    ]
    clusters = cluster_entries_into_events(entries, independence_interval=600)
    assert len(clusters) == 2
    assert [e[0] for e in clusters[0]] == ["a/img1.jpg", "a/img2.jpg"]
    assert [e[0] for e in clusters[1]] == ["a/img3.jpg"]


def test_different_folders_never_merge():
    entries = [
        ("a/img1.jpg", _t(0)),
        ("b/img1.jpg", _t(0)),
    ]
    clusters = cluster_entries_into_events(entries, independence_interval=86400)
    assert len(clusters) == 2


def test_skips_none_timestamps():
    entries = [
        ("a/img1.jpg", _t(0)),
        ("a/img2.jpg", None),
        ("a/img3.jpg", _t(1)),
    ]
    clusters = cluster_entries_into_events(entries, independence_interval=600)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_deterministic_folder_order():
    entries = [
        ("z/img.jpg", _t(0)),
        ("a/img.jpg", _t(0)),
        ("m/img.jpg", _t(0)),
    ]
    clusters = cluster_entries_into_events(entries, independence_interval=600)
    folders = [str(e[0][0]).split("/")[0] for e in clusters]
    assert folders == ["a", "m", "z"]
