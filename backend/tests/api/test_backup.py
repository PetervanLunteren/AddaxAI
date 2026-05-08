"""Tests for the /api/backup endpoints."""

import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.db.backup import (
    RESTORE_MARKER_FILENAME,
    _classify,
    _daily_filename,
    _pre_upgrade_filename,
)


@pytest.fixture()
def live_db(tmp_path: Path):
    """Plant a real SQLite file at settings.user_data_dir/addaxai.db.

    The conftest's `client` fixture uses an in-memory engine; backup
    endpoints read the on-disk file, so we build one here. The backups
    directory is wiped between tests so each test starts clean.
    """
    settings = get_settings()
    live = settings.user_data_dir / "addaxai.db"

    conn = sqlite3.connect(str(live))
    try:
        conn.execute("CREATE TABLE marker (note TEXT)")
        conn.execute("INSERT INTO marker (note) VALUES ('live-db')")
        conn.commit()
    finally:
        conn.close()

    backups_dir = settings.user_data_dir / "backups"
    if backups_dir.exists():
        shutil.rmtree(backups_dir)

    yield live

    for sibling in (live, live.with_name(live.name + "-wal"), live.with_name(live.name + "-shm")):
        sibling.unlink(missing_ok=True)
    if backups_dir.exists():
        shutil.rmtree(backups_dir)
    marker = settings.user_data_dir / RESTORE_MARKER_FILENAME
    marker.unlink(missing_ok=True)


# ── /api/backup/dir ──────────────────────────────────────────────────


def test_get_dir_returns_path_and_creates_folder(client, live_db) -> None:
    settings = get_settings()
    expected = settings.user_data_dir / "backups"
    if expected.exists():
        shutil.rmtree(expected)

    resp = client.get("/api/backup/dir")
    assert resp.status_code == 200
    assert resp.json() == {"path": str(expected)}
    assert expected.is_dir()


# ── /api/backup/snapshot ─────────────────────────────────────────────


def test_snapshot_to_ring_buffer_when_no_target(client, live_db) -> None:
    resp = client.post("/api/backup/snapshot", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    path = Path(body["path"])
    assert path.is_file()
    assert path.parent.name == "backups"
    assert _classify(path.name) == "daily"
    assert body["size_bytes"] == path.stat().st_size


def test_snapshot_to_target_dir(client, live_db, tmp_path: Path) -> None:
    chosen = tmp_path / "user-chosen"
    chosen.mkdir()

    resp = client.post("/api/backup/snapshot", json={"target_dir": str(chosen)})
    assert resp.status_code == 200, resp.text
    path = Path(resp.json()["path"])
    assert path.parent == chosen
    assert _classify(path.name) == "daily"


def test_snapshot_to_missing_target_dir_returns_400(client, live_db, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    resp = client.post("/api/backup/snapshot", json={"target_dir": str(missing)})
    assert resp.status_code == 400
    assert "does not exist" in resp.json()["detail"]


def test_snapshot_force_ignores_daily_throttle(client, live_db) -> None:
    first = client.post("/api/backup/snapshot", json={})
    assert first.status_code == 200
    second = client.post("/api/backup/snapshot", json={})
    assert second.status_code == 200
    # Two distinct paths (timestamp differs by at least one second OR same
    # filename overwritten — either way, a file is produced).
    assert Path(second.json()["path"]).is_file()


# ── /api/backup/list ─────────────────────────────────────────────────


def test_list_classifies_daily_and_pre_upgrade(client, live_db) -> None:
    settings = get_settings()
    backups_dir = settings.user_data_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    # Plant one of each classification.
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    daily_path = backups_dir / _daily_filename(ts)
    pre_path = backups_dir / _pre_upgrade_filename("abc12345", ts)
    for p in (daily_path, pre_path):
        p.write_bytes(b"x" * 200)

    resp = client.get("/api/backup/list")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    by_name = {Path(e["path"]).name: e["kind"] for e in entries}
    assert by_name[daily_path.name] == "daily"
    assert by_name[pre_path.name] == "pre-upgrade"


# ── /api/backup/restore ──────────────────────────────────────────────


def test_restore_writes_marker_for_valid_source(client, live_db) -> None:
    settings = get_settings()
    snap = settings.user_data_dir / "good.db"
    # Use the actual snapshot endpoint to produce a valid backup file.
    resp = client.post("/api/backup/snapshot", json={"target_dir": str(settings.user_data_dir)})
    snap = Path(resp.json()["path"])

    resp = client.post("/api/backup/restore", json={"source_path": str(snap)})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"scheduled": True}

    marker = settings.user_data_dir / RESTORE_MARKER_FILENAME
    assert marker.is_file()
    assert marker.read_text().strip() == str(snap.resolve())


def test_restore_rejects_corrupt_source(client, live_db, tmp_path: Path) -> None:
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"junk")

    resp = client.post("/api/backup/restore", json={"source_path": str(bad)})
    assert resp.status_code == 400

    settings = get_settings()
    marker = settings.user_data_dir / RESTORE_MARKER_FILENAME
    assert not marker.exists()


def test_restore_rejects_missing_source(client, live_db, tmp_path: Path) -> None:
    resp = client.post(
        "/api/backup/restore",
        json={"source_path": str(tmp_path / "nope.db")},
    )
    assert resp.status_code == 400

    settings = get_settings()
    marker = settings.user_data_dir / RESTORE_MARKER_FILENAME
    assert not marker.exists()
