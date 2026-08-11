"""Tests for `app.db.backup`."""

import os
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.config import Settings
from app.db.backup import (
    DAILY_BACKUP_KEEP,
    RESTORE_MARKER_FILENAME,
    BackupInvalidError,
    _classify,
    _daily_filename,
    _manual_note,
    _pre_upgrade_filename,
    _prune_ring_buffer,
    _slugify_note,
    consume_restore_marker,
    force_ring_buffer_backup,
    list_ring_buffer,
    manual_backup_filename,
    manual_snapshot,
    pre_upgrade_backup,
    restore_db,
    ring_buffer_backup,
    schedule_restore,
    snapshot_db,
    validate_backup,
)


def _make_sqlite(path: Path) -> None:
    """Create a tiny but valid AddaxAI-shaped SQLite database at `path`.

    `alembic_version` is part of "valid": `validate_backup` requires it,
    because a database without one predates the 2026-05-08 alembic
    wiring and `init_db` would refuse it on the next launch. Restoring
    such a file would drop the user straight back on the startup error
    page with no idea the file they picked was the problem.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32))")
        conn.execute("INSERT INTO alembic_version VALUES ('3c4d5e6f7a8b')")
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t (x) VALUES (1)")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """A Settings pointing at a fresh tmp user-data dir with a live DB file."""
    db_path = tmp_path / "addaxai.db"
    _make_sqlite(db_path)
    return Settings(
        user_data_dir=tmp_path,
        database_url=f"sqlite:///{db_path}",
        models_dir=tmp_path / "models",
    )


# ── snapshot_db ──────────────────────────────────────────────────────


def test_snapshot_db_produces_valid_file(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_sqlite(src)
    dst = tmp_path / "out" / "snap.db"

    snapshot_db(src, dst)

    assert dst.is_file()
    validate_backup(dst)
    # Snapshot should contain the same data as src.
    with sqlite3.connect(str(dst)) as conn:
        rows = conn.execute("SELECT x FROM t").fetchall()
    assert rows == [(1,)]


def test_snapshot_db_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        snapshot_db(tmp_path / "nope.db", tmp_path / "out.db")


# ── validate_backup ──────────────────────────────────────────────────


def test_validate_backup_accepts_snapshotted_live_db(tmp_path: Path) -> None:
    src = tmp_path / "live.db"
    _make_sqlite(src)
    dst = tmp_path / "snap.db"
    snapshot_db(src, dst)
    validate_backup(dst)  # must not raise


def test_validate_backup_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(BackupInvalidError, match="Not a file"):
        validate_backup(tmp_path / "nope.db")


def test_validate_backup_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    empty.touch()
    with pytest.raises(BackupInvalidError, match="too small"):
        validate_backup(empty)


def test_validate_backup_rejects_plain_text(tmp_path: Path) -> None:
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not a sqlite database, but it's well over 100 bytes long " * 4)
    with pytest.raises(BackupInvalidError):
        validate_backup(junk)


def test_validate_backup_rejects_truncated(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_sqlite(src)
    truncated = tmp_path / "trunc.db"
    truncated.write_bytes(src.read_bytes()[:200])
    with pytest.raises(BackupInvalidError):
        validate_backup(truncated)


# ── ring_buffer_backup (daily throttle) ──────────────────────────────


def test_ring_buffer_backup_runs_once_per_day(tmp_settings: Settings) -> None:
    first = ring_buffer_backup(tmp_settings)
    assert first is not None
    assert first.is_file()

    # Same UTC day → no-op.
    second = ring_buffer_backup(tmp_settings)
    assert second is None


def test_force_ring_buffer_backup_ignores_throttle(tmp_settings: Settings) -> None:
    first = ring_buffer_backup(tmp_settings)
    assert first is not None
    second = force_ring_buffer_backup(tmp_settings)
    assert second.is_file()


# ── _prune_ring_buffer ───────────────────────────────────────────────


def _make_dummy_backup(path: Path, mtime: float) -> None:
    """Create a fake backup file with a controlled mtime."""
    path.write_bytes(b"x" * 200)
    os.utime(path, (mtime, mtime))


def test_prune_keeps_newest_daily(tmp_path: Path) -> None:
    base_ts = time.time()
    for i in range(7):
        ts_str = datetime.fromtimestamp(base_ts - i * 86400, tz=UTC).strftime(
            "%Y-%m-%dT%H%M%SZ"
        )
        _make_dummy_backup(tmp_path / _daily_filename(ts_str), mtime=base_ts - i * 86400)

    deleted = _prune_ring_buffer(tmp_path, keep=DAILY_BACKUP_KEEP)

    assert len(deleted) == 2
    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert len(survivors) == DAILY_BACKUP_KEEP


def test_prune_does_not_touch_pre_upgrade(tmp_path: Path) -> None:
    base_ts = time.time()
    pre_upgrade_files = []
    for i in range(3):
        ts_str = datetime.fromtimestamp(base_ts - i * 86400, tz=UTC).strftime(
            "%Y-%m-%dT%H%M%SZ"
        )
        p = tmp_path / _pre_upgrade_filename(f"abc{i}", ts_str)
        _make_dummy_backup(p, mtime=base_ts - i * 86400)
        pre_upgrade_files.append(p)

    # Add many daily files so prune runs.
    for i in range(8):
        ts_str = datetime.fromtimestamp(base_ts - (i + 100) * 86400, tz=UTC).strftime(
            "%Y-%m-%dT%H%M%SZ"
        )
        _make_dummy_backup(tmp_path / _daily_filename(ts_str), mtime=base_ts - (i + 100) * 86400)

    _prune_ring_buffer(tmp_path, keep=DAILY_BACKUP_KEEP)

    for p in pre_upgrade_files:
        assert p.exists(), f"Pre-upgrade backup {p.name} was incorrectly pruned"


# ── pre_upgrade_backup ───────────────────────────────────────────────


def test_pre_upgrade_backup_writes_classified_file(tmp_settings: Settings) -> None:
    path = pre_upgrade_backup(tmp_settings, rev="9c173fff3bcd")
    assert path.is_file()
    assert _classify(path.name) == "pre-upgrade"
    assert "pre-upgrade-9c173fff" in path.name


def test_pre_upgrade_backup_handles_unknown_rev(tmp_settings: Settings) -> None:
    path = pre_upgrade_backup(tmp_settings, rev=None)
    assert path is not None
    assert _classify(path.name) == "pre-upgrade"
    assert "pre-upgrade-unknown" in path.name


def test_pre_upgrade_backup_skips_a_revision_it_already_has(
    tmp_settings: Settings,
) -> None:
    """One snapshot per revision, however many times we launch.

    A second copy of the same unmigrated database is worth nothing and
    costs a full DB copy. Without this, repeated launches at the same
    revision (a `uvicorn --reload` storm, or a user clicking Retry on
    the startup error page) push the real pre-upgrade snapshot out of
    the ring buffer with identical duplicates, and that is the one
    snapshot that matters on the day a migration eats data.
    """
    first = pre_upgrade_backup(tmp_settings, rev="9c173fff3bcd")
    assert first is not None

    assert pre_upgrade_backup(tmp_settings, rev="9c173fff3bcd") is None

    backups = list((tmp_settings.user_data_dir / "backups").glob("*.db"))
    assert [p.name for p in backups] == [first.name]

    # A different revision is a different moment in the database's life,
    # so it still gets its own snapshot.
    second = pre_upgrade_backup(tmp_settings, rev="2540e6edbee2")
    assert second is not None and second.name != first.name


# ── manual snapshots and note slugs ──────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Camera Trap 3", "camera-trap-3"),
        ("  Überraschung!  ", "uberraschung"),
        ("wifi 2.4GHz", "wifi-2-4ghz"),
        ("---", None),
        ("", None),
        (None, None),
        ("🦊🦊", None),
        ("a" * 50, "a" * 40),
        # Truncation landing on a hyphen must not leave one dangling.
        ("a" * 39 + " bc", "a" * 39),
    ],
)
def test_slugify_note(raw: str | None, expected: str | None) -> None:
    assert _slugify_note(raw) == expected


def test_slugify_note_is_idempotent() -> None:
    for raw in ("Camera Trap 3", "wifi 2.4GHz", "a" * 50):
        once = _slugify_note(raw)
        assert _slugify_note(once) == once


def test_manual_backup_filename_round_trip() -> None:
    plain = manual_backup_filename()
    noted = manual_backup_filename("Before the Big Run!")
    assert _classify(plain) == "manual"
    assert _classify(noted) == "manual"
    assert _manual_note(plain) is None
    assert _manual_note(noted) == "before-the-big-run"


def test_manual_snapshot_with_note_lists_note(tmp_settings: Settings) -> None:
    path = manual_snapshot(tmp_settings, note="My Note")
    assert path.is_file()
    notes = {e.path.name: e.note for e in list_ring_buffer(tmp_settings)}
    assert notes[path.name] == "my-note"


def test_manual_snapshot_without_note_keeps_legacy_shape(
    tmp_settings: Settings,
) -> None:
    """Regression guard for the note-less wipe caller in the lifespan."""
    path = manual_snapshot(tmp_settings)
    assert re.fullmatch(
        r"addaxai-manual-\d{4}-\d{2}-\d{2}T\d{6}Z\.db", path.name
    )
    entry = next(
        e for e in list_ring_buffer(tmp_settings) if e.path.name == path.name
    )
    assert entry.kind == "manual"
    assert entry.note is None


def test_hand_renamed_manual_backup_stays_listed(tmp_settings: Settings) -> None:
    """The parse language is wider than what we generate, on purpose:
    a file the user renamed by hand stays listable while it stays
    lowercase."""
    backups_dir = tmp_settings.user_data_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    renamed = backups_dir / "addaxai-manual-2026-01-01T000000Z-a-.db"
    renamed.write_bytes(b"x" * 200)

    entries = {e.path.name: e for e in list_ring_buffer(tmp_settings)}
    assert entries[renamed.name].kind == "manual"
    assert entries[renamed.name].note == "a-"


def test_prune_does_not_touch_manual(tmp_path: Path) -> None:
    base_ts = time.time()
    manual_files = []
    for i in range(3):
        p = tmp_path / f"addaxai-manual-2026-01-0{i + 1}T000000Z-note-{i}.db"
        _make_dummy_backup(p, mtime=base_ts - i * 86400)
        manual_files.append(p)

    # Add many daily files so prune runs.
    for i in range(8):
        ts_str = datetime.fromtimestamp(base_ts - (i + 100) * 86400, tz=UTC).strftime(
            "%Y-%m-%dT%H%M%SZ"
        )
        _make_dummy_backup(tmp_path / _daily_filename(ts_str), mtime=base_ts - (i + 100) * 86400)

    _prune_ring_buffer(tmp_path, keep=DAILY_BACKUP_KEEP)

    for p in manual_files:
        assert p.exists(), f"Manual backup {p.name} was incorrectly pruned"


def test_schedule_restore_accepts_noted_manual(tmp_settings: Settings) -> None:
    snap = manual_snapshot(tmp_settings, note="before big run")
    marker = schedule_restore(tmp_settings, snap)
    assert marker.read_text().strip() == str(snap.resolve())


# ── list_ring_buffer ─────────────────────────────────────────────────


def test_list_ring_buffer_classifies_and_sorts_newest_first(
    tmp_settings: Settings,
) -> None:
    daily = ring_buffer_backup(tmp_settings)
    assert daily is not None
    # ensure mtimes are distinct enough for the sort
    time.sleep(0.01)
    pre_up = pre_upgrade_backup(tmp_settings, rev="abc12345")

    entries = list_ring_buffer(tmp_settings)

    kinds = {e.path.name: e.kind for e in entries}
    assert kinds[daily.name] == "daily"
    assert kinds[pre_up.name] == "pre-upgrade"
    # Newest first.
    assert entries[0].created_utc >= entries[-1].created_utc


def test_list_ring_buffer_ignores_unrelated_files(tmp_settings: Settings) -> None:
    backups_dir = tmp_settings.user_data_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    (backups_dir / "stray.txt").write_text("hello")
    (backups_dir / "addaxai-not-a-real-backup.db").write_bytes(b"x" * 200)
    # Uppercase in a note slug is outside the parse language.
    (backups_dir / "addaxai-manual-2026-01-01T000000Z-CAPS.db").write_bytes(b"x" * 200)
    ring_buffer_backup(tmp_settings)

    entries = list_ring_buffer(tmp_settings)
    names = {e.path.name for e in entries}
    assert "stray.txt" not in names
    assert "addaxai-not-a-real-backup.db" not in names
    assert "addaxai-manual-2026-01-01T000000Z-CAPS.db" not in names


# ── restore_db ───────────────────────────────────────────────────────


def test_restore_db_swaps_and_creates_safety_snapshot(tmp_settings: Settings) -> None:
    # Build a "different" source DB so we can verify the swap happened.
    other = tmp_settings.user_data_dir / "other.db"
    _make_sqlite(other)
    conn = sqlite3.connect(str(other))
    try:
        conn.execute("CREATE TABLE marker (note TEXT)")
        conn.execute("INSERT INTO marker (note) VALUES ('from-source')")
        conn.commit()
    finally:
        conn.close()

    backups_before = {p.name for p in (tmp_settings.user_data_dir / "backups").iterdir()} \
        if (tmp_settings.user_data_dir / "backups").is_dir() else set()

    restore_db(tmp_settings, other)

    # Live DB now has the source's content.
    live = tmp_settings.user_data_dir / "addaxai.db"
    with sqlite3.connect(str(live)) as conn:
        row = conn.execute("SELECT note FROM marker").fetchone()
    assert row == ("from-source",)

    # A pre-restore safety snapshot of the original live DB landed in backups/.
    backups_after = {p.name for p in (tmp_settings.user_data_dir / "backups").iterdir()}
    new_files = backups_after - backups_before
    assert any(_classify(name) == "pre-restore" for name in new_files)


def test_restore_db_rejects_invalid_source(tmp_settings: Settings) -> None:
    bad = tmp_settings.user_data_dir / "bad.db"
    bad.write_bytes(b"junk")
    with pytest.raises(BackupInvalidError):
        restore_db(tmp_settings, bad)


# ── schedule_restore + consume_restore_marker ────────────────────────


def test_schedule_restore_writes_marker_after_validation(tmp_settings: Settings) -> None:
    snap = tmp_settings.user_data_dir / "snap.db"
    snapshot_db(tmp_settings.user_data_dir / "addaxai.db", snap)

    marker = schedule_restore(tmp_settings, snap)

    assert marker.is_file()
    assert marker.read_text().strip() == str(snap.resolve())


def test_schedule_restore_rejects_invalid_source(tmp_settings: Settings) -> None:
    bad = tmp_settings.user_data_dir / "bad.db"
    bad.write_bytes(b"x")
    with pytest.raises(BackupInvalidError):
        schedule_restore(tmp_settings, bad)
    assert not (tmp_settings.user_data_dir / RESTORE_MARKER_FILENAME).exists()


def test_consume_restore_marker_no_op_when_absent(tmp_settings: Settings) -> None:
    consume_restore_marker(tmp_settings)  # must not raise


def test_consume_restore_marker_swaps_db_and_consumes(tmp_settings: Settings) -> None:
    # Build a "different" source DB
    other = tmp_settings.user_data_dir / "other.db"
    _make_sqlite(other)
    conn = sqlite3.connect(str(other))
    try:
        conn.execute("CREATE TABLE marker (note TEXT)")
        conn.execute("INSERT INTO marker (note) VALUES ('from-source')")
        conn.commit()
    finally:
        conn.close()

    schedule_restore(tmp_settings, other)
    assert (tmp_settings.user_data_dir / RESTORE_MARKER_FILENAME).is_file()

    consume_restore_marker(tmp_settings)

    # Marker consumed.
    assert not (tmp_settings.user_data_dir / RESTORE_MARKER_FILENAME).exists()

    # Live DB swapped.
    live = tmp_settings.user_data_dir / "addaxai.db"
    with sqlite3.connect(str(live)) as conn:
        row = conn.execute("SELECT note FROM marker").fetchone()
    assert row == ("from-source",)


def test_consume_restore_marker_self_cleans_on_missing_source(
    tmp_settings: Settings,
) -> None:
    # Hand-craft a marker pointing at a non-existent file (bypasses
    # schedule_restore's validation).
    marker = tmp_settings.user_data_dir / RESTORE_MARKER_FILENAME
    marker.write_text(str(tmp_settings.user_data_dir / "nope.db"))

    consume_restore_marker(tmp_settings)

    # Marker consumed even though the source was bad.
    assert not marker.exists()
    # Live DB untouched.
    assert (tmp_settings.user_data_dir / "addaxai.db").is_file()


def test_consume_restore_marker_self_cleans_on_corrupt_source(
    tmp_settings: Settings,
) -> None:
    bad = tmp_settings.user_data_dir / "bad.db"
    bad.write_bytes(b"x")
    marker = tmp_settings.user_data_dir / RESTORE_MARKER_FILENAME
    marker.write_text(str(bad))

    consume_restore_marker(tmp_settings)

    assert not marker.exists()


def test_consume_restore_marker_self_cleans_on_empty_marker(
    tmp_settings: Settings,
) -> None:
    marker = tmp_settings.user_data_dir / RESTORE_MARKER_FILENAME
    marker.write_text("")

    consume_restore_marker(tmp_settings)

    assert not marker.exists()
