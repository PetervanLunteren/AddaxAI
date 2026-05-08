"""
Database backups: rolling daily snapshots, pre-upgrade snapshots, and
manual snapshots driven by the user.

Snapshots use SQLite's online backup API (`sqlite3.Connection.backup`),
which is WAL-safe and produces a single consolidated `.db` file with
no `-wal` / `-shm` siblings.

Storage layout under `~/AddaxAI/backups/`:
- `addaxai-<utc-iso>.db`                        — daily rolling snapshot
- `addaxai-pre-upgrade-<rev>-<utc-iso>.db`      — pre-upgrade snapshot
- `.last-rolling-utc-date`                      — daily-throttle marker

The ring buffer keeps the 5 newest daily files. Pre-upgrade files are
never auto-pruned: they are the safety net for the day a migration
eats data, and we want them to survive a string of routine restarts.
"""

import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from app.core.config import Settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

DAILY_BACKUP_KEEP = 5
ROLLING_MARKER_FILENAME = ".last-rolling-utc-date"
RESTORE_MARKER_FILENAME = ".restore-on-next-launch"

_DAILY_RE = re.compile(r"^addaxai-(\d{4}-\d{2}-\d{2}T\d{6}Z)\.db$")
_PRE_UPGRADE_RE = re.compile(
    r"^addaxai-pre-upgrade-([^-\s]+)-(\d{4}-\d{2}-\d{2}T\d{6}Z)\.db$"
)


class BackupInvalidError(RuntimeError):
    """Raised when a candidate backup fails validation."""


@dataclass(frozen=True)
class BackupEntry:
    """One snapshot file on disk."""

    path: Path
    size_bytes: int
    created_utc: datetime
    kind: Literal["daily", "pre-upgrade"]


def snapshot_db(src: Path, dst: Path) -> None:
    """Take an online, WAL-safe snapshot of `src` into `dst`.

    `src` must be an existing SQLite database file. `dst.parent` is
    created if needed. If `dst` already exists it is overwritten.
    """
    if not src.is_file():
        raise FileNotFoundError(f"Source DB not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(str(src))
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def validate_backup(path: Path) -> None:
    """Validate a candidate backup file. Raises `BackupInvalidError` if invalid.

    Checks: file exists, has at least the SQLite header (100 bytes), and
    `PRAGMA integrity_check` returns `ok`.
    """
    if not path.is_file():
        raise BackupInvalidError(f"Not a file: {path}")
    if path.stat().st_size < 100:
        raise BackupInvalidError("File too small to be a SQLite database")

    conn = sqlite3.connect(str(path))
    try:
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as e:
            raise BackupInvalidError(f"Not a valid SQLite database: {e}") from e
    finally:
        conn.close()

    if row is None or row[0] != "ok":
        raise BackupInvalidError(f"Integrity check failed: {row}")


def ring_buffer_backup(settings: Settings) -> Path | None:
    """Take a daily rolling snapshot if one hasn't run today (UTC).

    Returns the new snapshot path, or `None` if today's snapshot already
    exists. Prunes the daily ring buffer down to `DAILY_BACKUP_KEEP`.
    """
    today = _today_str_utc()
    backups_dir = _backups_dir(settings)
    marker = backups_dir / ROLLING_MARKER_FILENAME

    if marker.is_file() and marker.read_text().strip() == today:
        return None

    path = force_ring_buffer_backup(settings)
    marker.write_text(today)
    return path


def force_ring_buffer_backup(settings: Settings) -> Path:
    """Take a daily-format snapshot ignoring the daily throttle.

    Used for: manual "back up now" actions, and the safety snapshot we
    take immediately before swapping in a restored DB.
    """
    src = _live_db_path(settings)
    backups_dir = _backups_dir(settings)
    dst = backups_dir / _daily_filename(_backup_timestamp())
    snapshot_db(src, dst)
    _prune_ring_buffer(backups_dir, keep=DAILY_BACKUP_KEEP)
    logger.info(f"Wrote backup: {dst.name}")
    return dst


def pre_upgrade_backup(settings: Settings, rev: str | None) -> Path:
    """Snapshot the live DB to a pre-upgrade-tagged file."""
    src = _live_db_path(settings)
    dst = _backups_dir(settings) / _pre_upgrade_filename(rev, _backup_timestamp())
    snapshot_db(src, dst)
    logger.info(f"Wrote pre-upgrade backup: {dst.name}")
    return dst


def list_ring_buffer(settings: Settings) -> list[BackupEntry]:
    """Return all backup files (daily + pre-upgrade), newest first."""
    backups_dir = _backups_dir(settings)
    entries: list[BackupEntry] = []
    for child in backups_dir.iterdir():
        if not child.is_file():
            continue
        kind = _classify(child.name)
        if kind is None:
            continue
        stat = child.stat()
        entries.append(
            BackupEntry(
                path=child,
                size_bytes=stat.st_size,
                created_utc=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                kind=kind,
            )
        )
    entries.sort(key=lambda e: e.created_utc, reverse=True)
    return entries


def schedule_restore(settings: Settings, source_path: Path) -> Path:
    """Validate `source_path` and schedule it as the next-launch restore.

    Writes `~/AddaxAI/.restore-on-next-launch` containing the absolute
    source path. The frontend should ask Electron to quit immediately
    after; the next launch consumes the marker via
    `consume_restore_marker()` before `init_db()` runs.

    Returns the marker path (mostly for tests).
    """
    validate_backup(source_path)
    marker = settings.user_data_dir / RESTORE_MARKER_FILENAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(source_path.resolve()))
    return marker


def consume_restore_marker(settings: Settings) -> None:
    """Process and remove a pending restore-on-next-launch marker.

    No-op if the marker is absent. On failure (invalid source, missing
    file, IO error) logs at error level and consumes the marker anyway,
    so a corrupt request can't loop the user through restore-fail-
    restore-fail forever. The live DB is left untouched on failure.
    """
    marker = settings.user_data_dir / RESTORE_MARKER_FILENAME
    if not marker.is_file():
        return
    try:
        raw = marker.read_text().strip()
        if not raw:
            raise BackupInvalidError("Restore marker is empty")
        restore_db(settings, Path(raw))
    except Exception as e:
        logger.error(f"Restore from marker failed: {e}", exc_info=True)
    finally:
        marker.unlink(missing_ok=True)


def restore_db(settings: Settings, source_path: Path) -> None:
    """Replace the live DB with a validated backup file.

    Caller must have already validated the source. We re-validate as a
    defence against time-of-check / time-of-use races. The current DB
    is force-snapshotted to the ring buffer first as a safety net so a
    wrong-file mistake stays recoverable.
    """
    validate_backup(source_path)

    live = _live_db_path(settings)
    if live.is_file():
        force_ring_buffer_backup(settings)

    for sibling in (live, live.with_name(live.name + "-wal"), live.with_name(live.name + "-shm")):
        sibling.unlink(missing_ok=True)

    shutil.copyfile(source_path, live)
    logger.warning(f"Restored DB from {source_path}")


# ── internals ────────────────────────────────────────────────────────


def _backups_dir(settings: Settings) -> Path:
    """`~/AddaxAI/backups/`, created on first use."""
    path = settings.user_data_dir / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _live_db_path(settings: Settings) -> Path:
    """Resolve the configured `database_url` to a filesystem path.

    Crashes for non-`sqlite:///` URLs because nothing else in the app
    supports a different backend, and silently no-op'ing would mask a
    config bug.
    """
    url = settings.database_url
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise RuntimeError(f"Unsupported database_url for backup: {url}")
    return Path(url[len(prefix):])


def _today_str_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _backup_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def _daily_filename(ts: str) -> str:
    return f"addaxai-{ts}.db"


def _pre_upgrade_filename(rev: str | None, ts: str) -> str:
    rev_short = (rev or "unknown")[:8]
    return f"addaxai-pre-upgrade-{rev_short}-{ts}.db"


def _classify(name: str) -> Literal["daily", "pre-upgrade"] | None:
    """Map a filename to its backup kind, or None if it's not a backup."""
    if _PRE_UPGRADE_RE.match(name):
        return "pre-upgrade"
    if _DAILY_RE.match(name):
        return "daily"
    return None


def _prune_ring_buffer(backups_dir: Path, keep: int) -> list[Path]:
    """Delete oldest daily backups beyond `keep`. Returns the deleted paths.

    Pre-upgrade backups are never touched.
    """
    daily = sorted(
        (p for p in backups_dir.iterdir() if p.is_file() and _DAILY_RE.match(p.name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    deleted: list[Path] = []
    for old in daily[keep:]:
        try:
            old.unlink()
            deleted.append(old)
        except OSError as e:
            logger.warning(f"Could not prune {old.name}: {e}")
    return deleted
