"""
Database backups: rolling daily snapshots, pre-upgrade snapshots, and
manual snapshots driven by the user.

Snapshots use SQLite's online backup API (`sqlite3.Connection.backup`),
which is WAL-safe and produces a single consolidated `.db` file with
no `-wal` / `-shm` siblings.

Storage layout under `~/AddaxAI/backups/`:
- `addaxai-<utc-iso>.db`                        — daily rolling snapshot
- `addaxai-pre-upgrade-<rev>-<utc-iso>.db`      — pre-upgrade snapshot
- `addaxai-pre-restore-<utc-iso>.db`            — safety snapshot before a restore
- `addaxai-manual-<utc-iso>[-<note>].db`        — user-initiated "back up now"
- `.last-rolling-utc-date`                      — daily-throttle marker

Manual backups may carry an optional user note, slugged into the
filename (lowercase a-z, 0-9, hyphens). The filename is the only place
the note lives, so it travels with the file to external drives and
needs no sidecar state. One consequence: app versions older than the
note feature do not match the noted filename and will not list such a
backup in their restore picker; it stays restorable there via the
"Restore from a file" escape hatch.

Each automatic kind keeps its own rolling ring of the 5 newest (daily,
pre-upgrade, pre-restore): they are full DB copies, so unbounded
accumulation is wasted disk, and older ones span schema versions you
can't roll back across anyway. Manual "back up now" files are the
user's deliberate action and are never auto-pruned. The filename prefix
is what lets the restore UI label each snapshot by kind. Backups the
user saves to their own folder are not listed here and are theirs to
manage.
"""

import os
import re
import shutil
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.db.migrations import revision_is_known

logger = get_logger(__name__)

# Ring-buffer depth applied per automatic kind (daily / pre-upgrade /
# pre-restore). Manual backups are the user's deliberate action and are
# never auto-pruned. DAILY_BACKUP_KEEP is kept as an alias for callers/tests.
BACKUP_KEEP = 5
DAILY_BACKUP_KEEP = BACKUP_KEEP
ROLLING_MARKER_FILENAME = ".last-rolling-utc-date"
RESTORE_MARKER_FILENAME = ".restore-on-next-launch"

_TS = r"\d{4}-\d{2}-\d{2}T\d{6}Z"
_DAILY_RE = re.compile(rf"^addaxai-({_TS})\.db$")
_PRE_UPGRADE_RE = re.compile(rf"^addaxai-pre-upgrade-([^-\s]+)-({_TS})\.db$")
_PRE_RESTORE_RE = re.compile(rf"^addaxai-pre-restore-({_TS})\.db$")
# The optional note group accepts more than `_slugify_note` generates
# (trailing hyphens, `--` runs), on purpose: a file the user renamed by
# hand should stay listable as long as it stays lowercase.
_MANUAL_RE = re.compile(rf"^addaxai-manual-({_TS})(?:-([a-z0-9][a-z0-9-]*))?\.db$")

# Longest note slug we ever write into a filename.
NOTE_SLUG_MAX_LEN = 40

BackupKind = Literal["daily", "pre-upgrade", "pre-restore", "manual"]


class BackupInvalidError(RuntimeError):
    """Raised when a candidate backup fails validation."""


@dataclass(frozen=True)
class BackupEntry:
    """One snapshot file on disk."""

    path: Path
    size_bytes: int
    created_utc: datetime
    kind: BackupKind
    # User note parsed from a manual backup's filename; None for the
    # automatic kinds and for manual backups saved without one.
    note: str | None = None


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

    Checks: file exists, has at least the SQLite header (100 bytes),
    `PRAGMA integrity_check` returns `ok`, and it carries one
    `alembic_version` row holding a revision this build knows.

    The version checks are what stop a restore turning into a loop: they
    refuse every file `init_db` would refuse at the next launch. A
    database with no `alembic_version` predates 2026-05-08. One with a
    revision this build does not have was written by a newer AddaxAI,
    which is what every backup taken after an update is once the user
    goes back to the older version. Without these checks the user
    restores it happily and meets the same startup error, with no clue
    that the file they picked was the problem (a beta tester did that
    five times, 2026-11-12).
    """
    if not path.is_file():
        raise BackupInvalidError(f"Not a file: {path}")
    if path.stat().st_size < 100:
        raise BackupInvalidError("File too small to be a SQLite database")

    conn = sqlite3.connect(str(path))
    # Both bound before their try blocks. Neither can actually be read
    # unassigned, because every except below raises, but mypy's
    # possibly-undefined check widens its flow analysis inside a
    # try/finally and cannot see that. Binding is cheaper than an ignore
    # comment and survives someone later adding an except that returns
    # instead of raising.
    row: tuple | None = None
    versions: list = []
    try:
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as e:
            raise BackupInvalidError(f"Not a valid SQLite database: {e}") from e

        if row is not None and row[0] == "ok":
            try:
                versions = conn.execute(
                    "SELECT version_num FROM alembic_version"
                ).fetchall()
            except sqlite3.DatabaseError as e:
                raise BackupInvalidError(
                    "This file is not an AddaxAI database, or it is from an "
                    "early beta that this version cannot open."
                ) from e
            if len(versions) != 1:
                raise BackupInvalidError(
                    f"Expected one schema version in the backup, found "
                    f"{len(versions)}."
                )
            if not revision_is_known(versions[0][0]):
                raise BackupInvalidError(
                    "This backup was made by a newer version of AddaxAI, "
                    "so this version cannot open it. Pick an older backup."
                )
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

    The daily ring buffer's forced writer. Prunes to `DAILY_BACKUP_KEEP`.
    Manual and pre-restore snapshots have their own tagged writers below
    so the restore UI can tell them apart from a daily rollover.
    """
    src = live_db_path(settings)
    backups_dir = _backups_dir(settings)
    dst = backups_dir / _daily_filename(_backup_timestamp())
    snapshot_db(src, dst)
    _prune_ring_buffer(backups_dir, keep=DAILY_BACKUP_KEEP)
    logger.info(f"Wrote backup: {dst.name}")
    return dst


def manual_snapshot(settings: Settings, note: str | None = None) -> Path:
    """Snapshot the live DB to a manual-tagged file (user "back up now").

    Kept indefinitely (not part of the daily cap): the user asked for it.
    An optional note is slugged into the filename so the restore picker
    can show why the snapshot was taken.
    """
    src = live_db_path(settings)
    dst = _backups_dir(settings) / manual_backup_filename(note)
    snapshot_db(src, dst)
    logger.info(f"Wrote manual backup: {dst.name}")
    return dst


def manual_backup_filename(note: str | None = None) -> str:
    """Filename for a manual snapshot taken now, with the note slugged in.

    The one public builder for the manual pattern; also used by the
    snapshot endpoint's save-to-chosen-folder branch.
    """
    ts = _backup_timestamp()
    slug = _slugify_note(note)
    return f"addaxai-manual-{ts}-{slug}.db" if slug else f"addaxai-manual-{ts}.db"


def pre_restore_snapshot(settings: Settings) -> Path:
    """Snapshot the live DB to a pre-restore-tagged file.

    Taken right before a restore swaps a backup in, so the pre-restore
    state stays recoverable (the "undo the restore" safety net). Rolls:
    keeps the newest `BACKUP_KEEP`.
    """
    src = live_db_path(settings)
    backups_dir = _backups_dir(settings)
    dst = backups_dir / _pre_restore_filename(_backup_timestamp())
    snapshot_db(src, dst)
    _prune_kind(backups_dir, _PRE_RESTORE_RE, keep=BACKUP_KEEP)
    logger.info(f"Wrote pre-restore backup: {dst.name}")
    return dst


def pre_upgrade_backup(settings: Settings, rev: str | None) -> Path | None:
    """Snapshot the live DB to a pre-upgrade-tagged file.

    Returns `None` when a snapshot for `rev` was already taken today
    (UTC), the same once-a-day rule as the rolling backup. Repeated
    launches at the same revision (a `uvicorn --reload` storm in dev, or
    a user clicking Retry on the startup error page) would otherwise
    push the real pre-upgrade snapshot out of the ring buffer with
    identical duplicates. That is the one snapshot that matters on the
    day a migration eats data.

    Only today's, not any earlier one. The same revision on a later day
    is a different database: a user who updated, went back to the older
    version, restored, and worked on for weeks is at this revision again
    when they update a second time. Skipping then left them with a
    pre-upgrade backup from before those weeks, the file the startup
    error tells them to restore. The price is that an upgrade which
    keeps failing gets one more copy per day it is retried on.

    Also returns `None` for a revision this build does not know. That
    database was written by a newer AddaxAI, so nothing is about to be
    upgraded, and the copy would be the one file in the backups folder
    named "pre-upgrade" that this version cannot open. It is also the
    newest, so it is the one a user picks (beta report 2026-11-12).

    Rolls: keeps the newest `BACKUP_KEEP`. Older pre-upgrade snapshots
    span schema versions you can't roll back across anyway, and each is
    a full DB copy, so unbounded accumulation is just wasted disk.
    """
    if rev is not None and not revision_is_known(rev):
        logger.info(
            f"Database is at revision {rev}, which this version does not "
            "know: no upgrade ahead, so no pre-upgrade backup"
        )
        return None

    src = live_db_path(settings)
    backups_dir = _backups_dir(settings)
    dst = backups_dir / _pre_upgrade_filename(rev, _backup_timestamp())

    # Group 1 is the revision tag, group 2 the timestamp, which starts
    # with the UTC date.
    tag = _PRE_UPGRADE_RE.match(dst.name).group(1)  # type: ignore[union-attr]
    today = _today_str_utc()
    if any(
        (m := _PRE_UPGRADE_RE.match(p.name))
        and m.group(1) == tag
        and m.group(2).startswith(today)
        for p in backups_dir.glob("addaxai-pre-upgrade-*.db")
    ):
        logger.info(f"Pre-upgrade backup for {tag} already taken today, skipping")
        return None

    snapshot_db(src, dst)
    _prune_kind(backups_dir, _PRE_UPGRADE_RE, keep=BACKUP_KEEP)
    logger.info(f"Wrote pre-upgrade backup: {dst.name}")
    return dst


def list_ring_buffer(settings: Settings) -> list[BackupEntry]:
    """Return all tagged backup files in the app dir, newest first.

    Covers daily / pre-upgrade / pre-restore / manual. Backups the user
    saved to their own folder live outside this dir and aren't listed.
    """
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
                note=_manual_note(child.name) if kind == "manual" else None,
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
    # UTF-8 on both sides of the marker, here and in
    # `consume_restore_marker`: the startup error page (Electron) writes
    # it as UTF-8 too, and Python's default on Windows is the ANSI code
    # page, which turns a "ñ" in the path into a file that is not there.
    marker.write_text(str(source_path.resolve()), encoding="utf-8")
    return marker


def consume_restore_marker(settings: Settings) -> str | None:
    """Process and remove a pending restore-on-next-launch marker.

    No-op if the marker is absent. On failure (invalid source, missing
    file, IO error) logs at error level and consumes the marker anyway,
    so a corrupt request can't loop the user through restore-fail-
    restore-fail forever. The live DB is left untouched on failure.

    Returns why the restore failed, or `None` when it worked or there
    was nothing to do. The startup error page writes the marker without
    validating the file, so this is the only place that learns the pick
    was bad, and the caller shows it: otherwise the user lands on the
    same error screen with nothing saying their restore did not happen.
    """
    marker = settings.user_data_dir / RESTORE_MARKER_FILENAME
    if not marker.is_file():
        return None
    try:
        raw = marker.read_text(encoding="utf-8").strip()
        if not raw:
            raise BackupInvalidError("Restore marker is empty")
        restore_db(settings, Path(raw))
    except Exception as e:
        logger.error(f"Restore from marker failed: {e}", exc_info=True)
        return str(e)
    finally:
        marker.unlink(missing_ok=True)
    return None


def restore_db(settings: Settings, source_path: Path) -> None:
    """Replace the live DB with a validated backup file.

    Caller must have already validated the source. We re-validate as a
    defence against time-of-check / time-of-use races. The current DB
    is force-snapshotted to the ring buffer first as a safety net so a
    wrong-file mistake stays recoverable.

    The source is copied to a staging file before anything destructive
    runs, and the live DB is replaced by one atomic rename at the end.
    Both halves are load-bearing: the pre-restore snapshot prunes its
    ring, which deletes the selected source when it is the oldest of
    `BACKUP_KEEP` pre-restore backups, and a copy that fails partway
    (full disk, unplugged drive) must not leave the live DB missing.
    The live DB therefore survives until its replacement is fully on
    disk, whatever fails.
    """
    validate_backup(source_path)

    live = live_db_path(settings)
    # Same directory as the live DB, so os.replace stays on one
    # filesystem. Matches no backup filename pattern, so the restore
    # picker never lists a leftover from a crashed restore.
    staged = live.with_name(live.name + ".restore-staging")
    staged.unlink(missing_ok=True)
    try:
        shutil.copyfile(source_path, staged)

        if live.is_file():
            pre_restore_snapshot(settings)

        # The sidecars belong to the outgoing DB; the snapshot above
        # read through the WAL, so nothing in them is lost.
        for sibling in (live.with_name(live.name + "-wal"), live.with_name(live.name + "-shm")):
            sibling.unlink(missing_ok=True)

        os.replace(staged, live)
    finally:
        staged.unlink(missing_ok=True)

    logger.warning(f"Restored DB from {source_path}")


# ── internals ────────────────────────────────────────────────────────


def _backups_dir(settings: Settings) -> Path:
    """`~/AddaxAI/backups/`, created on first use."""
    path = settings.user_data_dir / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_db_path(settings: Settings) -> Path:
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


def _pre_restore_filename(ts: str) -> str:
    return f"addaxai-pre-restore-{ts}.db"


def _slugify_note(raw: str | None) -> str | None:
    """Reduce a free-text note to a filename slug, or None if nothing survives.

    ASCII-fold, lowercase, collapse every run of anything else into one
    hyphen, strip the ends, cap at `NOTE_SLUG_MAX_LEN`. Deliberately not
    the app's other slugifiers: `export_formats.slugify` keeps non-ASCII
    letters (which `_MANUAL_RE` would refuse, silently unlisting the
    file) and `separate_folders._slug` separates with underscores.
    """
    if not raw:
        return None
    folded = (
        unicodedata.normalize("NFKD", raw)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return slug[:NOTE_SLUG_MAX_LEN].rstrip("-") or None


def _manual_note(name: str) -> str | None:
    """The note slug in a manual backup's filename, or None."""
    m = _MANUAL_RE.match(name)
    return m.group(2) if m else None


def _classify(name: str) -> BackupKind | None:
    """Map a filename to its backup kind, or None if it's not a backup.

    The tagged prefixes are checked before the bare daily pattern (which
    would otherwise only match `addaxai-<ts>.db` anyway).
    """
    if _PRE_UPGRADE_RE.match(name):
        return "pre-upgrade"
    if _PRE_RESTORE_RE.match(name):
        return "pre-restore"
    if _MANUAL_RE.match(name):
        return "manual"
    if _DAILY_RE.match(name):
        return "daily"
    return None


def _prune_kind(
    backups_dir: Path, pattern: re.Pattern[str], keep: int
) -> list[Path]:
    """Delete the oldest files matching `pattern` beyond `keep` newest.

    Only one kind is touched per call, so each automatic kind keeps its own
    independent ring. Manual backups have no pattern passed here, so they
    are never pruned. Returns the deleted paths.
    """
    files = sorted(
        (p for p in backups_dir.iterdir() if p.is_file() and pattern.match(p.name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    deleted: list[Path] = []
    for old in files[keep:]:
        try:
            old.unlink()
            deleted.append(old)
        except OSError as e:
            logger.warning(f"Could not prune {old.name}: {e}")
    return deleted


def _prune_ring_buffer(backups_dir: Path, keep: int) -> list[Path]:
    """Prune the daily ring buffer to `keep` newest. Thin wrapper over
    `_prune_kind` for the daily pattern; other kinds prune themselves."""
    return _prune_kind(backups_dir, _DAILY_RE, keep)
