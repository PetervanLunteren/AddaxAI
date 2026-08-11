"""
Database-backup endpoints.

Four operations exposed under `/api/backup`:

- `GET  /dir`       — return the absolute path of `~/AddaxAI/backups/`.
- `GET  /list`      — list daily + pre-upgrade snapshots, newest first.
- `POST /snapshot`  — write a snapshot. With `target_dir` the file lands
                      in that folder; without, it lands in the ring
                      buffer (force, ignoring the daily throttle).
- `POST /restore`   — validate a backup file and write the
                      `.restore-on-next-launch` marker. The frontend
                      asks Electron to quit; the next launch swaps in
                      the file before init_db.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.backup import (
    BackupDirResponse,
    BackupEntryResponse,
    BackupListResponse,
    RestoreRequest,
    RestoreResponse,
    SnapshotRequest,
    SnapshotResponse,
)
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.db.backup import (
    BackupInvalidError,
    list_ring_buffer,
    manual_backup_filename,
    manual_snapshot,
    schedule_restore,
    snapshot_db,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/backup", tags=["Backup"])


@router.get("/dir", response_model=BackupDirResponse)
def get_backup_dir() -> BackupDirResponse:
    """Return the absolute path of the backups folder, creating it if needed."""
    settings = get_settings()
    backups_dir = settings.user_data_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    return BackupDirResponse(path=str(backups_dir))


@router.get("/list", response_model=BackupListResponse)
def list_backups() -> BackupListResponse:
    """List app-dir snapshots (daily, pre-upgrade, pre-restore, manual), newest first."""
    settings = get_settings()
    entries = [
        BackupEntryResponse(
            path=str(e.path),
            size_bytes=e.size_bytes,
            created_utc=e.created_utc,
            kind=e.kind,
            note=e.note,
        )
        for e in list_ring_buffer(settings)
    ]
    return BackupListResponse(entries=entries)


@router.post("/snapshot", response_model=SnapshotResponse)
def take_snapshot(req: SnapshotRequest) -> SnapshotResponse:
    """Write a manual snapshot of the live DB.

    Without `target_dir`: writes a manual-tagged file to the backups
    folder (always produces one; ignores the daily throttle).

    With `target_dir`: writes to the chosen folder using the same
    `addaxai-manual-<utc-iso>[-<note>].db` filename. The folder must
    exist. These are the user's to manage and are not listed by
    GET /list.

    `note` is optional free text; it is slugged into the filename on
    both paths so the restore picker can show why the snapshot exists.
    """
    settings = get_settings()

    if req.target_dir is None:
        path = manual_snapshot(settings, note=req.note)
        return SnapshotResponse(path=str(path), size_bytes=path.stat().st_size)

    target = Path(req.target_dir)
    if not target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target folder does not exist: {target}",
        )

    src = settings.user_data_dir / "addaxai.db"
    dst = target / manual_backup_filename(req.note)
    snapshot_db(src, dst)
    logger.info(f"Wrote backup to user-chosen folder: {dst}")
    return SnapshotResponse(path=str(dst), size_bytes=dst.stat().st_size)


@router.post("/restore", response_model=RestoreResponse)
def restore_from_backup(req: RestoreRequest) -> RestoreResponse:
    """Validate the source file and schedule restore on next launch.

    Caller is expected to ask Electron to quit immediately afterwards
    so the next launch swaps the DB in before init_db runs.
    """
    settings = get_settings()
    source = Path(req.source_path)
    try:
        schedule_restore(settings, source)
    except BackupInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    logger.warning(f"Restore scheduled from {source}")
    return RestoreResponse()
