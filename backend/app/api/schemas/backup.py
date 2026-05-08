"""Pydantic schemas for the database-backup API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class BackupEntryResponse(BaseModel):
    """One snapshot file as returned by GET /api/backup/list."""

    path: str
    size_bytes: int
    created_utc: datetime
    kind: Literal["daily", "pre-upgrade"]


class BackupListResponse(BaseModel):
    entries: list[BackupEntryResponse]


class BackupDirResponse(BaseModel):
    path: str


class SnapshotRequest(BaseModel):
    """If `target_dir` is set we write there; otherwise we force-write to the ring buffer."""

    target_dir: str | None = None


class SnapshotResponse(BaseModel):
    path: str
    size_bytes: int


class RestoreRequest(BaseModel):
    source_path: str


class RestoreResponse(BaseModel):
    scheduled: Literal[True] = True
