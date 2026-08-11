"""Pydantic schemas for the database-backup API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class BackupEntryResponse(BaseModel):
    """One snapshot file as returned by GET /api/backup/list."""

    path: str
    size_bytes: int
    created_utc: datetime
    kind: Literal["daily", "pre-upgrade", "pre-restore", "manual"]
    # Note slug from a manual backup's filename; None otherwise.
    note: str | None = None


class BackupListResponse(BaseModel):
    entries: list[BackupEntryResponse]


class BackupDirResponse(BaseModel):
    path: str


class SnapshotRequest(BaseModel):
    """If `target_dir` is set we write there; otherwise we force-write to the ring buffer."""

    target_dir: str | None = None
    # Free text; the backend slugs it into the filename. The cap is a
    # request sanity bound, the real limit is the slug's 40 chars.
    note: str | None = Field(default=None, max_length=120)


class SnapshotResponse(BaseModel):
    path: str
    size_bytes: int


class RestoreRequest(BaseModel):
    source_path: str


class RestoreResponse(BaseModel):
    scheduled: Literal[True] = True
