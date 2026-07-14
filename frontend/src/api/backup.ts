/**
 * Database-backup API.
 *
 * The four endpoints under /api/backup, plus their TypeScript types.
 * Restore is two-phase: POST /restore validates and writes the
 * .restore-on-next-launch marker; the caller then asks Electron to
 * quit, and the next launch swaps the DB in before init_db runs.
 */

import { api } from "../lib/api-client";

export type BackupKind = "daily" | "pre-upgrade" | "pre-restore" | "manual";

export interface BackupEntry {
  path: string;
  size_bytes: number;
  created_utc: string;
  kind: BackupKind;
}

export interface BackupListResponse {
  entries: BackupEntry[];
}

export interface BackupDirResponse {
  path: string;
}

export interface SnapshotResponse {
  path: string;
  size_bytes: number;
}

export const backupApi = {
  getDir: () => api.get<BackupDirResponse>("/api/backup/dir"),
  list: () => api.get<BackupListResponse>("/api/backup/list"),
  snapshotToRingBuffer: () =>
    api.post<SnapshotResponse>("/api/backup/snapshot", {}),
  snapshotToFolder: (target_dir: string) =>
    api.post<SnapshotResponse>("/api/backup/snapshot", { target_dir }),
  restore: (source_path: string) =>
    api.post<{ scheduled: true }>("/api/backup/restore", { source_path }),
};
