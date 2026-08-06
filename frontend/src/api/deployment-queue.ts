/**
 * Deployment Queue API endpoints.
 *
 * Following DEVELOPERS.MD principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";
import type {
  CsvImportPreview,
  CsvImportResult,
  DeploymentImportRow,
} from "./types";

export type { DeploymentImportRow };

export interface DeploymentQueueEntry {
  id: string;
  project_id: string;
  folder_path: string;
  video_count: number;
  image_count: number;
  site_id: string | null;
  datetime_offset_seconds: number | null;
  /** Fill missing capture dates from each file's modification time.
   * Never overrides a real capture date. */
  use_file_mtime_fallback: boolean;
  notes: string | null;
  tags: Record<string, string>;
  status: "pending" | "processing" | "completed" | "failed";
  created_at_utc: string;
  processed_at_utc: string | null;
  error: string | null;
  /** Newline-joined paths of files skipped during ingest for non-fatal
   * reasons (e.g. no extractable capture timestamp). Null when nothing
   * was skipped. */
  warnings: string | null;
  deployment_id: string | null;
}

export interface DeploymentQueueCreate {
  project_id: string;
  folder_path: string;
  site_id?: string | null;
  video_count?: number;
  image_count?: number;
  datetime_offset_seconds?: number | null;
  use_file_mtime_fallback?: boolean;
  notes?: string | null;
  tags?: Record<string, string>;
}

export interface ProcessQueueRequest {
  project_id: string;
}

export interface ProcessQueueResponse {
  message: string;
  jobs_started: number;
  job_ids: string[];
  queue_entry_ids: string[];
}

export const deploymentQueueApi = {
  /**
   * List all queue entries for a project
   */
  list: (projectId: string, status?: string) => {
    const params = new URLSearchParams({ project_id: projectId });
    if (status) params.append("status", status);
    return api.get<DeploymentQueueEntry[]>(
      `/api/deployment-queue?${params.toString()}`
    );
  },

  /**
   * Create a new queue entry
   */
  create: (data: DeploymentQueueCreate) =>
    api.post<DeploymentQueueEntry>("/api/deployment-queue", data),

  /**
   * Get queue entry by ID
   */
  get: (id: string) =>
    api.get<DeploymentQueueEntry>(`/api/deployment-queue/${id}`),

  /**
   * Remove entry from queue
   */
  remove: (id: string) => api.delete(`/api/deployment-queue/${id}`),

  /**
   * Process all pending entries in queue
   */
  process: (data: ProcessQueueRequest) =>
    api.post<ProcessQueueResponse>("/api/deployment-queue/process", data),

  /**
   * Check a deployment CSV without writing anything.
   */
  importPreview: (projectId: string, file: File) =>
    api.upload<CsvImportPreview<DeploymentImportRow>>(
      `/api/deployment-queue/import/preview?project_id=${projectId}`,
      file
    ),

  /**
   * Import a deployment CSV into the queue, all or nothing. The file is sent
   * again rather than the previewed rows, so the backend checks one thing in
   * one way.
   */
  importCsv: (projectId: string, file: File) =>
    api.upload<CsvImportResult>(
      `/api/deployment-queue/import?project_id=${projectId}`,
      file
    ),
};
