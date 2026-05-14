/**
 * Folder runs API client.
 *
 * Wraps the backend folder-run orchestration endpoints. Used by the
 * folder-run stepper (FolderRunLayout + step components) to create a
 * run, resume one by id, and persist the current step so a reopened
 * run lands the user back where they were.
 */

import { api } from "../lib/api-client";
import type { ProjectResponse } from "./types";

export type FolderRunStep = "folder" | "model" | "run" | "review" | "save";

/** Queue entry shape carried on a folder-run response. Matches the
 * DeploymentQueueResponse Pydantic schema but only the fields the
 * stepper consumes are listed. */
export interface FolderRunQueueEntry {
  id: string;
  project_id: string;
  folder_path: string;
  site_id: string | null;
  video_count: number;
  image_count: number;
  status: "pending" | "processing" | "completed" | "failed";
  created_at_utc: string;
  processed_at_utc: string | null;
  error: string | null;
  warnings: string | null;
  deployment_id: string | null;
}

export interface FolderRunResponse {
  project: ProjectResponse;
  queue_entry: FolderRunQueueEntry | null;
  step: FolderRunStep;
}

export interface FolderRunCreate {
  source_folder: string;
  /** Falls back to the folder basename when omitted. */
  name?: string;
  video_count?: number;
  image_count?: number;
}

export const folderRunsApi = {
  /** Create a folder run. Returns the new project + queue entry. */
  create: (payload: FolderRunCreate) =>
    api.post<FolderRunResponse>("/api/folder-runs", payload),

  /** Resume an existing folder run by id (the project id). */
  get: (runId: string) =>
    api.get<FolderRunResponse>(`/api/folder-runs/${runId}`),

  /** Persist the current step so a reopened run lands here. */
  updateStep: (runId: string, step: FolderRunStep) =>
    api.patch<FolderRunResponse>(`/api/folder-runs/${runId}/step`, { step }),
};
