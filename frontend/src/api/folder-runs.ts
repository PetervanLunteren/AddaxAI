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

export type FolderRunStep =
  | "folder"
  | "model"
  | "run"
  | "review"
  | "overview"
  | "save";

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

export type SeparateMode = "copy" | "move" | "symlink";
export type SeparateGroupBy = "taxonomic" | "flat";
export type ExifMode = "overwrite" | "copy";

/** Per-module summary from the save-outputs endpoint. */
export interface SeparateFoldersResult {
  copied_count: number;
  moved_count: number;
  linked_count: number;
  written_count: number;
  skipped_missing_source: number;
  renamed_count: number;
  /** Distinct source files that ended up in more than one folder
   * because their detections covered multiple species. */
  multi_placement_count: number;
  by_label: Record<string, number>;
  errors: string[];
}

export interface VisualisedImagesResult {
  written_count: number;
  skipped_no_bbox: number;
  skipped_missing_source: number;
  renamed_count: number;
  errors: string[];
}

export interface BlurPeopleResult {
  written_count: number;
  blurred_box_count: number;
  skipped_no_target: number;
  skipped_missing_source: number;
  renamed_count: number;
  errors: string[];
}

export interface RecognitionJsonResult {
  output_path: string;
  image_count: number;
  detection_count: number;
  classification_count: number;
  errors: string[];
}

export interface ObservationsCsvResult {
  output_path: string;
  row_count: number;
  errors: string[];
}

export interface ObservationsXlsxResult {
  output_path: string;
  row_count: number;
  errors: string[];
}

export interface ExifMetadataResult {
  mode: ExifMode;
  written_count: number;
  skipped_no_detections: number;
  skipped_video: number;
  skipped_missing_source: number;
  errors: string[];
}

export interface RunReadmeResult {
  output_path: string;
  bytes_written: number;
  errors: string[];
}

/** Aggregate counts the Save step uses to render its live folder
 * preview. Numbers are exact, not estimates — placement rules are
 * deterministic from the DB state. */
export interface OutputPreview {
  total_files: number;
  image_count: number;
  video_count: number;
  /** Sum of size_bytes across files with the column populated. */
  total_bytes: number;
  files_with_known_size: number;
  /** Animal files dropped because every passing label was in the
   * user's exclusion set. */
  dropped_by_filter: number;
  /** Files surviving the exclusion filter — what every output module
   * will iterate over. */
  in_scope_files: number;
  /** In-scope split by file type. Used for visualised / blurred /
   * exif-tagged counts (which only write per image). */
  in_scope_image_count: number;
  in_scope_video_count: number;
  /** Sum of size_bytes for in-scope files only. */
  in_scope_bytes: number;
  /** Slash-separated taxonomic paths → placement counts. Keys look
   * like "Mammalia/Carnivora/Canidae/Canis/dog". The Save step
   * parses these into a nested tree at render time. Non-animal
   * files contribute single-segment paths ("person", "blank", ...). */
  by_taxonomic_tree: Record<string, number>;
  /** Flat single-segment placements: one folder per species label
   * (or per non-animal observation type, or animal/ fallback). */
  by_flat: Record<string, number>;
  /** Distinct source files appearing in more than one leaf folder. */
  multi_species_files: number;
}

export interface OutputPreviewRequest {
  /** Label identifiers to exclude — LabelTaxonomy UUIDs and / or raw
   * label strings, matching the leaf IDs the label-tree endpoint
   * emits. */
  excluded_label_ids?: string[];
}

export interface SaveOutputsRequest {
  output_dir: string;
  separate_folders?: boolean;
  separate_method?: SeparateMode;
  separate_group_by?: SeparateGroupBy;
  /** Label identifiers to exclude from every output. Each entry is a
   * LabelTaxonomy.id UUID or a raw Detection.label string, matching
   * the heterogeneous output of the label-tree endpoint. Empty /
   * omitted = no filter. */
  excluded_label_ids?: string[];
  visualised_images?: boolean;
  blur_people?: boolean;
  write_exif?: boolean;
  exif_mode?: ExifMode;
  recognition_json?: boolean;
  csv?: boolean;
  xlsx?: boolean;
}

/** Response from POST /save-outputs — just the spawned job's id.
 * Per-module results land on the job's WebSocket completion event
 * (data payload), shaped as ``SaveOutputsResult``. */
export interface SaveOutputsResponse {
  job_id: string;
}

/** Shape of the data payload emitted on the job's complete event.
 * Same per-module summaries the synchronous endpoint used to
 * return inline — the completion screen still consumes this shape. */
export interface SaveOutputsResult {
  output_dir: string;
  separate_folders?: SeparateFoldersResult;
  visualised_images?: VisualisedImagesResult;
  blur_people?: BlurPeopleResult;
  write_exif?: ExifMetadataResult;
  recognition_json?: RecognitionJsonResult;
  csv?: ObservationsCsvResult;
  xlsx?: ObservationsXlsxResult;
  run_readme?: RunReadmeResult;
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

  /** Run the chosen postprocess outputs synchronously. */
  saveOutputs: (runId: string, payload: SaveOutputsRequest) =>
    api.post<SaveOutputsResponse>(
      `/api/folder-runs/${runId}/save-outputs`,
      payload,
    ),

  /** Aggregate file counts for the Save step's live folder preview.
   * POST because the body carries the species exclusion set. */
  getOutputPreview: (runId: string, payload: OutputPreviewRequest = {}) =>
    api.post<OutputPreview>(
      `/api/folder-runs/${runId}/output-preview`,
      payload,
    ),
};
