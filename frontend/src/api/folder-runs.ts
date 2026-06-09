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
  | "model"
  | "labels"
  | "counts"
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
  /** Id of an analysis job currently running for this run, or null.
   * Used by the Setup step to re-attach the progress modal after a
   * refresh (modal == running run). */
  active_job_id: string | null;
}

export interface FolderRunCreate {
  source_folder: string;
  /** Falls back to the folder basename when omitted. */
  name?: string;
  video_count?: number;
  image_count?: number;
  /** "Discard and start over": when true and a folder-run project
   * already points at this source folder, the existing project is
   * cascade-deleted (DB rows + on-disk .addaxai cache) before the
   * fresh one is created. Default false keeps the create-or-resume
   * behaviour for callers that don't care. */
  force_new?: boolean;
}

/** Summary returned by GET /api/folder-runs/lookup. The Step 1 notice
 * card renders this when the user picks a folder that already has a
 * folder-run project. ``null`` from the API means "no existing run".
 *
 * ``verified_detection_count`` is the canonical "how much has the
 * user reviewed" signal: marking a file as verified cascades
 * Detection.verified=True onto every visible detection in the file,
 * so this count grows whether the user works file-by-file or
 * observation-by-observation in the verify grid. */
export interface FolderRunLookup {
  id: string;
  name: string;
  created_at_utc: string;
  updated_at_utc: string;
  detection_model_id: string | null;
  classification_model_id: string | null;
  /** Friendly name from the local manifest; falls back to the id
   * when the model isn't installed (catalog drift, fresh install). */
  detection_model_name: string | null;
  classification_model_name: string | null;
  step: FolderRunStep;
  file_count: number;
  detection_count: number;
  species_count: number;
  verified_file_count: number;
  verified_detection_count: number;
}

export type SeparateGroupBy = "taxonomic" | "flat" | "none";

/** Per-module summary from the save-outputs endpoint. */
export interface SeparateFoldersResult {
  copied_count: number;
  moved_count: number;
  written_count: number;
  skipped_missing_source: number;
  skipped_excluded: number;
  renamed_count: number;
  by_label: Record<string, number>;
  errors: string[];
}

/** Combined per-file annotation pass — blur people / vehicles and / or
 * draw detection boxes, single image per source written into each
 * separated destination (or directly under the output root when
 * separation is off). */
export interface AnnotatedCopiesResult {
  /** Saved destinations (one per source file × number of separated
   * placements). */
  written_count: number;
  /** Total bbox + pill placements drawn across all sources. */
  bbox_count: number;
  /** Total person / vehicle bboxes blurred across all sources. */
  blurred_box_count: number;
  /** Files where neither effect would have produced a visible change
   * (e.g. only anonymise on, no person / vehicle detections). */
  skipped_no_change: number;
  skipped_missing_source: number;
  skipped_excluded: number;
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
  /** Files dropped because every passing identified detection (species,
   * or builtin person / vehicle) was in the user's exclusion set. */
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
  /** "No subfolders" mode: source subfolder path → file count, rendered
   * with the same builder as the species tree. */
  by_source_tree: Record<string, number>;
  /** "No subfolders" mode: a capped sample of loose root-file names.
   * Root total = in_scope_files − sum(by_source_tree). */
  root_files: string[];
}

export interface OutputPreviewRequest {
  /** Copy empty captures (no animal / person / vehicle) too. Off /
   * omitted = empties are skipped, matching the save request. */
  include_empty?: boolean;
  /** Common vs scientific species-name leaf, mirroring the save request
   * so the previewed tree matches what will be written. */
  name_mode?: "common" | "scientific";
  /** Label identifiers to exclude (LabelTaxonomy.id UUIDs or raw label
   * strings). Mirrors the save request so the preview reflects the
   * species filter. */
  excluded_label_ids?: string[];
  /** Event grouping, mirroring the save request so the previewed counts
   * match what the save will write. */
  group_events?: boolean;
}

export interface SaveOutputsRequest {
  output_dir: string;
  separate_folders?: boolean;
  separate_group_by?: SeparateGroupBy;
  /** Keep a burst together: every file in an event goes to one folder
   * (the event's main species). */
  group_events?: boolean;
  /** Label identifiers to exclude from the media copies / visualisations
   * (LabelTaxonomy.id UUIDs or raw label strings). The data exports are
   * never filtered. */
  excluded_label_ids?: string[];
  /** Copy empty captures (no animal / person / vehicle) too. Off /
   * omitted = empties are skipped. */
  include_empty?: boolean;
  /** Draw detection bounding boxes + pill labels on annotated copies. */
  draw_bboxes?: boolean;
  /** Blur person / vehicle detections on annotated copies (privacy
   * mode for shareable datasets). */
  anonymise?: boolean;
  recognition_json?: boolean;
  csv?: boolean;
  xlsx?: boolean;
  /** Which species name to burn into the visualised images. Mirrors the
   * UI display preference. EXIF metadata always carries both names. */
  name_mode?: "common" | "scientific";
}

/** Response from POST /save-outputs — just the spawned job's id.
 * Per-module results land on the job's WebSocket completion event
 * (data payload), shaped as ``SaveOutputsResult``. */
export interface SaveOutputsResponse {
  job_id: string;
}

/** Shape of the data payload emitted on the job's complete event.
 * Per-module summaries that the completion screen aggregates into
 * a single confirmation + an error banner when applicable. */
export interface SaveOutputsResult {
  output_dir: string;
  /** Count of source files in the run, computed once on the backend
   * so the completion tally doesn't have to reconstruct it from
   * multi-placement-inflated per-module counters. */
  source_file_count: number;
  separate_folders?: SeparateFoldersResult;
  annotated_copies?: AnnotatedCopiesResult;
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

  /** Probe for an existing folder-run project pointing at this source
   * folder. Returns null when there's no match (the common case). */
  lookup: (folder: string) =>
    api.get<FolderRunLookup | null>(
      `/api/folder-runs/lookup?folder=${encodeURIComponent(folder)}`,
    ),

  /** Persist the current step so a reopened run lands here. */
  updateStep: (runId: string, step: FolderRunStep) =>
    api.patch<FolderRunResponse>(`/api/folder-runs/${runId}/step`, { step }),

  /** Wipe a folder run's analysis output so the queue entry can be
   * re-processed. Caller surfaces a destructive confirm dialog. */
  rerun: (runId: string) =>
    api.post<FolderRunResponse>(`/api/folder-runs/${runId}/rerun`, {}),

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
