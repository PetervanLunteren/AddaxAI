/**
 * Timelapse Analyser integration API client.
 *
 * Single endpoint that kicks off a DB-less analysis run on a folder.
 * The response carries a job_id the caller subscribes to via the
 * existing /ws/jobs/{job_id} websocket (see useTaskProgress).
 *
 * Field shape mirrors `Project` settings (seconds for
 * independence_interval, `detection_threshold`, etc.) so defaults and
 * UI logic stay in lockstep with the main app.
 */

import { api } from "../lib/api-client";

export type SmoothingStrength = "off" | "mild" | "normal" | "aggressive";

export interface TimelapseRunRequest {
  folder_path: string;
  classification_model_id: string | null;
  detection_model_id: string;
  excluded_classes: string[];
  /** null = let the ML subprocess pick its built-in default. */
  detection_batch_size: number | null;
  classification_batch_size: number | null;
  video_fps: number;
  smoothing_strength: SmoothingStrength;
  taxonomic_rollup: boolean;
}

export interface TimelapseRunResponse {
  job_id: string;
}

export const timelapseApi = {
  run: (req: TimelapseRunRequest) =>
    api.post<TimelapseRunResponse>("/api/timelapse/run", req),
};
