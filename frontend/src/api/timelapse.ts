/**
 * Timelapse Analyser integration API client.
 *
 * Single endpoint that kicks off a DB-less analysis run on a folder.
 * The response carries a job_id the caller subscribes to via the
 * existing /ws/jobs/{job_id} websocket (see useTaskProgress).
 */

import { api } from "../lib/api-client";

export type SmoothingStrength = "off" | "mild" | "normal" | "aggressive";

export interface TimelapseRunRequest {
  folder_path: string;
  classification_model_id: string | null;
  detection_model_id: string;
  excluded_classes: string[];
  detection_confidence_threshold: number;
  detection_batch_size: number;
  classification_batch_size: number;
  video_fps: number;
  independence_interval_minutes: number;
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
