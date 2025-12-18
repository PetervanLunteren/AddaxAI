/**
 * TypeScript types for API requests and responses.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Matches backend Pydantic schemas
 */

// Project types
export interface ProjectCreate {
  name: string;
  description?: string | null;
}

export interface ProjectUpdate {
  name?: string | null;
  description?: string | null;
}

export interface ProjectResponse {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectWithStats extends ProjectResponse {
  site_count: number;
  deployment_count: number;
  file_count: number;
  detection_count: number;
}

// Site types
export interface SiteCreate {
  project_id: string;
  name: string;
  latitude?: number | null;
  longitude?: number | null;
  elevation_m?: number | null;
  habitat_type?: string | null;
  notes?: string | null;
}

export interface SiteUpdate {
  name?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  elevation_m?: number | null;
  habitat_type?: string | null;
  notes?: string | null;
}

export interface SiteResponse {
  id: string;
  project_id: string;
  name: string;
  latitude: number | null;
  longitude: number | null;
  elevation_m: number | null;
  habitat_type: string | null;
  notes: string | null;
  created_at: string;
}

// Job types
export type JobType =
  | "deployment_analysis"
  | "import"
  | "ml_inference"
  | "export"
  | "event_computation";

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type DetectionModel = "MD5A-0-0" | "MD5B-0-0";
export type ClassificationModel = "EUR-DF-v1-3" | "NAM-ADS-v1" | "none";

// ML Model Status
export type ModelStatus = "ready" | "needs_weights" | "needs_env" | "needs_both";

export interface ModelStatusResponse {
  model_id: string;
  friendly_name: string;
  weights_ready: boolean;
  env_ready: boolean;
  weights_size_mb: number | null;
  status: ModelStatus;
}

export interface ModelPrepareResponse {
  model_id: string;
  message: string;
  task_id: string;
}

export interface DeploymentAnalysisPayload {
  project_id: string;
  folder_path: string;
  detection_model: DetectionModel;
  classification_model: ClassificationModel;
}

export interface JobCreate {
  type: JobType;
  payload: Record<string, unknown>;
}

export interface JobResponse {
  id: string;
  type: string;
  status: string;
  progress_current: number;
  progress_total: number | null;
  payload: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface RunQueueResponse {
  message: string;
  jobs_started: number;
  job_ids: string[];
}

// File types
export interface DetectionResponse {
  id: string;
  category: string;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  species: string | null;
  species_confidence: number | null;
}

export interface FileResponse {
  id: string;
  deployment_id: string;
  file_path: string;
  file_type: string;
  file_format: string;
  size_bytes: number | null;
  width_px: number | null;
  height_px: number | null;
  timestamp: string;
  created_at: string;
}

export interface FileWithDetections extends FileResponse {
  detections: DetectionResponse[];
}

// Model options for deployment analysis
export const DETECTION_MODELS: DetectionModel[] = [
  "MD5A-0-0",
  "MD5B-0-0",
];

export const CLASSIFICATION_MODELS: ClassificationModel[] = [
  "EUR-DF-v1-3",
  "NAM-ADS-v1",
];
