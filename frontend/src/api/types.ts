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

export type DetectionModel = "MegaDetector 5A" | "MegaDetector v1000 Redwood";
export type ClassificationModel = "Europe" | "Africa";

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

// Model options for deployment analysis
export const DETECTION_MODELS: DetectionModel[] = [
  "MegaDetector 5A",
  "MegaDetector v1000 Redwood",
];

export const CLASSIFICATION_MODELS: ClassificationModel[] = [
  "Europe",
  "Africa",
];
