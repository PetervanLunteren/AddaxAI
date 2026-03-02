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
  detection_model_id: string;
  classification_model_id: string | null;
  embedding_model_id: string | null;
  excluded_classes: string[];
  shortcut_labels: Record<string, { value: string; category: string; species: string | null }>;
  country_code?: string | null;
  state_code?: string | null;
  video_fps: number;
  detection_threshold: number;
  event_smoothing: boolean;
  taxonomic_rollup: boolean;
  taxonomic_rollup_threshold: number;
  independence_interval: number;
  min_cluster_size: number;
  min_samples: number;
}

export interface ProjectUpdate {
  name?: string | null;
  description?: string | null;
  detection_model_id?: string | null;
  classification_model_id?: string | null;
  embedding_model_id?: string | null;
  excluded_classes?: string[] | null;
  shortcut_labels?: Record<string, { value: string; category: string; species: string | null }> | null;
  country_code?: string | null;
  state_code?: string | null;
  video_fps?: number | null;
  detection_threshold?: number | null;
  event_smoothing?: boolean | null;
  taxonomic_rollup?: boolean | null;
  taxonomic_rollup_threshold?: number | null;
  independence_interval?: number | null;
  min_cluster_size?: number | null;
  min_samples?: number | null;
}

export interface ProjectResponse {
  id: string;
  name: string;
  description: string | null;
  detection_model_id: string;
  classification_model_id: string | null;
  embedding_model_id: string | null;
  excluded_classes: string[];
  shortcut_labels: Record<string, { value: string; category: string; species: string | null }>;
  country_code: string | null;
  state_code: string | null;
  video_fps: number;
  detection_threshold: number;
  event_smoothing: boolean;
  taxonomic_rollup: boolean;
  taxonomic_rollup_threshold: number;
  independence_interval: number;
  min_cluster_size: number;
  min_samples: number;
  postprocessing_settings_hash: string | null;
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
  | "event_computation"
  | "postprocessing";

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

// Observation type (Camtrap DP observationType vocabulary)
export type ObservationType = "animal" | "human" | "vehicle" | "blank" | "unknown" | "unclassified";

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
  classification_method: string | null;
  frame_number: number | null;
  verified: boolean;
  verified_at: string | null;
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
  best_frame_number: number | null;
  best_frame_path: string | null;
  frame_rate: number | null;
  observation_type: ObservationType;
  verified: boolean;
  verified_at: string | null;
  notes: string | null;
  favorited: boolean;
  source_video_id: string | null;
  source_frame_number: number | null;
}

export interface FileWithDetections extends FileResponse {
  detections: DetectionResponse[];
}

// Event filter types
export type VerificationFilter =
  | "all"
  | "fully_verified"
  | "not_fully_verified"
  | "unverified_representative"
  | "verified_representative"
  | "none_verified";

export interface EventFilterParams {
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  species?: string[];
  verification?: VerificationFilter;
  min_confidence?: number;
  max_confidence?: number;
}

export interface EventFilterOptions {
  species: string[];
  date_range: { min: string; max: string } | null;
  species_event_counts: Record<string, number>;
}

// Event types
export interface EventSummary {
  id: string;
  deployment_id: string;
  start_time: string;
  end_time: string;
  file_count: number;
  representative_file_id: string | null;
  site_name: string | null;
  species: string[];
  observation_type: string;
  observation_types: string[];
  image_count: number;
  frame_count: number;
  video_count: number;
  verified_count: number;
  total_count: number;
}

export interface EventWithFiles {
  id: string;
  deployment_id: string;
  start_time: string;
  end_time: string;
  file_count: number;
  representative_file_id: string | null;
  created_at: string;
  site_name: string | null;
  files: FileWithDetections[];
}

export interface AdjacentEventsResponse {
  previous_id: string | null;
  next_id: string | null;
  next_unverified_id: string | null;
  current_index: number;
  total_count: number;
}

// Detection create/update types
export interface DetectionCreate {
  file_id: string;
  category: string;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  species?: string | null;
}

export interface DetectionUpdate {
  category?: string;
  bbox_x?: number;
  bbox_y?: number;
  bbox_width?: number;
  bbox_height?: number;
  species?: string | null;
  species_confidence?: number | null;
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

// Model Info types (for UI dropdowns)
export interface ModelInfo {
  model_id: string;
  friendly_name: string;
  emoji: string;
  type: "detection" | "classification" | "embedding";
  description: string;
  description_short?: string | null;
  developer?: string | null;
  owner?: string | null;
  info_url?: string | null;
  citation?: string | null;
  license?: string | null;
  min_app_version?: string | null;
  embedding_dim?: number | null;
}

// Taxonomy types
export interface TaxonomyNode {
  id: string;
  name: string;
  level: number;
  children: TaxonomyNode[];
  selected: boolean;
}

export interface TaxonomyResponse {
  tree: TaxonomyNode[];
  all_classes: string[];
}

// SpeciesNet Locations types
export interface LocationsResponse {
  countries: Record<string, string>;  // Display name -> ISO code
  us_states: Record<string, string>;  // Display name -> State code
}

// Similarity types
export interface SimilarityFilters {
  species?: string[];
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  min_confidence?: number;
  category?: string;
  verified?: boolean;
}

export interface SortRequest {
  filters?: SimilarityFilters;
  reverse?: boolean;
}

export interface SearchRequest {
  anchor_detection_id: string;
  filters?: SimilarityFilters;
  limit?: number;
  threshold?: number;
}

export interface CropBbox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface DetectionSummary {
  detection_id: string;
  file_id: string;
  species: string | null;
  species_confidence: number | null;
  confidence: number;
  category: string;
  verified: boolean;
  classification_method: string | null;
  distance_to_centroid: number | null;
  similarity: number | null;
  neighbor_agreement: number | null;
  neighbor_top_label: string | null;
  site_name: string | null;
  deployment_id: string | null;
  timestamp: string | null;
  crop_url: string;
  crop_bbox: CropBbox | null;
}

export interface SortResponse {
  detections: DetectionSummary[];
  total_detections: number;
}

export interface SearchResponse {
  anchor: DetectionSummary;
  results: DetectionSummary[];
  total_results: number;
  threshold_applied: number;
}

export interface SimilarityStatsResponse {
  total_detections: number;
  embedded_detections: number;
  missing_embeddings: number;
  embedding_model_id: string | null;
  embedding_dimension: number | null;
}
