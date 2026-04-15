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
  shortcut_labels: Record<string, { value: string; category: string; label: string | null }>;
  country_code?: string | null;
  state_code?: string | null;
  /** IANA timezone name, e.g. "Europe/Amsterdam". Required. */
  timezone: string;
  video_fps: number;
  detection_threshold: number;
  event_smoothing: boolean;
  smoothing_strength: string;
  taxonomic_rollup: boolean;
  taxonomic_rollup_threshold: number;
  independence_interval: number;
  min_cluster_size: number;
  min_samples: number;
  detection_batch_size: number | null;
  classification_batch_size: number | null;
  embedding_batch_size: number | null;
}

export interface ProjectUpdate {
  name?: string | null;
  description?: string | null;
  detection_model_id?: string | null;
  classification_model_id?: string | null;
  embedding_model_id?: string | null;
  excluded_classes?: string[] | null;
  shortcut_labels?: Record<string, { value: string; category: string; label: string | null }> | null;
  country_code?: string | null;
  state_code?: string | null;
  /** IANA timezone name, optional on update. */
  timezone?: string | null;
  video_fps?: number | null;
  detection_threshold?: number | null;
  event_smoothing?: boolean | null;
  smoothing_strength?: string | null;
  taxonomic_rollup?: boolean | null;
  taxonomic_rollup_threshold?: number | null;
  independence_interval?: number | null;
  min_cluster_size?: number | null;
  min_samples?: number | null;
  detection_batch_size?: number | null;
  classification_batch_size?: number | null;
  embedding_batch_size?: number | null;
}

export interface ProjectResponse {
  id: string;
  name: string;
  description: string | null;
  detection_model_id: string;
  classification_model_id: string | null;
  embedding_model_id: string | null;
  excluded_classes: string[];
  shortcut_labels: Record<string, { value: string; category: string; label: string | null }>;
  country_code: string | null;
  state_code: string | null;
  timezone: string;
  video_fps: number;
  detection_threshold: number;
  event_smoothing: boolean;
  smoothing_strength: string;
  taxonomic_rollup: boolean;
  taxonomic_rollup_threshold: number;
  independence_interval: number;
  min_cluster_size: number;
  min_samples: number;
  detection_batch_size: number | null;
  classification_batch_size: number | null;
  embedding_batch_size: number | null;
  postprocessing_settings_hash: string | null;
  thumbnail_path: string | null;
  created_at_utc: string;
  updated_at_utc: string;
}

export interface ProjectWithStats extends ProjectResponse {
  site_count: number;
  deployment_count: number;
  file_count: number;
  observation_count: number;
  trap_nights: number;
}

// Custom label types
export interface CustomLabelResponse {
  id: string;
  name: string;
  level: string;
  taxon_class: string | null;
  taxon_order: string | null;
  taxon_family: string | null;
  taxon_genus: string | null;
  taxon_species: string | null;
}

export interface CustomLabelUpdate {
  name?: string | null;
  taxon_class?: string | null;
  taxon_order?: string | null;
  taxon_family?: string | null;
  taxon_genus?: string | null;
  taxon_species?: string | null;
}

export interface GBIFSuggestion {
  gbif_key: number;
  scientific_name: string;
  canonical_name: string;
  rank: string;
  taxon_class: string | null;
  taxon_order: string | null;
  taxon_family: string | null;
  taxon_genus: string | null;
  taxon_species: string | null;
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
  tags?: Record<string, string> | null;
}

export interface SiteUpdate {
  name?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  elevation_m?: number | null;
  habitat_type?: string | null;
  notes?: string | null;
  tags?: Record<string, string> | null;
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
  tags: Record<string, string>;
  created_at_utc: string;
}

export interface SiteWithStats extends SiteResponse {
  deployment_count: number;
}

// Deployment types
export interface DeploymentResponse {
  id: string;
  site_id: string;
  folder_path: string | null;
  folder_status: "valid" | "needs_relink";
  last_validated_at_utc: string | null;
  start_date_local: string;
  end_date_local: string | null;
  camera_model: string | null;
  camera_serial: string | null;
  notes: string | null;
  tags: Record<string, string>;
  datetime_offset_seconds: number | null;
  created_at_utc: string;
}

export interface DeploymentUpdate {
  start_date_local?: string | null;
  end_date_local?: string | null;
  camera_model?: string | null;
  camera_serial?: string | null;
  notes?: string | null;
  tags?: Record<string, string> | null;
  datetime_offset_seconds?: number | null;
}

export interface BulkRelinkItem {
  deployment_id: string;
  new_folder_path: string;
}

export interface BulkRelinkRequest {
  replacements: BulkRelinkItem[];
}

export interface BulkRelinkResultItem {
  deployment_id: string;
  success: boolean;
  files_rewritten: number;
  mismatches: string[];
}

export interface BulkRelinkResponse {
  results: BulkRelinkResultItem[];
}

export interface SuggestRelinkTargetRequest {
  missing_path: string;
}

export interface SuggestRelinkTargetResponse {
  existing_parent: string | null;
  suggested_path: string | null;
  candidates: string[];
}

export interface GroupBrokenItem {
  id: string;
  folder_path: string;
}

export interface GroupBrokenRequest {
  items: GroupBrokenItem[];
}

export interface GroupBrokenGroup {
  prefix: string;
  existing_parent: string | null;
  suggested_path: string | null;
  items: GroupBrokenItem[];
}

export interface GroupBrokenResponse {
  groups: GroupBrokenGroup[];
}

export interface DeploymentStatsOnly {
  file_count: number;
  event_count: number;
  detection_count: number;
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
  created_at_utc: string;
  started_at_utc: string | null;
  completed_at_utc: string | null;
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
  label: string | null;
  label_confidence: number | null;
  display_name: string | null;
  label_taxonomy_id: string | null;
  classification_method: string | null;
  frame_number: number | null;
  verified: boolean;
  verified_at_utc: string | null;
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
  /** ISO 8601 with the project's local UTC offset, e.g. "2026-04-14T07:30:00+02:00". */
  captured_at_local: string;
  created_at_utc: string;
  best_frame_number: number | null;
  best_frame_path: string | null;
  frame_rate: number | null;
  observation_type: ObservationType;
  verified: boolean;
  verified_at_utc: string | null;
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
  | "unverified_maxn"
  | "all_maxn_verified"
  | "some_maxn_verified"
  | "none_verified";

export interface EventFilterParams {
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  labels?: string[];
  verification?: VerificationFilter;
  min_confidence?: number;
  max_confidence?: number;
}

export interface EventFilterOptions {
  labels: string[];
  date_range: { min: string; max: string } | null;
  label_event_counts: Record<string, number>;
  display_labels?: Record<string, string>;
}

// MaxN frame reference
export interface MaxNFrame {
  file_id: string;
  label: string | null;
  label_taxonomy_id: string | null;
  max_n: number;
}

// Event types
export interface EventSummary {
  id: string;
  deployment_id: string;
  /** ISO 8601 with the project's local UTC offset. */
  event_start_local: string;
  event_end_local: string;
  file_count: number;
  thumbnail_file_id: string | null;
  max_n_frames: MaxNFrame[];
  site_name: string | null;
  labels: string[];
  display_labels?: Record<string, string>;
  observation_type: string;
  observation_types: string[];
  image_count: number;
  frame_count: number;
  video_count: number;
  verified_count: number;
  total_count: number;
  verified_maxn_count: number;
  total_maxn_count: number;
}

export interface EventWithFiles {
  id: string;
  deployment_id: string;
  /** ISO 8601 with the project's local UTC offset. */
  event_start_local: string;
  event_end_local: string;
  file_count: number;
  max_n_frames: MaxNFrame[];
  created_at_utc: string;
  site_name: string | null;
  files: FileWithDetections[];
}

export interface EventVerificationStats {
  total_files: number;
  verified_files: number;
  total_max_n_frames: number;
  verified_max_n_frames: number;
  total_observations: number;
  verified_detections: number;
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
  label?: string | null;
}

export interface DetectionUpdate {
  category?: string;
  bbox_x?: number;
  bbox_y?: number;
  bbox_width?: number;
  bbox_height?: number;
  label?: string | null;
  label_confidence?: number | null;
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
  // Per-pipeline default batch sizes the worker will use when the project's
  // batch_size override is null. Used to label the "Default" option in the
  // Performance card. Same numbers for every model in the same pipeline.
  default_batch_size_gpu: number;
  default_batch_size_cpu: number;
}

// Taxonomy types
export interface TaxonomyNode {
  id: string;
  name: string;
  level: number;
  children: TaxonomyNode[];
  selected: boolean;
  annotation?: string;
  count?: number;
  child_count?: number;
}

export interface TaxonomyResponse {
  tree: TaxonomyNode[];
  all_classes: string[];
}

export interface LabelTreeResponse {
  tree: TaxonomyNode[];
  all_leaf_ids: string[];
  label_event_counts: Record<string, number>;
  count_unit: string;
}

// Geofence types
export interface GeofenceResponse {
  has_geofence: boolean;
  countries?: Record<string, string>;
  us_states?: Record<string, string>;
  allowed_labels?: string[];
  excluded_labels?: string[];
  excluded_count?: number;
  total_count?: number;
}

// Similarity types
export interface SimilarityFilters {
  labels?: string[];
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
  label: string | null;
  label_confidence: number | null;
  display_name: string | null;
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
  /** ISO 8601 with the project's local UTC offset. */
  captured_at_local: string | null;
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
