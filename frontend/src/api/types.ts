/**
 * TypeScript types for API requests and responses.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Matches backend Pydantic schemas
 */

// Project types

/**
 * Workflow mode for a project.
 *
 * - `research`: the full project workspace with Sites, Deployments,
 *   Insights, Exports. Default for new projects created via the
 *   Research projects flow.
 * - `folder_run`: the legacy-style point-at-a-folder workflow.
 *   Hidden from the Research projects list; surfaced separately on
 *   the home screen.
 */
export type ProjectMode = "folder_run" | "research";

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
  mode?: ProjectMode;
  folder_run_state?: Record<string, unknown> | null;
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
  /** Promotion sets this to "research". */
  mode?: ProjectMode | null;
  folder_run_state?: Record<string, unknown> | null;
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
  mode: ProjectMode;
  folder_run_state: Record<string, unknown> | null;
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

export interface SiteFileCounts {
  total: number;
  images: number;
  videos: number;
}

export interface SiteTopSpecies {
  label: string;
  display_name: string | null;
  count: number;
}

export interface SiteDetectionCategories {
  animal: number;
  person: number;
  vehicle: number;
  empty: number;
}

export interface SiteVerification {
  verified: number;
  total: number;
}

export interface SiteInfo {
  site_id: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation_m: number | null;
  habitat_type: string | null;
  notes: string | null;
  tags: Record<string, string>;
  deployment_count: number;
  files: SiteFileCounts;
  total_size_bytes: number;
  verification: SiteVerification;
  event_count: number;
  observation_count: number;
  detection_categories: SiteDetectionCategories;
  top_species: SiteTopSpecies[];
  /** Sum of per-deployment (end - start + 1) days. Null when any
   * deployment is open-ended or the site has no deployments. */
  trap_nights: number | null;
  observation_rate_per_100_trap_nights: number | null;
  first_captured_at_local: string | null;
  last_captured_at_local: string | null;
}

/** One row in DeploymentResponse.warnings — a file the pipeline skipped
 * for a non-fatal reason. `path` is the file path (relative or absolute
 * depending on which phase recorded it). `reason` is present for
 * decoder failures, absent for missing-timestamp skips. */
export interface DeploymentWarning {
  type: "missing_timestamp" | "video_processing_failure" | string;
  path: string;
  reason?: string;
}

// Deployment types
export interface DeploymentResponse {
  id: string;
  project_id: string;
  /** Null means the deployment has no camera site assigned
   * (deployment-agnostic batch, unknown location, or data spanning
   * multiple sites). Features that need GPS skip null-site rows. */
  site_id: string | null;
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
  /** Non-fatal issues from this deployment's analysis run. Null when
   * the run had nothing to flag. */
  warnings: DeploymentWarning[] | null;
}

export interface DeploymentUpdate {
  /** Null clears the site (user moved deployment to a site-less batch). */
  site_id?: string | null;
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

export interface DeploymentFileCounts {
  total: number;
  images: number;
  videos: number;
}

export interface DeploymentTopSpecies {
  label: string;
  display_name: string | null;
  count: number;
}

export interface DeploymentDetectionCategories {
  animal: number;
  person: number;
  vehicle: number;
  empty: number;
}

export interface DeploymentVerification {
  verified: number;
  total: number;
}

export interface DeploymentInfo {
  deployment_id: string;
  folder_path: string | null;
  /** Null when the deployment has no camera site assigned. */
  site_id: string | null;
  /** Null when the deployment has no camera site assigned. */
  site_name: string | null;
  start_date_local: string;
  end_date_local: string | null;
  files: DeploymentFileCounts;
  /** Sum of File.size_bytes across files in this deployment. */
  total_size_bytes: number;
  verification: DeploymentVerification;
  event_count: number;
  /** Sum of EventObservation.max_n across all events in this deployment. */
  observation_count: number;
  detection_categories: DeploymentDetectionCategories;
  top_species: DeploymentTopSpecies[];
  /** (end - start) + 1 days. Null when end_date_local is null. */
  trap_nights: number | null;
  /** observations / trap_nights * 100. Null when trap_nights is null or 0. */
  observation_rate_per_100_trap_nights: number | null;
  /** Null when no detections pass the threshold-with-verified filter. */
  mean_detection_confidence: number | null;
  /** Null when no detection has a classification label. */
  mean_classification_confidence: number | null;
  first_captured_at_local: string | null;
  last_captured_at_local: string | null;
  /** Non-fatal issues from this deployment's analysis run, persisted on
   * the deployment so they survive queue cleanup. Null when the run
   * had nothing to flag. */
  warnings: DeploymentWarning[] | null;
}

export interface SplitPreviewTarget {
  folder_path: string;
  name: string;
  image_count: number;
  video_count: number;
}

export interface SplitPreview {
  original_folder: string | null;
  depth: number;
  max_depth: number;
  can_decrease: boolean;
  can_increase: boolean;
  targets: SplitPreviewTarget[];
  /** Non-null when the split cannot proceed. UI shows it, disables OK. */
  blocked_reason: string | null;
}

export interface SplitResponse {
  created_deployment_ids: string[];
  message: string;
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
  /** All four bbox fields are null together for event-level observations
   * (a species seen in a video clip without a frame-anchored ROI). For
   * AI-produced and user-drawn detections they are all set. */
  bbox_x: number | null;
  bbox_y: number | null;
  bbox_width: number | null;
  bbox_height: number | null;
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
  flagged: boolean;
  flagged_at_utc: string | null;
  source_video_id: string | null;
  source_frame_number: number | null;
}

export interface FileWithDetections extends FileResponse {
  detections: DetectionResponse[];
}

// Shared verify-tab filter type. Most tabs use a simple binary (plus
// "all"): an event is verified when all its MaxN frames are verified
// (blank events fall back to "any file verified"); a file is verified
// when File.verified is true. The Observations tab adds "suspicious"
// — unverified detections whose nearest-neighbour label disagrees with
// the assigned label (post-filtered client-side from neighbor_agreement).
// Other tabs ignore "suspicious"; their dropdown never offers it.
export type VerificationFilter =
  | "all"
  | "verified"
  | "unverified"
  | "suspicious";

export type FlaggedFilter = "all" | "flagged" | "not_flagged";
export type FavoritedFilter = "all" | "favorited" | "not_favorited";
/** "show_only" = empties only, "hide" = no empties, "all" = both. */
export type EmptyFilter = "all" | "show_only" | "hide";

/** Sort modes shared across the verify tabs.
 *
 * Events and Files use the first four. Observations uses similarity by
 * default and supports `similarity_reverse`; the metadata-only modes
 * (`newest`, `oldest`, `cls_low`) are also available. `random` only
 * applies to Events / Files (it relies on a stable seed for paginated
 * grids and is not meaningful for the Observations grid). */
export type VerifySort =
  | "newest"
  | "oldest"
  | "random"
  | "cls_low"
  | "similarity"
  | "similarity_reverse";

export interface EventFilterParams {
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  labels?: string[];
  verification?: VerificationFilter;
  flagged?: FlaggedFilter;
  favorited?: FavoritedFilter;
  empty?: EmptyFilter;
  /** Detector confidence (Detection.confidence) range. Min handle is
   *  clamped client-side at the project's detection_threshold. */
  min_confidence?: number;
  max_confidence?: number;
  /** Classifier confidence (Detection.label_confidence) range. NULL
   *  classifications are excluded once either bound is set. */
  min_label_confidence?: number;
  max_label_confidence?: number;
  /** Sort mode. Default "newest". `random` requires `seed` for stable
   *  ordering across pagination and modal navigation. */
  sort?: VerifySort;
  seed?: number;
}

// File summary for the Files verify tab grid.
export interface FileSummaryDetection {
  id: string;
  category: string;
  confidence: number;
  /** All four bbox fields are null together for event-level observations. */
  bbox_x: number | null;
  bbox_y: number | null;
  bbox_width: number | null;
  bbox_height: number | null;
  label: string | null;
  label_taxonomy_id: string | null;
  /** Video detections carry their frame index; image detections are null. */
  frame_number: number | null;
}

export interface FileSummary {
  id: string;
  deployment_id: string;
  file_type: string;
  file_format: string | null;
  width_px: number | null;
  height_px: number | null;
  /** ISO 8601 with the project's local UTC offset. */
  captured_at_local: string;
  site_id: string | null;
  site_name: string | null;
  observation_type: string;
  observation_types: string[];
  labels: string[];
  display_labels: Record<string, string>;
  verified: boolean;
  favorited: boolean;
  flagged: boolean;
  source_video_id: string | null;
  /** Video rows expose this so the grid overlay can filter detections
   * to the one frame the thumbnail actually shows. Null for images. */
  best_frame_number: number | null;
  detections: FileSummaryDetection[];
}

export interface FileVerificationStats {
  total_files: number;
  verified_files: number;
}

export interface AdjacentFilesResponse {
  previous_id: string | null;
  next_id: string | null;
  next_unverified_id: string | null;
  current_index: number;
  total_count: number;
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
  /** Up to four file IDs picked by the backend for the event-card collage:
   *  one frame per dominant species first, then padded by max detection
   *  confidence. Empty when the event has no files. */
  collage_file_ids: string[];
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
  /**
   * AddaxAI rule: an event is verified when all its MaxN frames are
   * verified. Blank events (no MaxN) are verified when any file is
   * verified. Drives the corner verified badge on event cards.
   */
  is_verified: boolean;
  any_file_flagged: boolean;
  any_file_favorited: boolean;
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
  /** Events whose MaxN frames are all verified (blank-event fallback: any file verified). */
  events_fully_verified: number;
  events_total: number;
  total_files: number;
  verified_files: number;
  total_max_n_frames: number;
  verified_max_n_frames: number;
  total_observations: number;
  total_detections: number;
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
  /** Anchors the new box to a frame for videos (so the overlay still
   * renders it). Null for images. */
  frame_number?: number | null;
}

/** Event-level observation: user spots an animal in a clip (or anywhere
 * else without a bbox). File-level fact; no bbox, no frame_number —
 * matches Camtrap-DP observationLevel="event". */
export interface DetectionCreateObservation {
  file_id: string;
  category: string;
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
  /** Geographic region the cls model is trained for. Drives the
   *  grouping in classification dropdowns. `null` for detection /
   *  embedding models, and for any cls manifest not yet annotated. */
  region?:
    | "global"
    | "africa"
    | "americas"
    | "asia"
    | "europe"
    | "oceania"
    | null;
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

// Observations types (embedding-backed sort + search for the Observations
// verify tab). Filter shape carries site/date/label predicates; the
// underlying cosine-similarity algorithm lives in the subprocess script.
export interface ObservationFilters {
  labels?: string[];
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  min_confidence?: number;
  max_confidence?: number;
  min_label_confidence?: number;
  max_label_confidence?: number;
  category?: string;
  verified?: boolean;
}

/** Subset of VerifySort that is valid for Observations. */
export type ObservationSort =
  | "similarity"
  | "similarity_reverse"
  | "newest"
  | "oldest"
  | "cls_low";

export interface SortRequest {
  filters?: ObservationFilters;
  sort?: ObservationSort;
  /** Per-user memory budget for one sort, set in the Observations
   * view-options popover (localStorage). Backend defaults to 20000
   * when omitted. */
  max_detections?: number;
}

export interface SearchRequest {
  anchor_detection_id: string;
  filters?: ObservationFilters;
  limit?: number;
  threshold?: number;
  /** Same cap the sort endpoint takes; bounds the candidate pool the
   * subprocess loads. */
  max_detections?: number;
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
  neighbor_top_display_name: string | null;
  site_name: string | null;
  deployment_id: string | null;
  /** ISO 8601 with the project's local UTC offset. */
  captured_at_local: string | null;
  crop_url: string;
  crop_bbox: CropBbox | null;
  /** Video detections carry their frame index; image detections are null. */
  frame_number: number | null;
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

export interface ObservationStatsResponse {
  total_detections: number;
  embedded_detections: number;
  missing_embeddings: number;
  embedding_model_id: string | null;
  embedding_dimension: number | null;
}

export interface MissingModel {
  model_id: string;
  friendly_name: string;
  emoji: string;
  category: "detection" | "classification" | "embedding" | "unknown";
  needs_weights: boolean;
  needs_env: boolean;
}

export interface ProjectModelReadiness {
  ready: boolean;
  missing: MissingModel[];
}
