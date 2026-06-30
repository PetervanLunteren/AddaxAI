/**
 * Project API endpoints.
 *
 * Following DEVELOPERS.MD principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";
import type {
  CustomLabelResponse,
  CustomLabelUpdate,
  GBIFSuggestion,
  ProjectCreate,
  ProjectMode,
  ProjectModelReadiness,
  ProjectResponse,
  ProjectUpdate,
  ProjectWithStats,
} from "./types";

// Re-export types for convenience
export type {
  ProjectCreate,
  ProjectMode,
  ProjectResponse,
  ProjectUpdate,
  ProjectWithStats,
};

/**
 * Backend filter values for the list endpoint. `research` and
 * `folder_run` map to the DB column; `all` is a query-only sentinel
 * that bypasses the filter.
 */
export type ProjectListMode = ProjectMode | "all";

/**
 * Banner payload returned by /deployments-without-site. Used by
 * GPS-dependent pages (Map, Activity overlap sun mode, Dashboard sun
 * bands, CamtrapDP / GeoJSON exports) to render a "X deployments
 * without a site" notice.
 */
export interface DeploymentsWithoutSiteResponse {
  count: number;
  deployment_ids: string[];
}

/** Payload for duplicating a project's structure into a new one. */
export interface ProjectDuplicatePayload {
  name: string;
  description?: string | null;
  classification_model_id?: string | null;
  excluded_classes: string[];
  country_code?: string | null;
  state_code?: string | null;
  copy_settings: boolean;
  copy_sites: boolean;
  copy_deployments: boolean;
}

export const projectsApi = {
  /**
   * List projects with statistics, filtered by workflow mode.
   *
   * Defaults to `research` so the Research projects list (and any
   * caller that does not pass a mode) excludes folder runs. Pass
   * `folder_run` for the home recents strip or `all` to include both.
   */
  getProjects: (mode: ProjectListMode = "research") =>
    api.get<ProjectWithStats[]>(`/api/projects?mode=${mode}`),

  /**
   * Alias for getProjects with the same default.
   */
  list: (mode: ProjectListMode = "research") =>
    api.get<ProjectWithStats[]>(`/api/projects?mode=${mode}`),

  /**
   * Create a new project
   */
  create: (data: ProjectCreate) =>
    api.post<ProjectResponse>("/api/projects", data),

  /**
   * Duplicate an existing project's structure into a new project.
   */
  duplicate: (sourceId: string, data: ProjectDuplicatePayload) =>
    api.post<ProjectResponse>(`/api/projects/${sourceId}/duplicate`, data),

  /**
   * Get project by ID
   */
  get: (id: string) => api.get<ProjectResponse>(`/api/projects/${id}`),

  /**
   * Update project
   */
  update: (id: string, data: ProjectUpdate) =>
    api.patch<ProjectResponse>(`/api/projects/${id}`, data),

  /**
   * Delete project
   */
  delete: (id: string) => api.delete<void>(`/api/projects/${id}`),

  /**
   * Get project with statistics
   */
  getWithStats: (id: string) =>
    api.get<ProjectWithStats>(`/api/projects/${id}/stats`),

  /**
   * Check whether every model configured for this project has its
   * weights + a valid env on disk. Drives the project-open setup
   * dialog and the pre-analysis safety check.
   */
  getModelReadiness: (id: string) =>
    api.get<ProjectModelReadiness>(`/api/projects/${id}/model-readiness`),

  /**
   * Reprocess classifications (apply/revert smoothing)
   */
  reprocess: (id: string) =>
    api.post<{ message: string; job_id: string }>(
      `/api/projects/${id}/reprocess`
    ),

  /**
   * Re-embed all detections with the current embedding model
   */
  reEmbed: (id: string) =>
    api.post<{ message: string; job_id: string | null }>(
      `/api/projects/${id}/re-embed`
    ),

  /**
   * Count of deployments in this project that have no camera site.
   * Feeds the NoSiteBanner on GPS-dependent pages.
   */
  getDeploymentsWithoutSite: (id: string) =>
    api.get<DeploymentsWithoutSiteResponse>(
      `/api/projects/${id}/deployments-without-site`,
    ),

  /**
   * Get postprocessing status (needs reprocessing?)
   */
  getPostprocessingStatus: (id: string) =>
    api.get<{ needs_reprocessing: boolean; has_classifications: boolean }>(
      `/api/projects/${id}/postprocessing-status`
    ),

  /**
   * Get count of detections at or above a confidence threshold
   */
  getDetectionCount: (id: string, threshold: number) =>
    api.get<{ count: number }>(
      `/api/projects/${id}/detection-count?threshold=${threshold}`
    ),

  /**
   * Get top label counts, optionally filtered by confidence threshold
   */
  getLabelStats: (id: string, threshold?: number) =>
    api.get<{ label: string; count: number }[]>(
      `/api/projects/${id}/label-stats${threshold ? `?threshold=${threshold}` : ""}`
    ),

  /**
   * Get taxonomy fields for all labels in a project (model + custom)
   */
  getLabelTaxonomyMap: (projectId: string) =>
    api.get<Record<string, {
      taxon_class: string | null;
      taxon_order: string | null;
      taxon_family: string | null;
      taxon_genus: string | null;
      taxon_species: string | null;
      common_name: string | null;
      scientific_name: string | null;
    }>>(`/api/projects/${projectId}/label-taxonomy-map`),

  /**
   * List custom labels for a project
   */
  getCustomLabels: (projectId: string) =>
    api.get<CustomLabelResponse[]>(`/api/projects/${projectId}/custom-labels`),

  /**
   * Create a custom label for a project
   */
  createCustomLabel: (projectId: string, name: string) =>
    api.post<CustomLabelResponse>(`/api/projects/${projectId}/custom-labels`, { name }),

  /**
   * Update a custom label (name and/or taxonomy fields)
   */
  updateCustomLabel: (projectId: string, labelId: string, data: CustomLabelUpdate) =>
    api.patch<CustomLabelResponse>(`/api/projects/${projectId}/custom-labels/${labelId}`, data),

  /**
   * Delete a custom label from a project
   */
  deleteCustomLabel: (projectId: string, labelId: string) =>
    api.delete<void>(`/api/projects/${projectId}/custom-labels/${labelId}`),

  /**
   * Upload a project card thumbnail image
   */
  uploadThumbnail: (id: string, file: File) =>
    api.upload<{ message: string }>(`/api/projects/${id}/thumbnail`, file),

  /**
   * Remove the project card thumbnail
   */
  deleteThumbnail: (id: string) =>
    api.delete<void>(`/api/projects/${id}/thumbnail`),

  /**
   * Search GBIF for species suggestions by vernacular name
   */
  gbifSuggest: (query: string) =>
    api.get<GBIFSuggestion[]>(`/api/projects/gbif/suggest?q=${encodeURIComponent(query)}`),

  /**
   * Get independent event counts per label for a given interval
   */
  getIndependentEventStats: (
    id: string,
    interval: number,
    threshold?: number,
  ) =>
    api.get<{ total: number; labels: { label: string; count: number }[] }>(
      `/api/projects/${id}/independent-event-stats?interval=${interval}${threshold ? `&threshold=${threshold}` : ""}`
    ),

  /**
   * Sum of MaxN (peak individuals per event), per label, for a given interval
   */
  getIndependentObservationStats: (
    id: string,
    interval: number,
    threshold?: number,
  ) =>
    api.get<{ total: number; labels: { label: string; count: number }[] }>(
      `/api/projects/${id}/independent-observation-stats?interval=${interval}${threshold ? `&threshold=${threshold}` : ""}`
    ),
};
