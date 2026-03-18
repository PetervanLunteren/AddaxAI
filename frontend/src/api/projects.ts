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
  ProjectResponse,
  ProjectUpdate,
  ProjectWithStats,
} from "./types";

// Re-export types for convenience
export type { ProjectCreate, ProjectResponse, ProjectUpdate, ProjectWithStats };

export const projectsApi = {
  /**
   * List all projects
   */
  getProjects: () => api.get<ProjectResponse[]>("/api/projects"),

  /**
   * List all projects (alias for getProjects)
   */
  list: () => api.get<ProjectResponse[]>("/api/projects"),

  /**
   * Create a new project
   */
  create: (data: ProjectCreate) =>
    api.post<ProjectResponse>("/api/projects", data),

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
};
