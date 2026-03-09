/**
 * Project API endpoints.
 *
 * Following DEVELOPERS.MD principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";
import type {
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
   * Get top species counts, optionally filtered by confidence threshold
   */
  getSpeciesStats: (id: string, threshold?: number) =>
    api.get<{ species: string; count: number }[]>(
      `/api/projects/${id}/species-stats${threshold ? `?threshold=${threshold}` : ""}`
    ),

  /**
   * List custom species for a project
   */
  getCustomSpecies: (projectId: string) =>
    api.get<{ id: string; name: string }[]>(`/api/projects/${projectId}/custom-species`),

  /**
   * Create a custom species for a project
   */
  createCustomSpecies: (projectId: string, name: string) =>
    api.post<{ id: string; name: string }>(`/api/projects/${projectId}/custom-species`, { name }),

  /**
   * Get independent event counts per species for a given interval
   */
  getIndependentEventStats: (
    id: string,
    interval: number,
    threshold?: number,
  ) =>
    api.get<{ total: number; species: { species: string; count: number }[] }>(
      `/api/projects/${id}/independent-event-stats?interval=${interval}${threshold ? `&threshold=${threshold}` : ""}`
    ),
};
