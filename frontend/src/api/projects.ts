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
};
