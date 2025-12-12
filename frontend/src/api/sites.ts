/**
 * Site API endpoints.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";
import type { SiteCreate, SiteResponse, SiteUpdate } from "./types";

export const sitesApi = {
  /**
   * List all sites, optionally filtered by project
   */
  list: (projectId?: string) => {
    const endpoint = projectId
      ? `/api/sites?project_id=${projectId}`
      : "/api/sites";
    return api.get<SiteResponse[]>(endpoint);
  },

  /**
   * Create a new site
   */
  create: (data: SiteCreate) => api.post<SiteResponse>("/api/sites", data),

  /**
   * Get site by ID
   */
  get: (id: string) => api.get<SiteResponse>(`/api/sites/${id}`),

  /**
   * Update site
   */
  update: (id: string, data: SiteUpdate) =>
    api.patch<SiteResponse>(`/api/sites/${id}`, data),

  /**
   * Delete site
   */
  delete: (id: string) => api.delete<void>(`/api/sites/${id}`),
};
