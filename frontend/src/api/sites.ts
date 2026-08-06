/**
 * Site API endpoints.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";
import type {
  CsvImportPreview,
  CsvImportResult,
  SiteCreate,
  SiteImportRow,
  SiteDetectionCategories,
  SiteFileCounts,
  SiteInfo,
  SiteResponse,
  SiteTopSpecies,
  SiteUpdate,
  SiteVerification,
  SiteWithStats,
} from "./types";

// Re-export types
export type {
  SiteCreate,
  SiteDetectionCategories,
  SiteFileCounts,
  SiteInfo,
  SiteResponse,
  SiteTopSpecies,
  SiteUpdate,
  SiteVerification,
  SiteWithStats,
};

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

  /**
   * List sites with deployment counts (for metadata table)
   */
  listWithStats: (projectId: string) =>
    api.get<SiteWithStats[]>(`/api/sites/with-stats?project_id=${projectId}`),

  /**
   * Investigation-level payload for the Sites → Info sheet.
   * Aggregates across every deployment at this site.
   */
  getInfo: (id: string) => api.get<SiteInfo>(`/api/sites/${id}/info`),

  /**
   * Check a site CSV without writing anything.
   */
  importPreview: (projectId: string, file: File) =>
    api.upload<CsvImportPreview<SiteImportRow>>(
      `/api/sites/import/preview?project_id=${projectId}`,
      file
    ),

  /**
   * Import a site CSV, all or nothing. The file is sent again rather than
   * the previewed rows, so the backend checks one thing in one way.
   */
  importCsv: (projectId: string, file: File) =>
    api.upload<CsvImportResult>(`/api/sites/import?project_id=${projectId}`, file),
};
