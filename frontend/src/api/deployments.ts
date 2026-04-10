/**
 * Deployment API endpoints.
 */

import { api } from "../lib/api-client";
import type { DeploymentResponse, DeploymentUpdate, DeploymentStatsOnly } from "./types";

export type { DeploymentResponse, DeploymentUpdate, DeploymentStatsOnly };

export const deploymentsApi = {
  /**
   * List deployments, optionally filtered by site or project
   */
  list: (params?: { siteId?: string; projectId?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.siteId) searchParams.set("site_id", params.siteId);
    if (params?.projectId) searchParams.set("project_id", params.projectId);
    const qs = searchParams.toString();
    return api.get<DeploymentResponse[]>(`/api/deployments${qs ? `?${qs}` : ""}`);
  },

  /**
   * Get deployment by ID
   */
  get: (id: string) => api.get<DeploymentResponse>(`/api/deployments/${id}`),

  /**
   * Update deployment
   */
  update: (id: string, data: DeploymentUpdate) =>
    api.patch<DeploymentResponse>(`/api/deployments/${id}`, data),

  /**
   * Delete deployment
   */
  delete: (id: string) => api.delete<void>(`/api/deployments/${id}`),

  /**
   * Get bulk stats (file/event/detection counts) for all deployments in a project
   */
  getBulkStats: (projectId: string) =>
    api.get<Record<string, DeploymentStatsOnly>>(
      `/api/deployments/bulk-stats?project_id=${projectId}`
    ),
};
