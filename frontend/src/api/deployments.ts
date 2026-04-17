/**
 * Deployment API endpoints.
 */

import { api } from "../lib/api-client";
import type {
  BulkRelinkRequest,
  BulkRelinkResponse,
  DeploymentDetectionCategories,
  DeploymentFileCounts,
  DeploymentInfo,
  DeploymentResponse,
  DeploymentStatsOnly,
  DeploymentTopSpecies,
  DeploymentUpdate,
  DeploymentVerification,
  GroupBrokenGroup,
  GroupBrokenItem,
  GroupBrokenRequest,
  GroupBrokenResponse,
  SuggestRelinkTargetRequest,
  SuggestRelinkTargetResponse,
} from "./types";

export type {
  BulkRelinkRequest,
  BulkRelinkResponse,
  DeploymentDetectionCategories,
  DeploymentFileCounts,
  DeploymentInfo,
  DeploymentResponse,
  DeploymentStatsOnly,
  DeploymentTopSpecies,
  DeploymentUpdate,
  DeploymentVerification,
  GroupBrokenGroup,
  GroupBrokenItem,
  GroupBrokenRequest,
  GroupBrokenResponse,
  SuggestRelinkTargetRequest,
  SuggestRelinkTargetResponse,
};

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

  /**
   * Investigation-level payload for the Deployments → Info sheet.
   * Read-only snapshot of the deployment's counts, confidences, and
   * first/last capture timestamps.
   */
  getInfo: (id: string) =>
    api.get<DeploymentInfo>(`/api/deployments/${id}/info`),

  /**
   * Re-stat the deployment's folder_path and refresh folder_status.
   */
  checkFolder: (id: string) =>
    api.post<DeploymentResponse>(`/api/deployments/${id}/check-folder`, {}),

  /**
   * Relink multiple deployments at once. Returns per-item success / failure.
   */
  bulkRelink: (data: BulkRelinkRequest) =>
    api.post<BulkRelinkResponse>("/api/deployments/bulk-relink", data),

  /**
   * Suggest a replacement folder for a missing deployment path by
   * scanning siblings of its deepest existing ancestor.
   */
  suggestRelinkTarget: (data: SuggestRelinkTargetRequest) =>
    api.post<SuggestRelinkTargetResponse>(
      "/api/deployments/suggest-relink-target",
      data
    ),

  /**
   * Group a list of broken deployments by their deepest missing
   * ancestor and return per-group auto-suggested replacements.
   */
  groupBroken: (data: GroupBrokenRequest) =>
    api.post<GroupBrokenResponse>("/api/deployments/group-broken", data),
};
