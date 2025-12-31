/**
 * Deployment Queue API endpoints.
 *
 * Following DEVELOPERS.MD principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";

export interface DeploymentQueueEntry {
  id: string;
  project_id: string;
  folder_path: string;
  site_id: string | null;
  status: "pending" | "processing" | "completed" | "failed";
  created_at: string;
  processed_at: string | null;
  error: string | null;
  deployment_id: string | null;
}

export interface DeploymentQueueCreate {
  project_id: string;
  folder_path: string;
  site_id?: string | null;
}

export interface ProcessQueueRequest {
  project_id: string;
}

export const deploymentQueueApi = {
  /**
   * List all queue entries for a project
   */
  list: (projectId: string, status?: string) => {
    const params = new URLSearchParams({ project_id: projectId });
    if (status) params.append("status", status);
    return api.get<DeploymentQueueEntry[]>(
      `/api/deployment-queue?${params.toString()}`
    );
  },

  /**
   * Create a new queue entry
   */
  create: (data: DeploymentQueueCreate) =>
    api.post<DeploymentQueueEntry>("/api/deployment-queue", data),

  /**
   * Get queue entry by ID
   */
  get: (id: string) =>
    api.get<DeploymentQueueEntry>(`/api/deployment-queue/${id}`),

  /**
   * Remove entry from queue
   */
  remove: (id: string) => api.delete(`/api/deployment-queue/${id}`),

  /**
   * Process all pending entries in queue
   */
  process: (data: ProcessQueueRequest) =>
    api.post("/api/deployment-queue/process", data),
};
