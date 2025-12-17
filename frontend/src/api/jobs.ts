/**
 * Job API endpoints.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";
import type { JobCreate, JobResponse, RunQueueResponse } from "./types";

// Re-export types
export type { JobCreate, JobResponse, RunQueueResponse };

export const jobsApi = {
  /**
   * List all jobs, optionally filtered by type, status, and/or project_id
   */
  list: (params?: { type?: string; status?: string; project_id?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.type) searchParams.append("type", params.type);
    if (params?.status) searchParams.append("status", params.status);
    if (params?.project_id) searchParams.append("project_id", params.project_id);

    const query = searchParams.toString();
    const endpoint = query ? `/api/jobs?${query}` : "/api/jobs";

    return api.get<JobResponse[]>(endpoint);
  },

  /**
   * Create a new job (add to queue)
   */
  create: (data: JobCreate) => api.post<JobResponse>("/api/jobs", data),

  /**
   * Get job by ID
   */
  get: (id: string) => api.get<JobResponse>(`/api/jobs/${id}`),

  /**
   * Delete job (remove from queue)
   */
  delete: (id: string) => api.delete<void>(`/api/jobs/${id}`),

  /**
   * Trigger queue processing, optionally for a specific project
   */
  runQueue: (project_id?: string) => {
    const endpoint = project_id
      ? `/api/jobs/run-queue?project_id=${project_id}`
      : "/api/jobs/run-queue";
    return api.post<RunQueueResponse>(endpoint, {});
  },
};
