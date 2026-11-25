/**
 * Job API endpoints.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";
import type { JobResponse } from "./types";

// Re-export types
export type { JobResponse };

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
};
