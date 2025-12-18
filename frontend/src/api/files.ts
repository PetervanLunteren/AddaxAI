/**
 * Files API client
 */

import { apiClient } from "./client";
import type { FileResponse, FileWithDetections } from "./types";

export const filesApi = {
  /**
   * List files with optional filters
   */
  list: async (params?: {
    deployment_id?: string;
    project_id?: string;
    skip?: number;
    limit?: number;
  }): Promise<FileResponse[]> => {
    const searchParams = new URLSearchParams();
    if (params?.deployment_id) searchParams.set("deployment_id", params.deployment_id);
    if (params?.project_id) searchParams.set("project_id", params.project_id);
    if (params?.skip !== undefined) searchParams.set("skip", params.skip.toString());
    if (params?.limit !== undefined) searchParams.set("limit", params.limit.toString());

    const query = searchParams.toString();
    const url = query ? `/api/files?${query}` : "/api/files";

    return apiClient.get<FileResponse[]>(url);
  },

  /**
   * Get file by ID with detections
   */
  get: async (id: string): Promise<FileWithDetections> => {
    return apiClient.get<FileWithDetections>(`/api/files/${id}`);
  },
};
