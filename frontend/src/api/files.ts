/**
 * Files API client
 */

import { api } from "../lib/api-client";
import type { FileResponse, FileWithDetections, ObservationType } from "./types";

export const filesApi = {
  /**
   * List files with optional filters
   */
  list: async (params?: {
    deployment_id?: string;
    project_id?: string;
    observation_type?: ObservationType;
    skip?: number;
    limit?: number;
  }): Promise<FileResponse[]> => {
    const searchParams = new URLSearchParams();
    if (params?.deployment_id) searchParams.set("deployment_id", params.deployment_id);
    if (params?.project_id) searchParams.set("project_id", params.project_id);
    if (params?.observation_type) searchParams.set("observation_type", params.observation_type);
    if (params?.skip !== undefined) searchParams.set("skip", params.skip.toString());
    if (params?.limit !== undefined) searchParams.set("limit", params.limit.toString());

    const query = searchParams.toString();
    const url = query ? `/api/files?${query}` : "/api/files";

    return api.get<FileResponse[]>(url);
  },

  /**
   * Get file by ID with detections
   */
  get: async (id: string): Promise<FileWithDetections> => {
    return api.get<FileWithDetections>(`/api/files/${id}`);
  },

  /**
   * Get observation type counts for a project
   */
  getObservationTypeStats: async (projectId: string): Promise<Record<string, number>> => {
    return api.get<Record<string, number>>(
      `/api/files/stats/observation-types?project_id=${projectId}`
    );
  },

  /**
   * Update file verification status and/or notes
   */
  update: async (
    id: string,
    data: { verified?: boolean; notes?: string }
  ): Promise<FileResponse> => {
    return api.patch<FileResponse>(`/api/files/${id}`, data);
  },
};
