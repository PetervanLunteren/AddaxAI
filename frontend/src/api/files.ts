/**
 * Files API client
 */

import { api } from "../lib/api-client";
import type { FileResponse, FileWithDetections } from "./types";

export const filesApi = {
  /**
   * Get file by ID with detections
   */
  get: async (id: string, options?: { signal?: AbortSignal }): Promise<FileWithDetections> => {
    return api.get<FileWithDetections>(`/api/files/${id}`, options);
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
    data: { verified?: boolean; notes?: string; favorited?: boolean }
  ): Promise<FileResponse> => {
    return api.patch<FileResponse>(`/api/files/${id}`, data);
  },
};
