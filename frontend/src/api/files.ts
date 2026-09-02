/**
 * Files API client
 */

import { api } from "../lib/api-client";
import type {
  FileResponse,
  FileWithDetections,
  FilmstripResponse,
} from "./types";

export const filesApi = {
  /**
   * Get file by ID with detections
   */
  get: async (id: string, options?: { signal?: AbortSignal }): Promise<FileWithDetections> => {
    return api.get<FileWithDetections>(`/api/files/${id}`, options);
  },

  /**
   * Get an on-demand filmstrip (evenly-spaced low-res frames) for a video.
   */
  getFilmstrip: async (id: string): Promise<FilmstripResponse> => {
    return api.get<FilmstripResponse>(`/api/files/${id}/filmstrip`);
  },

  /**
   * Update file verification status and/or notes
   */
  update: async (
    id: string,
    data: {
      verified?: boolean;
      notes?: string;
      favorited?: boolean;
      flagged?: boolean;
    }
  ): Promise<FileResponse> => {
    return api.patch<FileResponse>(`/api/files/${id}`, data);
  },

  /**
   * Verify or unverify up to 500 files in one request. Same rule per
   * file as `update(id, { verified })`.
   */
  bulkVerify: async (
    ids: string[],
    verified = true,
  ): Promise<{ updated_count: number }> => {
    return api.post<{ updated_count: number }>("/api/files/bulk-verify", {
      file_ids: ids,
      verified,
    });
  },
};
