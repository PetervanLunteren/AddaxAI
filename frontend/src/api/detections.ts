/**
 * Detections API client
 */

import { api } from "../lib/api-client";
import type { DetectionResponse, DetectionCreate, DetectionUpdate } from "./types";

export const detectionsApi = {
  /** Create a human-drawn detection. */
  create: async (data: DetectionCreate): Promise<DetectionResponse> => {
    return api.post<DetectionResponse>("/api/detections", data);
  },

  /** Update a detection's category, bbox, or species. */
  update: async (
    id: string,
    data: DetectionUpdate
  ): Promise<DetectionResponse> => {
    return api.patch<DetectionResponse>(`/api/detections/${id}`, data);
  },

  /** Delete a detection. */
  delete: async (id: string): Promise<void> => {
    return api.delete(`/api/detections/${id}`);
  },

  /** Delete all detections for a file. */
  deleteByFile: async (fileId: string): Promise<{ deleted_count: number }> => {
    return api.delete(`/api/detections/by-file/${fileId}`);
  },

  /** Verify or unverify a single detection. */
  verify: async (
    id: string,
    verified: boolean
  ): Promise<DetectionResponse> => {
    return api.patch<DetectionResponse>(`/api/detections/${id}/verify`, {
      verified,
    });
  },

  /** Bulk verify/unverify detections. */
  bulkVerify: async (
    ids: string[],
    verified: boolean
  ): Promise<{ updated_count: number }> => {
    return api.post("/api/detections/bulk-verify", {
      detection_ids: ids,
      verified,
    });
  },

  /** Bulk relabel detections. */
  bulkRelabel: async (
    ids: string[],
    species: string | null,
    category?: string
  ): Promise<{ updated_count: number }> => {
    return api.post("/api/detections/bulk-relabel", {
      detection_ids: ids,
      species,
      category,
    });
  },
};
