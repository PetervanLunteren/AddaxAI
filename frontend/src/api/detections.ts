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
};
