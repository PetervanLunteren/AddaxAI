/**
 * Detections API client
 */

import { api } from "../lib/api-client";
import type {
  DetectionResponse,
  DetectionCreate,
  DetectionCreateObservation,
  DetectionUpdate,
} from "./types";

function chunkArray<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}

export const detectionsApi = {
  /** Create a human-drawn detection. */
  create: async (data: DetectionCreate): Promise<DetectionResponse> => {
    return api.post<DetectionResponse>("/api/detections", data);
  },

  /** Create an event-level observation (no bbox). Used when the user
   * spots an animal the AI missed or only visible in a non-best video
   * frame; the data shape is Camtrap-DP observationLevel="event". */
  createObservation: async (
    data: DetectionCreateObservation
  ): Promise<DetectionResponse> => {
    return api.post<DetectionResponse>("/api/detections/observation", data);
  },

  /** Update a detection's category, bbox, or label. */
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

  /** Bulk verify/unverify detections (auto-batches in chunks of 500). */
  bulkVerify: async (
    ids: string[],
    verified: boolean
  ): Promise<{ updated_count: number }> => {
    const chunks = chunkArray(ids, 500);
    const results = await Promise.all(
      chunks.map((chunk) =>
        api.post<{ updated_count: number }>("/api/detections/bulk-verify", {
          detection_ids: chunk,
          verified,
        })
      )
    );
    return { updated_count: results.reduce((sum, r) => sum + r.updated_count, 0) };
  },

  /** Bulk relabel detections (auto-batches in chunks of 500). */
  bulkRelabel: async (
    ids: string[],
    label: string | null,
    category?: string
  ): Promise<{ updated_count: number }> => {
    const chunks = chunkArray(ids, 500);
    const results = await Promise.all(
      chunks.map((chunk) =>
        api.post<{ updated_count: number }>("/api/detections/bulk-relabel", {
          detection_ids: chunk,
          label,
          category,
        })
      )
    );
    return { updated_count: results.reduce((sum, r) => sum + r.updated_count, 0) };
  },

  /** Dismiss/undismiss a cohort of suggestions (auto-batches in chunks of 500).
   *  Hides the detections from the suggestions review without changing
   *  their label or verified state. Pass dismissed=false to undo. */
  bulkDismiss: async (
    ids: string[],
    dismissed: boolean
  ): Promise<{ updated_count: number }> => {
    const chunks = chunkArray(ids, 500);
    const results = await Promise.all(
      chunks.map((chunk) =>
        api.post<{ updated_count: number }>("/api/detections/bulk-dismiss", {
          detection_ids: chunk,
          dismissed,
        })
      )
    );
    return { updated_count: results.reduce((sum, r) => sum + r.updated_count, 0) };
  },
};
