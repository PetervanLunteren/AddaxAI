/**
 * Detections API client
 */

import { api } from "../lib/api-client";
import type {
  DetectionResponse,
  DetectionCreate,
  DetectionUpdate,
} from "./types";

/** The reverted state of one detection returned by
 *  bulk-revert-to-original — enough to patch a grid crop in place. */
export interface RevertedDetection {
  detection_id: string;
  label: string | null;
  category: string;
  label_confidence: number | null;
  label_taxonomy_id: string | null;
  scientific_name: string | null;
  common_name: string | null;
  verified: boolean;
}

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

  /** Bulk verify/unverify detections (auto-batches in chunks of 500).
   *  `expandTracks` false keeps the verdict on the ids given instead of
   *  spreading it over their whole track: what an opened track needs,
   *  where the person is judging frames rather than the animal. */
  bulkVerify: async (
    ids: string[],
    verified: boolean,
    expandTracks = true
  ): Promise<{ updated_count: number }> => {
    const chunks = chunkArray(ids, 500);
    const results = await Promise.all(
      chunks.map((chunk) =>
        api.post<{ updated_count: number }>("/api/detections/bulk-verify", {
          detection_ids: chunk,
          verified,
          expand_tracks: expandTracks,
        })
      )
    );
    return { updated_count: results.reduce((sum, r) => sum + r.updated_count, 0) };
  },

  /** Bulk relabel detections (auto-batches in chunks of 500). */
  bulkRelabel: async (
    ids: string[],
    label: string | null,
    category?: string,
    expandTracks = true
  ): Promise<{ updated_count: number }> => {
    const chunks = chunkArray(ids, 500);
    const results = await Promise.all(
      chunks.map((chunk) =>
        api.post<{ updated_count: number }>("/api/detections/bulk-relabel", {
          detection_ids: chunk,
          label,
          category,
          expand_tracks: expandTracks,
        })
      )
    );
    return { updated_count: results.reduce((sum, r) => sum + r.updated_count, 0) };
  },

  /** Revert detections to the model's original prediction (undo of a
   *  human relabel / verify). Auto-batches in chunks of 500. Returns the
   *  reverted rows so the caller can patch its grid in place. */
  bulkRevertToOriginal: async (
    ids: string[],
    expandTracks = true,
  ): Promise<{ reverted: RevertedDetection[] }> => {
    const chunks = chunkArray(ids, 500);
    const results = await Promise.all(
      chunks.map((chunk) =>
        api.post<{ reverted: RevertedDetection[] }>(
          "/api/detections/bulk-revert-to-original",
          { detection_ids: chunk, expand_tracks: expandTracks },
        ),
      ),
    );
    return { reverted: results.flatMap((r) => r.reverted) };
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
