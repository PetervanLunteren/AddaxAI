/**
 * Detections API client
 */

import { api } from "../lib/api-client";
import { mergeSnapshots } from "../lib/undo-snapshot";
import type {
  BulkResult,
  DetectionResponse,
  DetectionCreate,
  DetectionUpdate,
} from "./types";

function chunkArray<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}

/** Run a bulk call in chunks of 500 and fold the answers into one.
 *
 *  Sequential on purpose: each chunk's `before` is read when that
 *  chunk runs, and two chunks can touch boxes of the same file. In
 *  parallel the later chunk would snapshot the file after the earlier
 *  one already flipped it, and undo would leave it flipped. */
async function inChunks(
  ids: string[],
  call: (chunk: string[]) => Promise<BulkResult>,
): Promise<BulkResult> {
  const results: BulkResult[] = [];
  for (const chunk of chunkArray(ids, 500)) {
    results.push(await call(chunk));
  }
  return {
    updated_count: results.reduce((sum, r) => sum + r.updated_count, 0),
    before: mergeSnapshots(results.map((r) => r.before)),
  };
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

  /** Verify or unverify detections. `before` is the undo snapshot. */
  bulkVerify: async (ids: string[], verified: boolean): Promise<BulkResult> => {
    return inChunks(ids, (chunk) =>
      api.post<BulkResult>("/api/detections/bulk-verify", {
        detection_ids: chunk,
        verified,
      }),
    );
  },

  /** Relabel and verify detections. `before` is the undo snapshot. */
  bulkRelabel: async (
    ids: string[],
    label: string | null,
    category?: string
  ): Promise<BulkResult> => {
    return inChunks(ids, (chunk) =>
      api.post<BulkResult>("/api/detections/bulk-relabel", {
        detection_ids: chunk,
        label,
        category,
      }),
    );
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
