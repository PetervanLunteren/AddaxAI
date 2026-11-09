/**
 * Undo API client: one call that puts a before-state back.
 */

import { api } from "../lib/api-client";
import type { DetectionResponse, UndoSnapshot } from "./types";

export const undoApi = {
  /** Restore the rows in the snapshot to the values it holds. Returns
   *  the boxes it wrote, so a grid can patch them in place. Rows that
   *  no longer exist are skipped, so the list can be shorter. */
  restore: async (
    snapshot: UndoSnapshot,
  ): Promise<{ detections: DetectionResponse[] }> => {
    return api.post<{ detections: DetectionResponse[] }>("/api/undo", snapshot);
  },
};
