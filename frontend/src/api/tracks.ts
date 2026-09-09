/**
 * Tracks API client: the frames behind one card on the Detections tab.
 */

import { api } from "../lib/api-client";
import type { TrackDetectionsResponse } from "./types";

export const tracksApi = {
  /** The boxes of one animal, in frame order.
   *
   * `minConfidence` is the Labels page slider. It has to travel: the
   * slider digs down below the project threshold, so a card visible at
   * a lowered slider must open to the frames that were visible with it.
   * Leaving it out would open an empty track under a card on screen.
   */
  detections: async (
    trackId: string,
    minConfidence?: number | null,
    options?: { signal?: AbortSignal },
  ): Promise<TrackDetectionsResponse> => {
    const q =
      minConfidence === null || minConfidence === undefined
        ? ""
        : `?min_confidence=${minConfidence}`;
    return api.get<TrackDetectionsResponse>(
      `/api/tracks/${trackId}/detections${q}`,
      options,
    );
  },
};
