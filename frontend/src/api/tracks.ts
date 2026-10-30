/**
 * Tracks API client: the frames behind one card on the Detections tab.
 */

import { api } from "../lib/api-client";
import type { TrackDetectionsResponse } from "./types";

export const tracksApi = {
  /** Every frame of one animal, in frame order, weak ones included.
   * No filter or slider travels with it: nothing is hidden inside a
   * track.
   */
  detections: async (
    trackId: string,
    options?: { signal?: AbortSignal },
  ): Promise<TrackDetectionsResponse> =>
    api.get<TrackDetectionsResponse>(`/api/tracks/${trackId}/detections`, options),
};
