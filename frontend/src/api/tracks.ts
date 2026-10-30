/**
 * Tracks API client: the frames behind one card on the Detections tab.
 */

import { api } from "../lib/api-client";
import type { TrackDetectionsResponse } from "./types";

export const tracksApi = {
  /** The frames of one animal worth checking, in frame order: those at
   * or above the classification gate its analysis ran with, or verified.
   * The backend owns that line; no filter or slider travels with it.
   */
  detections: async (
    trackId: string,
    options?: { signal?: AbortSignal },
  ): Promise<TrackDetectionsResponse> =>
    api.get<TrackDetectionsResponse>(`/api/tracks/${trackId}/detections`, options),
};
