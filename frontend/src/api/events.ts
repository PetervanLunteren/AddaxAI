/**
 * Events API client
 */

import { api } from "../lib/api-client";
import type {
  EventSummary,
  EventWithFiles,
  AdjacentEventsResponse,
} from "./types";

export const eventsApi = {
  /** Generate or regenerate events for a project. */
  generate: async (
    projectId: string
  ): Promise<{ event_count: number; message: string }> => {
    return api.post("/api/events/generate", { project_id: projectId });
  },

  /** List event summaries for a project. */
  list: async (params: {
    project_id: string;
    skip?: number;
    limit?: number;
  }): Promise<EventSummary[]> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", params.project_id);
    if (params.skip !== undefined)
      searchParams.set("skip", params.skip.toString());
    if (params.limit !== undefined)
      searchParams.set("limit", params.limit.toString());
    return api.get<EventSummary[]>(`/api/events?${searchParams.toString()}`);
  },

  /** Get total event count for a project. */
  count: async (projectId: string): Promise<{ count: number }> => {
    return api.get<{ count: number }>(
      `/api/events/count?project_id=${projectId}`
    );
  },

  /** Get event with all files and detections. */
  get: async (eventId: string): Promise<EventWithFiles> => {
    return api.get<EventWithFiles>(`/api/events/${eventId}`);
  },

  /** Get adjacent event IDs for navigation. */
  getAdjacent: async (
    eventId: string,
    projectId: string
  ): Promise<AdjacentEventsResponse> => {
    return api.get<AdjacentEventsResponse>(
      `/api/events/${eventId}/adjacent?project_id=${projectId}`
    );
  },
};
