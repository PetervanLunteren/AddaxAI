/**
 * Events API client
 */

import { api } from "../lib/api-client";
import type {
  EventSummary,
  EventWithFiles,
  AdjacentEventsResponse,
  EventFilterParams,
  EventFilterOptions,
  EventVerificationStats,
  LabelTreeResponse,
} from "./types";

/** Append filter params to a URLSearchParams instance. */
function appendFilterParams(
  searchParams: URLSearchParams,
  filters?: EventFilterParams
) {
  if (!filters) return;
  if (filters.site_ids?.length)
    searchParams.set("site_ids", filters.site_ids.join(","));
  if (filters.date_from) searchParams.set("date_from", filters.date_from);
  if (filters.date_to) searchParams.set("date_to", filters.date_to);
  if (filters.labels?.length)
    searchParams.set("labels", filters.labels.join(","));
  if (filters.verification && filters.verification !== "all")
    searchParams.set("verification", filters.verification);
  if (filters.flagged && filters.flagged !== "all")
    searchParams.set("flagged", filters.flagged);
  if (filters.favorited && filters.favorited !== "all")
    searchParams.set("favorited", filters.favorited);
  if (filters.min_confidence !== undefined)
    searchParams.set("min_confidence", filters.min_confidence.toString());
  if (filters.max_confidence !== undefined)
    searchParams.set("max_confidence", filters.max_confidence.toString());
}

export const eventsApi = {
  /** Generate or regenerate events for a project. */
  generate: async (
    projectId: string
  ): Promise<{ event_count: number; message: string }> => {
    return api.post("/api/events/generate", { project_id: projectId });
  },

  /** List event summaries for a project with optional filters. */
  list: async (params: {
    project_id: string;
    skip?: number;
    limit?: number;
    filters?: EventFilterParams;
  }): Promise<EventSummary[]> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", params.project_id);
    if (params.skip !== undefined)
      searchParams.set("skip", params.skip.toString());
    if (params.limit !== undefined)
      searchParams.set("limit", params.limit.toString());
    appendFilterParams(searchParams, params.filters);
    return api.get<EventSummary[]>(`/api/events?${searchParams.toString()}`);
  },

  /** Get total event count for a project with optional filters. */
  count: async (
    projectId: string,
    filters?: EventFilterParams
  ): Promise<{ count: number }> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFilterParams(searchParams, filters);
    return api.get<{ count: number }>(
      `/api/events/count?${searchParams.toString()}`
    );
  },

  /** Get event with all files and detections. */
  get: async (eventId: string): Promise<EventWithFiles> => {
    return api.get<EventWithFiles>(`/api/events/${eventId}`);
  },

  /** Get adjacent event IDs for navigation with optional filters. */
  getAdjacent: async (
    eventId: string,
    projectId: string,
    filters?: EventFilterParams
  ): Promise<AdjacentEventsResponse> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFilterParams(searchParams, filters);
    return api.get<AdjacentEventsResponse>(
      `/api/events/${eventId}/adjacent?${searchParams.toString()}`
    );
  },

  /** Get aggregate verification stats for filtered events. */
  verificationStats: async (
    projectId: string,
    filters?: EventFilterParams
  ): Promise<EventVerificationStats> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFilterParams(searchParams, filters);
    return api.get<EventVerificationStats>(
      `/api/events/verification-stats?${searchParams.toString()}`
    );
  },

  /** Get available filter options for a project. */
  getFilterOptions: async (
    projectId: string
  ): Promise<EventFilterOptions> => {
    return api.get<EventFilterOptions>(
      `/api/events/filter-options?project_id=${projectId}`
    );
  },

  /** Get the label filter tree (pre-built from label_taxonomy table). */
  getLabelTree: async (
    projectId: string,
    countBy?: string
  ): Promise<LabelTreeResponse | null> => {
    const params = `project_id=${projectId}${countBy ? `&count_by=${countBy}` : ""}`;
    return api.get<LabelTreeResponse | null>(
      `/api/events/label-tree?${params}`
    );
  },
};
