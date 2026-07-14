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
  if (filters.empty && filters.empty !== "all")
    searchParams.set("empty", filters.empty);
  if (filters.min_confidence !== undefined)
    searchParams.set("min_confidence", filters.min_confidence.toString());
  if (filters.max_confidence !== undefined)
    searchParams.set("max_confidence", filters.max_confidence.toString());
  if (filters.min_label_confidence !== undefined)
    searchParams.set(
      "min_label_confidence",
      filters.min_label_confidence.toString(),
    );
  if (filters.max_label_confidence !== undefined)
    searchParams.set(
      "max_label_confidence",
      filters.max_label_confidence.toString(),
    );
  if (filters.sort && filters.sort !== "newest")
    searchParams.set("sort", filters.sort);
  if (filters.seed !== undefined)
    searchParams.set("seed", filters.seed.toString());
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

  /** Set the human confirmation of the event's species and counts. */
  setConfirmed: async (
    eventId: string,
    confirmed: boolean,
  ): Promise<EventWithFiles> => {
    return api.patch<EventWithFiles>(`/api/events/${eventId}/confirm`, {
      confirmed,
    });
  },

  /** Set (or clear, count=null) the human count for one species. */
  setObservationCount: async (
    eventId: string,
    observationId: string,
    count: number | null,
  ): Promise<EventWithFiles> => {
    return api.patch<EventWithFiles>(
      `/api/events/${eventId}/observations/${observationId}`,
      { count },
    );
  },

  /** Record a species the AI missed (or bump an existing one). */
  addObservation: async (
    eventId: string,
    data: {
      category: string;
      count: number;
      label?: string | null;
      label_taxonomy_id?: string | null;
    },
  ): Promise<EventWithFiles> => {
    return api.post<EventWithFiles>(
      `/api/events/${eventId}/observations`,
      data,
    );
  },

  /** Change the species of one count row; its count moves to the target
   *  (summing into the target species when it already has a row). */
  relabelObservation: async (
    eventId: string,
    observationId: string,
    data: {
      category: string;
      label?: string | null;
      label_taxonomy_id?: string | null;
    },
  ): Promise<EventWithFiles> => {
    return api.patch<EventWithFiles>(
      `/api/events/${eventId}/observations/${observationId}/relabel`,
      data,
    );
  },

  /** Remove the human contribution to one species. */
  deleteObservation: async (
    eventId: string,
    observationId: string,
  ): Promise<EventWithFiles> => {
    return api.delete<EventWithFiles>(
      `/api/events/${eventId}/observations/${observationId}`,
    );
  },

  /** Drop every human count edit on the event, back to the AI proposal. */
  resetCounts: async (eventId: string): Promise<EventWithFiles> => {
    return api.post<EventWithFiles>(
      `/api/events/${eventId}/observations/reset`,
    );
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

  /** Get the label filter tree (pre-built from label_taxonomy table).
   *  Counts are scoped to the optional site + date filters. */
  getLabelTree: async (
    projectId: string,
    countBy?: string,
    scope?: { siteIds?: string[]; dateFrom?: string; dateTo?: string }
  ): Promise<LabelTreeResponse | null> => {
    const search = new URLSearchParams({ project_id: projectId });
    if (countBy) search.set("count_by", countBy);
    if (scope?.siteIds?.length) search.set("site_ids", scope.siteIds.join(","));
    if (scope?.dateFrom) search.set("date_from", scope.dateFrom);
    if (scope?.dateTo) search.set("date_to", scope.dateTo);
    return api.get<LabelTreeResponse | null>(
      `/api/events/label-tree?${search.toString()}`
    );
  },
};
