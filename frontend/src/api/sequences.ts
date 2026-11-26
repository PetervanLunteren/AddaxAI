/**
 * Sequences API client
 */

import { api } from "../lib/api-client";

/** What the label counts count: the Counts page's sequences, the Files
 *  tab's files, or the Labels grid's boxes. */
export type LabelCountUnit = "sequence" | "file" | "detection";

/** The filters a page is applying, for the label tree's counts. The
 *  same vocabulary as the list endpoints. */
export interface LabelTreeScope {
  siteIds?: string[];
  dateFrom?: string;
  dateTo?: string;
  verification?: string;
  flagged?: string;
  favorited?: string;
  empty?: string;
  minConfidence?: number;
  maxConfidence?: number;
  minLabelConfidence?: number;
  maxLabelConfidence?: number;
}
import type {
  SequenceSummary,
  SequenceWithFiles,
  HumanObservationRow,
  ObservationAttributesPatch,
  AdjacentSequencesResponse,
  SequenceFilterParams,
  SequenceFilterOptions,
  SequenceCounts,
  SequenceVerificationStats,
  LabelCounts,
} from "./types";

/** Append filter params to a URLSearchParams instance. */
function appendFilterParams(
  searchParams: URLSearchParams,
  filters?: SequenceFilterParams
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

export const sequencesApi = {
  /** List sequence summaries for a project with optional filters. */
  list: async (params: {
    project_id: string;
    skip?: number;
    limit?: number;
    filters?: SequenceFilterParams;
  }): Promise<SequenceSummary[]> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", params.project_id);
    if (params.skip !== undefined)
      searchParams.set("skip", params.skip.toString());
    if (params.limit !== undefined)
      searchParams.set("limit", params.limit.toString());
    appendFilterParams(searchParams, params.filters);
    return api.get<SequenceSummary[]>(`/api/sequences?${searchParams.toString()}`);
  },

  /** How many sequences pass the filters, and how many are confirmed. */
  count: async (
    projectId: string,
    filters?: SequenceFilterParams
  ): Promise<SequenceCounts> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFilterParams(searchParams, filters);
    return api.get<SequenceCounts>(
      `/api/sequences/count?${searchParams.toString()}`
    );
  },

  /** Get sequence with all files and detections. */
  get: async (sequenceId: string): Promise<SequenceWithFiles> => {
    return api.get<SequenceWithFiles>(`/api/sequences/${sequenceId}`);
  },

  /** Set the human confirmation of the sequence's species and counts. */
  setConfirmed: async (
    sequenceId: string,
    confirmed: boolean,
  ): Promise<SequenceWithFiles> => {
    return api.patch<SequenceWithFiles>(`/api/sequences/${sequenceId}/confirm`, {
      confirmed,
    });
  },

  /** Set (or clear, count=null) the human count for one species. */
  setObservationCount: async (
    sequenceId: string,
    observationId: string,
    count: number | null,
  ): Promise<SequenceWithFiles> => {
    return api.patch<SequenceWithFiles>(
      `/api/sequences/${sequenceId}/observations/${observationId}`,
      { count },
    );
  },

  /** Record a species the AI missed (or bump an existing one). */
  addObservation: async (
    sequenceId: string,
    data: {
      /** Only for a detector class; a species gets its class on the server. */
      category?: string;
      count: number;
      label?: string | null;
      label_taxonomy_id?: string | null;
    },
  ): Promise<SequenceWithFiles> => {
    return api.post<SequenceWithFiles>(
      `/api/sequences/${sequenceId}/observations`,
      data,
    );
  },

  /** Change the species of one count row; its count moves to the target
   *  (summing into the target species when it already has a row). */
  relabelObservation: async (
    sequenceId: string,
    observationId: string,
    data: {
      category?: string;
      label?: string | null;
      label_taxonomy_id?: string | null;
    },
  ): Promise<SequenceWithFiles> => {
    return api.patch<SequenceWithFiles>(
      `/api/sequences/${sequenceId}/observations/${observationId}/relabel`,
      data,
    );
  },

  /** Remove the human contribution to one species. */
  deleteObservation: async (
    sequenceId: string,
    observationId: string,
  ): Promise<SequenceWithFiles> => {
    return api.delete<SequenceWithFiles>(
      `/api/sequences/${sequenceId}/observations/${observationId}`,
    );
  },

  /** Set sex / life stage / behaviour on one row. Only the keys in the
   *  patch change; null clears a field. */
  setObservationAttributes: async (
    sequenceId: string,
    observationId: string,
    patch: ObservationAttributesPatch,
  ): Promise<SequenceWithFiles> => {
    return api.patch<SequenceWithFiles>(
      `/api/sequences/${sequenceId}/observations/${observationId}/attributes`,
      patch,
    );
  },

  /** Split one row into two of the same species (source minus one, new
   *  row at one) so each can get its own demographics. */
  splitObservation: async (
    sequenceId: string,
    observationId: string,
  ): Promise<SequenceWithFiles> => {
    return api.post<SequenceWithFiles>(
      `/api/sequences/${sequenceId}/observations/${observationId}/split`,
    );
  },

  /** Replace the sequence's visible cohorts with the given rows ("Same as
   *  last"). */
  setHumanObservations: async (
    sequenceId: string,
    observations: HumanObservationRow[],
  ): Promise<SequenceWithFiles> => {
    return api.put<SequenceWithFiles>(`/api/sequences/${sequenceId}/observations`, {
      observations,
    });
  },

  /** Set the sequence's free text. Never touches the confirmation. */
  setNotes: async (
    sequenceId: string,
    notes: string | null,
  ): Promise<SequenceWithFiles> => {
    return api.patch<SequenceWithFiles>(`/api/sequences/${sequenceId}/notes`, {
      notes,
    });
  },

  /** Drop every human count edit on the sequence, back to the AI proposal. */
  resetCounts: async (sequenceId: string): Promise<SequenceWithFiles> => {
    return api.post<SequenceWithFiles>(
      `/api/sequences/${sequenceId}/observations/reset`,
    );
  },

  /** Get adjacent sequence IDs for navigation with optional filters. */
  getAdjacent: async (
    sequenceId: string,
    projectId: string,
    filters?: SequenceFilterParams
  ): Promise<AdjacentSequencesResponse> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFilterParams(searchParams, filters);
    return api.get<AdjacentSequencesResponse>(
      `/api/sequences/${sequenceId}/adjacent?${searchParams.toString()}`
    );
  },

  /** The confirmed sequences and verified cards a reprocess must keep. */
  verificationStats: async (
    projectId: string,
    filters?: SequenceFilterParams
  ): Promise<SequenceVerificationStats> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFilterParams(searchParams, filters);
    return api.get<SequenceVerificationStats>(
      `/api/sequences/verification-stats?${searchParams.toString()}`
    );
  },

  /** Get available filter options for a project. */
  getFilterOptions: async (
    projectId: string
  ): Promise<SequenceFilterOptions> => {
    return api.get<SequenceFilterOptions>(
      `/api/sequences/filter-options?project_id=${projectId}`
    );
  },

  /** How many of each label the page would show with that label ticked
   *  alone, under its other filters given as `scope`. The tree they go
   *  on comes with `getFilterOptions`. An "all" value is the same as
   *  leaving the filter out. */
  getLabelCounts: async (
    projectId: string,
    countBy: LabelCountUnit,
    scope?: LabelTreeScope
  ): Promise<LabelCounts> => {
    const search = new URLSearchParams({ project_id: projectId });
    search.set("count_by", countBy);
    if (scope?.siteIds?.length) search.set("site_ids", scope.siteIds.join(","));
    if (scope?.dateFrom) search.set("date_from", scope.dateFrom);
    if (scope?.dateTo) search.set("date_to", scope.dateTo);
    const choice = (key: string, value: string | undefined) => {
      if (value && value !== "all") search.set(key, value);
    };
    choice("verification", scope?.verification);
    choice("flagged", scope?.flagged);
    choice("favorited", scope?.favorited);
    choice("empty", scope?.empty);
    const range = (key: string, value: number | undefined) => {
      if (value !== undefined) search.set(key, String(value));
    };
    range("min_confidence", scope?.minConfidence);
    range("max_confidence", scope?.maxConfidence);
    range("min_label_confidence", scope?.minLabelConfidence);
    range("max_label_confidence", scope?.maxLabelConfidence);
    return api.get<LabelCounts>(
      `/api/sequences/label-counts?${search.toString()}`
    );
  },
};
