/**
 * Files API client
 */

import { api } from "../lib/api-client";
import type {
  AdjacentFilesResponse,
  EventFilterParams,
  FileResponse,
  FileSummary,
  FileVerificationStats,
  FileWithDetections,
} from "./types";

/** Append verify-tab filter params to a URLSearchParams instance. */
function appendFileFilterParams(
  searchParams: URLSearchParams,
  filters?: EventFilterParams,
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
  if (filters.min_confidence !== undefined)
    searchParams.set("min_confidence", filters.min_confidence.toString());
  if (filters.max_confidence !== undefined)
    searchParams.set("max_confidence", filters.max_confidence.toString());
}

export const filesApi = {
  /**
   * Get file by ID with detections
   */
  get: async (id: string, options?: { signal?: AbortSignal }): Promise<FileWithDetections> => {
    return api.get<FileWithDetections>(`/api/files/${id}`, options);
  },

  /**
   * Get observation type counts for a project
   */
  getObservationTypeStats: async (projectId: string): Promise<Record<string, number>> => {
    return api.get<Record<string, number>>(
      `/api/files/stats/observation-types?project_id=${projectId}`
    );
  },

  /**
   * Update file verification status and/or notes
   */
  update: async (
    id: string,
    data: { verified?: boolean; notes?: string; favorited?: boolean }
  ): Promise<FileResponse> => {
    return api.patch<FileResponse>(`/api/files/${id}`, data);
  },

  /** List file summaries for the Files verify tab. */
  listForVerify: async (params: {
    project_id: string;
    skip?: number;
    limit?: number;
    filters?: EventFilterParams;
  }): Promise<FileSummary[]> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", params.project_id);
    if (params.skip !== undefined)
      searchParams.set("skip", params.skip.toString());
    if (params.limit !== undefined)
      searchParams.set("limit", params.limit.toString());
    appendFileFilterParams(searchParams, params.filters);
    return api.get<FileSummary[]>(
      `/api/files/list-for-verify?${searchParams.toString()}`
    );
  },

  /** Total file count for the Files verify tab with the given filters. */
  countForVerify: async (
    projectId: string,
    filters?: EventFilterParams,
  ): Promise<{ count: number }> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFileFilterParams(searchParams, filters);
    return api.get<{ count: number }>(
      `/api/files/count-for-verify?${searchParams.toString()}`
    );
  },

  /** Aggregate verified/total file counts for the Files verify tab. */
  verificationStats: async (
    projectId: string,
    filters?: EventFilterParams,
  ): Promise<FileVerificationStats> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFileFilterParams(searchParams, filters);
    return api.get<FileVerificationStats>(
      `/api/files/verification-stats?${searchParams.toString()}`
    );
  },

  /** Adjacent file IDs in the Files verify tab's filtered list. */
  getAdjacentForVerify: async (
    fileId: string,
    projectId: string,
    filters?: EventFilterParams,
  ): Promise<AdjacentFilesResponse> => {
    const searchParams = new URLSearchParams();
    searchParams.set("project_id", projectId);
    appendFileFilterParams(searchParams, filters);
    return api.get<AdjacentFilesResponse>(
      `/api/files/${fileId}/adjacent?${searchParams.toString()}`
    );
  },
};
