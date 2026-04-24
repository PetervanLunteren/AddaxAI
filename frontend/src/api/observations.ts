/**
 * Observations API client - embedding-based sort and search for the
 * Observations verify tab.
 *
 * Underlying technique is still "similarity" (cosine distance on DINOv2
 * embeddings); the user-facing naming reflects the unit of work instead.
 */

import { api } from "../lib/api-client";
import type {
  SortRequest,
  SortResponse,
  SearchRequest,
  SearchResponse,
  ObservationStatsResponse,
} from "./types";

export const observationsApi = {
  /** Sort observations by visual similarity (greedy nearest-neighbor chain). */
  sort: async (
    projectId: string,
    body: SortRequest
  ): Promise<SortResponse> => {
    return api.post<SortResponse>(
      `/api/projects/${projectId}/observations/sort`,
      body
    );
  },

  /** Find observations similar to an anchor. */
  search: async (
    projectId: string,
    body: SearchRequest
  ): Promise<SearchResponse> => {
    return api.post<SearchResponse>(
      `/api/projects/${projectId}/observations/search`,
      body
    );
  },

  /** Get embedding coverage stats for a project. */
  stats: async (projectId: string): Promise<ObservationStatsResponse> => {
    return api.get<ObservationStatsResponse>(
      `/api/projects/${projectId}/observations/stats`
    );
  },
};
