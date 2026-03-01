/**
 * Similarity API client - sort and search endpoints.
 */

import { api } from "../lib/api-client";
import type {
  SortRequest,
  SortResponse,
  SearchRequest,
  SearchResponse,
  SimilarityStatsResponse,
} from "./types";

export const similarityApi = {
  /** Sort detections by visual similarity (greedy nearest-neighbor chain). */
  sort: async (
    projectId: string,
    body: SortRequest
  ): Promise<SortResponse> => {
    return api.post<SortResponse>(
      `/api/projects/${projectId}/similarity/sort`,
      body
    );
  },

  /** Find detections similar to an anchor. */
  search: async (
    projectId: string,
    body: SearchRequest
  ): Promise<SearchResponse> => {
    return api.post<SearchResponse>(
      `/api/projects/${projectId}/similarity/search`,
      body
    );
  },

  /** Get embedding coverage stats. */
  stats: async (projectId: string): Promise<SimilarityStatsResponse> => {
    return api.get<SimilarityStatsResponse>(
      `/api/projects/${projectId}/similarity/stats`
    );
  },
};
