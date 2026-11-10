/**
 * Files API client
 */

import { queryOptions } from "@tanstack/react-query";

import { api } from "../lib/api-client";
import type { BulkResult, FileResponse, FileScope, FileWithDetections } from "./types";

export const filesApi = {
  /**
   * Get file by ID with its tracks, the boxes of the scope and its MaxN
   * per species. "all" is every box of every frame; "cards" is a
   * photo's boxes, or a clip's cover boxes plus one box per track.
   */
  get: async (
    id: string,
    options?: { signal?: AbortSignal; scope?: FileScope },
  ): Promise<FileWithDetections> => {
    const query = options?.scope === "cards" ? "?scope=cards" : "";
    return api.get<FileWithDetections>(`/api/files/${id}${query}`, { signal: options?.signal });
  },

  /**
   * Update file verification status and/or notes
   */
  update: async (
    id: string,
    data: {
      verified?: boolean;
      notes?: string;
      favorited?: boolean;
      flagged?: boolean;
    }
  ): Promise<FileResponse> => {
    return api.patch<FileResponse>(`/api/files/${id}`, data);
  },

  /**
   * Verify or unverify up to 500 files in one request. Same rule per
   * file as `update(id, { verified })`. `before` is the undo snapshot.
   */
  bulkVerify: async (ids: string[], verified = true): Promise<BulkResult> => {
    return api.post<BulkResult>("/api/files/bulk-verify", {
      file_ids: ids,
      verified,
    });
  },
};

/**
 * The query for a file in one scope, so every reader spells the key the
 * same way. The two scopes are two cache entries: a tile or a bulk
 * action holds the cards, a viewer the full file; both sit under the
 * `["file", id]` prefix, which the invalidations target. Until
 * 2026-11-02 tiles and viewers shared the full file, 28 MB per BRUV
 * clip, seven of them on one page.
 */
export function fileQueryOptions(id: string, scope: FileScope = "all") {
  return queryOptions({
    queryKey: scope === "cards" ? ["file", id, "cards"] : ["file", id],
    queryFn: ({ signal }) => filesApi.get(id, { signal, scope }),
  });
}
