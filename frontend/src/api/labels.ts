/**
 * Labels API client - embedding-based sort and search for the
 * Labels verify tab.
 *
 * Underlying technique is still "similarity" (cosine distance on DINOv2
 * embeddings); the user-facing naming reflects the unit of work instead.
 *
 * Sort and search return an NDJSON event stream from the backend. Use
 * `sortStream` / `searchStream` if you want progress events; the plain
 * `sort` / `search` helpers drain the stream and return only the final
 * result.
 */

import { api } from "../lib/api-client";
import type {
  CohortsResponse,
  LabelStatsResponse,
  SearchRequest,
  SearchResponse,
  SortRequest,
  SortResponse,
} from "./types";

export type LabelsProgressPhase = "load" | "sort" | "neighbors";

export interface LabelsProgressEvent {
  type: "progress";
  phase: LabelsProgressPhase;
  done: number;
  total: number;
}

interface ResultEvent<T> {
  type: "result";
  // The result payload is spread alongside `type`, matching the backend
  // event shape: `{"type":"result", ...SortResponse}`.
  [key: string]: unknown extends T ? unknown : T[keyof T] | "result";
}

interface ErrorEvent {
  type: "error";
  message: string;
}

type StreamEvent<T> =
  | LabelsProgressEvent
  | ResultEvent<T>
  | ErrorEvent;

/**
 * Stream NDJSON from a backend endpoint, calling `onProgress` for each
 * progress event and resolving with the final result payload.
 *
 * Pass `body=undefined` to make the request a GET (used by the cohorts
 * endpoint which has no body). Otherwise the body is JSON-encoded and
 * the request method is POST.
 *
 * Uses fetch + getReader rather than the shared `api` helper because
 * `api.post` parses the whole response body as JSON; here we need
 * line-by-line parsing.
 */
async function streamNdjson<T>(
  url: string,
  body: unknown,
  onProgress: (e: LabelsProgressEvent) => void,
  signal?: AbortSignal,
): Promise<T> {
  const isPost = body !== undefined;
  const init: RequestInit = {
    method: isPost ? "POST" : "GET",
    signal,
  };
  if (isPost) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const response = await fetch(url, init);

  if (!response.ok) {
    // Drain the body so the connection is reusable.
    const text = await response.text().catch(() => "");
    throw new Error(text || `HTTP ${response.status}`);
  }
  if (!response.body) {
    throw new Error("Response has no body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: T | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Newline-delimited JSON. Keep the trailing partial line in the
    // buffer until the next read fills it.
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const event = JSON.parse(trimmed) as StreamEvent<T>;
      if (event.type === "progress") {
        onProgress(event);
      } else if (event.type === "error") {
        throw new Error(event.message);
      } else if (event.type === "result") {
        const { type: _t, ...rest } = event as { type: "result" } & T;
        result = rest as unknown as T;
      }
    }
  }

  if (result === null) {
    throw new Error("No result event received");
  }
  return result;
}

export const labelsApi = {
  /**
   * Sort labels by visual similarity. Drains the NDJSON stream
   * and returns only the final result. Use `sortStream` if you also
   * need progress events.
   */
  sort: async (
    projectId: string,
    body: SortRequest,
  ): Promise<SortResponse> => {
    return labelsApi.sortStream(projectId, body, () => {});
  },

  sortStream: async (
    projectId: string,
    body: SortRequest,
    onProgress: (e: LabelsProgressEvent) => void,
    signal?: AbortSignal,
  ): Promise<SortResponse> => {
    return streamNdjson<SortResponse>(
      `/api/projects/${projectId}/labels/sort`,
      body,
      onProgress,
      signal,
    );
  },

  /** Find labels similar to an anchor. */
  search: async (
    projectId: string,
    body: SearchRequest,
  ): Promise<SearchResponse> => {
    return labelsApi.searchStream(projectId, body, () => {});
  },

  searchStream: async (
    projectId: string,
    body: SearchRequest,
    onProgress: (e: LabelsProgressEvent) => void,
    signal?: AbortSignal,
  ): Promise<SearchResponse> => {
    return streamNdjson<SearchResponse>(
      `/api/projects/${projectId}/labels/search`,
      body,
      onProgress,
      signal,
    );
  },

  /** Get embedding coverage stats for a project. */
  stats: async (projectId: string): Promise<LabelStatsResponse> => {
    return api.get<LabelStatsResponse>(
      `/api/projects/${projectId}/labels/stats`,
    );
  },

  /**
   * Cohort grouping for the promotion review panel. Drains the NDJSON
   * stream and returns only the final result. Use `cohortsStream` to
   * surface progress.
   */
  cohorts: async (projectId: string): Promise<CohortsResponse> => {
    return labelsApi.cohortsStream(projectId, () => {});
  },

  cohortsStream: async (
    projectId: string,
    onProgress: (e: LabelsProgressEvent) => void,
    signal?: AbortSignal,
  ): Promise<CohortsResponse> => {
    // GET request — no body. The endpoint reads `min_count` and
    // `max_cohorts` from query params; defaults (5 / 20) are baked into
    // the panel's expected workload, so callers don't override them.
    return streamNdjson<CohortsResponse>(
      `/api/projects/${projectId}/labels/cohorts`,
      undefined,
      onProgress,
      signal,
    );
  },
};
