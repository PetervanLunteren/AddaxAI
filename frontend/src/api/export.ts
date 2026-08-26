/**
 * Export API client.
 *
 * Each method returns a Blob so the caller can trigger a browser download.
 * We don't use the shared `api` wrapper because it assumes JSON responses.
 */

import { API_BASE_URL, ApiError } from "../lib/api-client";
import { logger } from "../lib/logger";

export type ObservationFormat = "csv" | "tsv" | "xlsx";
export type SpatialFormat = "geojson" | "shapefile" | "gpkg";

/** Optional export scope. Empty / omitted = whole project. Picking a site
 *  includes all its deployments (resolved server-side). */
export interface ExportScope {
  siteIds?: string[];
  deploymentIds?: string[];
}

/** Append the scope to an endpoint, handling URLs that already have a
 *  query string (e.g. `?format=csv`) and those that don't (spreadsheet). */
function withScope(endpoint: string, scope?: ExportScope): string {
  const parts: string[] = [];
  if (scope?.siteIds?.length) parts.push(`site_ids=${scope.siteIds.join(",")}`);
  if (scope?.deploymentIds?.length)
    parts.push(`deployment_ids=${scope.deploymentIds.join(",")}`);
  if (parts.length === 0) return endpoint;
  const sep = endpoint.includes("?") ? "&" : "?";
  return `${endpoint}${sep}${parts.join("&")}`;
}

async function fetchBlob(endpoint: string): Promise<Blob> {
  const url = `${API_BASE_URL}${endpoint}`;
  logger.info(`API GET ${endpoint}`);

  const response = await fetch(url);
  if (!response.ok) {
    let detail: unknown = null;
    let message = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const text = await response.text();
      const parsed = JSON.parse(text);
      detail = parsed.detail ?? parsed;
      if (typeof detail === "string") {
        message = detail;
      }
    } catch {
      // response body wasn't JSON
    }
    logger.error(`API GET ${endpoint} failed: ${message}`, {
      status: response.status,
      endpoint,
    });
    throw new ApiError(response.status, detail, message);
  }

  logger.info(`API GET ${endpoint} → ${response.status} OK`);
  return response.blob();
}

export const exportApi = {
  /** Location / effort table: one row per deployment (site, trap-nights). */
  downloadDeployments: (
    projectId: string,
    format: ObservationFormat,
    scope?: ExportScope,
  ): Promise<Blob> =>
    fetchBlob(
      withScope(`/api/projects/${projectId}/export/deployments?format=${format}`, scope),
    ),

  /** Media / membership table: one row per file, including empties. */
  downloadFiles: (
    projectId: string,
    format: ObservationFormat,
    scope?: ExportScope,
  ): Promise<Blob> =>
    fetchBlob(
      withScope(`/api/projects/${projectId}/export/files?format=${format}`, scope),
    ),

  /** Per-detection table (the labels grain): one row per bounding box. */
  downloadDetections: (
    projectId: string,
    format: ObservationFormat,
    scope?: ExportScope,
  ): Promise<Blob> =>
    fetchBlob(
      withScope(`/api/projects/${projectId}/export/detections?format=${format}`, scope),
    ),

  /** Event-level table (the counts grain): one row per species per event
   *  with the effective count. */
  downloadObservations: (
    projectId: string,
    format: ObservationFormat,
    scope?: ExportScope,
  ): Promise<Blob> =>
    fetchBlob(
      withScope(`/api/projects/${projectId}/export/observations?format=${format}`, scope),
    ),

  /** Overview table: one row per species (or person / vehicle / unclassified
   *  animal) with counts over the other tables. */
  downloadSummary: (
    projectId: string,
    format: ObservationFormat,
    scope?: ExportScope,
  ): Promise<Blob> =>
    fetchBlob(
      withScope(`/api/projects/${projectId}/export/summary?format=${format}`, scope),
    ),

  /** Combined workbook: Summary, Counts, Detections, Files and Deployments.
   *  XLSX only. */
  downloadSpreadsheetXlsx: (projectId: string, scope?: ExportScope): Promise<Blob> =>
    fetchBlob(withScope(`/api/projects/${projectId}/export/spreadsheet`, scope)),

  downloadSpatial: (projectId: string, format: SpatialFormat): Promise<Blob> =>
    fetchBlob(`/api/projects/${projectId}/export/spatial?format=${format}`),

  /** Kick off a CamTrap DP export job. Returns job_id; the client
   * tracks progress via the existing /ws/jobs/{job_id} WebSocket and
   * then calls `downloadCamtrapDPZip(projectId, job_id)` when the
   * WebSocket reports completion. */
  prepareCamtrapDP: async (
    projectId: string,
    includeThumbnails: boolean = false,
  ): Promise<{ job_id: string }> => {
    const res = await fetch(
      `${API_BASE_URL}/api/projects/${projectId}/export/camtrap-dp/prepare?include_thumbnails=${includeThumbnails}`,
      { method: "POST" },
    );
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || `Export prepare failed (${res.status})`);
    }
    return res.json();
  },

  /** Fetch the finished CamTrap DP ZIP for a completed export job. */
  downloadCamtrapDPZip: (projectId: string, jobId: string): Promise<Blob> =>
    fetchBlob(
      `/api/projects/${projectId}/export/camtrap-dp/download?job_id=${jobId}`,
    ),
};
