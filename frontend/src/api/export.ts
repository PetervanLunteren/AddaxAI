/**
 * Export API client.
 *
 * Each method returns a Blob so the caller can trigger a browser download.
 * We don't use the shared `api` wrapper because it assumes JSON responses.
 */

import { API_BASE_URL, ApiError, api } from "../lib/api-client";
import { logger } from "../lib/logger";
import type { OutputPreview, SeparateGroupBy } from "./folder-runs";

export type ObservationFormat = "csv" | "tsv" | "xlsx";

/** What the recognition-file export did, per deployment in scope. */
export interface RecognitionJsonExportResult {
  written: {
    deployment_id: string;
    folder_path: string;
    output_path: string;
    image_count: number;
  }[];
  skipped: {
    deployment_id: string;
    folder_path: string | null;
    reason: string;
  }[];
}
export type SpatialFormat = "geojson" | "shapefile" | "gpkg";

/** Body of the image copies preview: the folder-run media options plus
 *  which files. Mirrors `MediaExportOptions` on the backend. */
export interface ImageCopiesOptions {
  media_threshold: number;
  separate_group_by: SeparateGroupBy;
  group_events: boolean;
  separate_species_last: boolean;
  include_empty: boolean;
  /** Labels to leave out (LabelTaxonomy ids or raw label strings). */
  excluded_label_ids: string[];
  draw_bboxes: boolean;
  anonymise: boolean;
  name_mode: "common" | "scientific";
  /** Omitted = whole project. Never send an empty list: the backend
   *  refuses it, and the row is disabled in that state anyway. */
  deployment_ids?: string[];
  favorited_only: boolean;
  verified_only: boolean;
  /** ISO dates (YYYY-MM-DD), inclusive on whole days. */
  date_from?: string;
  date_to?: string;
}

export interface ImageCopiesRequest extends ImageCopiesOptions {
  output_dir: string;
}

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

  /** Not a download: starts a job that writes addaxai-recognitions.json
   *  into each deployment folder in scope. The job's completion payload
   *  is a RecognitionJsonExportResult. */
  prepareRecognitionJson: (
    projectId: string,
    scope?: ExportScope,
  ): Promise<{ job_id: string }> =>
    api.post<{ job_id: string }>(
      withScope(`/api/projects/${projectId}/export/recognition-json/prepare`, scope),
    ),

  /** What the image copies export would write (counts, bytes, folder
   *  tree), for the dialog's summary line. Same shape as the folder-run
   *  Save step's preview. */
  previewImageCopies: (
    projectId: string,
    body: ImageCopiesOptions,
  ): Promise<OutputPreview> =>
    api.post<OutputPreview>(`/api/projects/${projectId}/export/media/preview`, body),

  /** Not a download: starts the save-outputs job that copies the images
   *  into `output_dir/addaxai-media`. The job's completion payload is a
   *  SaveOutputsResult. */
  prepareImageCopies: (
    projectId: string,
    body: ImageCopiesRequest,
  ): Promise<{ job_id: string }> =>
    api.post<{ job_id: string }>(`/api/projects/${projectId}/export/media/prepare`, body),
};
