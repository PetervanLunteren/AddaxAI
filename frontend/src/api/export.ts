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
  downloadObservations: (projectId: string, format: ObservationFormat): Promise<Blob> =>
    fetchBlob(`/api/projects/${projectId}/export/observations?format=${format}`),

  downloadSpatial: (projectId: string, format: SpatialFormat): Promise<Blob> =>
    fetchBlob(`/api/projects/${projectId}/export/spatial?format=${format}`),

  downloadCamtrapDP: (projectId: string): Promise<Blob> =>
    fetchBlob(`/api/projects/${projectId}/export/camtrap-dp`),
};
