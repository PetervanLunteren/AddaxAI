/**
 * Export API client. Every export writes into a folder on the backend and
 * answers with what it wrote (or starts a job that does); nothing here is
 * a browser download.
 */

import { api } from "../lib/api-client";
import type { OutputPreview } from "./folder-runs";
import type { PackageSizeResponse } from "./packages";
import type { MediaOptionsRequest } from "../pages/folder-run/save/media-options";

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

/** Body of the image copies preview, and the media half of a folder-run
 *  save: one mapper on the frontend (`mediaRequestFields`), one schema on
 *  the backend (`MediaExportOptions`). A folder run previews through the
 *  same endpoint; it just never sends `deployment_ids`. */
export type ImageCopiesOptions = MediaOptionsRequest;

export interface ImageCopiesRequest extends ImageCopiesOptions {
  output_dir: string;
}

/** Optional export scope. Omitted = whole project. */
export interface ExportScope {
  deploymentIds?: string[];
}

/** Append the scope to a query-string endpoint. */
function withScope(endpoint: string, scope?: ExportScope): string {
  if (!scope?.deploymentIds?.length) return endpoint;
  const sep = endpoint.includes("?") ? "&" : "?";
  return `${endpoint}${sep}deployment_ids=${scope.deploymentIds.join(",")}`;
}

/** Where an export goes and which deployments it covers: the shared
 *  part of every export request (`ExportTarget` on the backend). */
export interface ExportTarget {
  output_dir: string;
  deployment_ids?: string[];
}

/** What a file export wrote: the folder and the file names in it. */
export interface WrittenFiles {
  output_dir: string;
  files: string[];
}

export const exportApi = {
  /** Summary, Counts, Detections, Files and Deployments: one XLSX
   *  workbook, or five CSV / TSV files. */
  writeTables: (
    projectId: string,
    body: ExportTarget & { format: ObservationFormat },
  ): Promise<WrittenFiles> =>
    api.post<WrittenFiles>(`/api/projects/${projectId}/export/tables`, body),

  writeSpatial: (
    projectId: string,
    body: ExportTarget & { format: SpatialFormat },
  ): Promise<WrittenFiles> =>
    api.post<WrittenFiles>(`/api/projects/${projectId}/export/spatial`, body),

  /** Starts the job that writes the Camtrap DP ZIP into the folder.
   *  The job's completion payload is a WrittenFiles. */
  prepareCamtrapDP: (
    projectId: string,
    body: ExportTarget & { include_thumbnails: boolean },
  ): Promise<{ job_id: string }> =>
    api.post<{ job_id: string }>(`/api/projects/${projectId}/export/camtrap-dp/prepare`, body),

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

  /** What the media copies would write (counts, bytes, folder tree),
   *  for the tree preview beside the options. The one preview endpoint:
   *  the folder-run Save step calls it with its run id as the project. */
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

  /** How much a project package of these deployments holds, for the
   *  size line in the export dialog. */
  packageSize: (projectId: string, scope?: ExportScope): Promise<PackageSizeResponse> =>
    api.get<PackageSizeResponse>(
      withScope(`/api/projects/${projectId}/export/package/size`, scope),
    ),

  /** Not a download: starts the job that writes a project package
   *  folder into `output_dir`. The job's completion payload is a
   *  PackageExportResult. */
  preparePackage: (
    projectId: string,
    body: { output_dir: string; deployment_ids?: string[]; include_media: boolean },
  ): Promise<{ job_id: string }> =>
    api.post<{ job_id: string }>(
      `/api/projects/${projectId}/export/package/prepare`,
      body,
    ),
};
