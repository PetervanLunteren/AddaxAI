/**
 * Models API client.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import type { QueryClient } from "@tanstack/react-query";

import { api, API_BASE_URL } from "../lib/api-client";
import type {
  CustomModelCreateRequest,
  CustomModelInfo,
  CustomModelInspectResponse,
  CustomModelUpdateRequest,
  CustomModelsResponse,
  CustomModelUploadSession,
  CustomModelUploadedFile,
  ModelInfo,
  ModelStatusResponse,
  TaxonomyResponse,
  GeofenceResponse,
} from "./types";

/**
 * Invalidate every query derived from a model's on-disk files. Call after
 * a model prepare (download / env build) completes: the geofence and
 * taxonomy queries may have been fetched (404) while the model dir was
 * still missing, and would otherwise stay cached as absent for the whole
 * session, hiding the country selector and the species tree.
 */
export function invalidateModelMetadata(
  queryClient: QueryClient,
  modelId: string | null | undefined,
) {
  if (!modelId) return;
  for (const key of ["model-status", "model-geofence", "taxonomy"]) {
    void queryClient.invalidateQueries({ queryKey: [key, modelId] });
  }
}

export function uploadCustomModelFile(
  uploadId: string,
  file: File,
  onProgress: (loaded: number, total: number) => void,
  signal: AbortSignal,
): Promise<CustomModelUploadedFile> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Upload canceled", "AbortError"));
      return;
    }
    const xhr = new XMLHttpRequest();
    const onAbort = () => xhr.abort();
    signal.addEventListener("abort", onAbort, { once: true });
    xhr.open(
      "PUT",
      `${API_BASE_URL}/api/ml/custom-models/uploads/${encodeURIComponent(uploadId)}/files/${encodeURIComponent(file.name)}`,
    );
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (event) => {
      onProgress(event.loaded, event.lengthComputable ? event.total : file.size);
    };
    xhr.onload = () => {
      signal.removeEventListener("abort", onAbort);
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as CustomModelUploadedFile);
        } catch {
          reject(new Error("The upload completed but the server response could not be read."));
        }
        return;
      }
      let message = `Upload failed (HTTP ${xhr.status})`;
      try {
        const body = JSON.parse(xhr.responseText) as { detail?: unknown };
        if (typeof body.detail === "string") message = body.detail;
      } catch { /* retain the status message */ }
      reject(new Error(message));
    };
    xhr.onerror = () => {
      signal.removeEventListener("abort", onAbort);
      reject(new Error("The model file upload failed. Check the connection and try again."));
    };
    xhr.onabort = () => {
      signal.removeEventListener("abort", onAbort);
      reject(new DOMException("Upload canceled", "AbortError"));
    };
    xhr.send(file);
  });
}

export const modelsApi = {
  /**
   * List all detection models
   */
  listDetectionModels: () => api.get<ModelInfo[]>("/api/ml/models/detection"),

  /**
   * List all classification models (includes "None" option)
   */
  listClassificationModels: () => api.get<ModelInfo[]>("/api/ml/models/classification"),

  /**
   * List all embedding models (includes "No embeddings" option)
   */
  listEmbeddingModels: () => api.get<ModelInfo[]>("/api/ml/models/embedding"),

  listCustomModels: () => api.get<CustomModelsResponse>("/api/ml/custom-models"),

  startCustomModelUpload: () =>
    api.post<CustomModelUploadSession>("/api/ml/custom-models/uploads", {}),

  cancelCustomModelUpload: (uploadId: string) =>
    api.delete<void>(`/api/ml/custom-models/uploads/${encodeURIComponent(uploadId)}`),

  inspectCustomModel: (sourcePath: string) =>
    api.post<CustomModelInspectResponse>("/api/ml/custom-models/inspect", { source_path: sourcePath }),

  createCustomModel: (model: CustomModelCreateRequest) =>
    api.post<CustomModelInfo>("/api/ml/custom-models", model),

  updateCustomModel: (modelId: string, changes: CustomModelUpdateRequest) =>
    api.put<CustomModelInfo>(`/api/ml/custom-models/${encodeURIComponent(modelId)}`, changes),

  deleteCustomModel: (modelId: string) =>
    api.delete<void>(`/api/ml/custom-models/${encodeURIComponent(modelId)}`),

  /**
   * Check if model weights and environment are ready
   */
  getModelStatus: (modelId: string) =>
    api.get<ModelStatusResponse>(`/api/ml/models/${modelId}/status`),

  /**
   * Prepare model (download weights + build environment)
   */
  prepareModel: (modelId: string) =>
    api.post<{ task_id: string }>(`/api/ml/models/${modelId}/prepare`),

  /**
   * Download model weights only
   */
  prepareWeights: (modelId: string) =>
    api.post(`/api/ml/models/${modelId}/prepare-weights`),

  /**
   * Build model environment only
   */
  prepareEnvironment: (modelId: string) =>
    api.post(`/api/ml/models/${modelId}/prepare-env`),

  /**
   * Get taxonomy tree for a classification model
   */
  getTaxonomy: (modelId: string) =>
    api.get<TaxonomyResponse>(`/api/ml/models/${modelId}/taxonomy`),

  /**
   * Get geofence information for a classification model
   */
  getModelGeofence: (modelId: string, country?: string, state?: string) => {
    const params = new URLSearchParams();
    if (country) params.set("country", country);
    if (state) params.set("state", state);
    const query = params.toString();
    return api.get<GeofenceResponse>(`/api/ml/models/${modelId}/geofence${query ? `?${query}` : ""}`);
  },
};
