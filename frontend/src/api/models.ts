/**
 * Models API client.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import type { QueryClient } from "@tanstack/react-query";

import { api } from "../lib/api-client";
import type {
  GeofenceResponse,
  ModelInfo,
  ModelPrepareResponse,
  ModelStatusResponse,
  TaxonomyResponse,
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

  /**
   * Check if model weights and environment are ready
   */
  getModelStatus: (modelId: string) =>
    api.get<ModelStatusResponse>(`/api/ml/models/${modelId}/status`),

  /**
   * Prepare model (download weights + build environment). Go through
   * `useModelPrepare`, which handles the licence step this can refuse with.
   */
  prepareModel: (modelId: string) =>
    api.post<ModelPrepareResponse>(`/api/ml/models/${modelId}/prepare`),

  /**
   * Record that the user accepts the ultralytics licence (AGPL-3.0), once
   * for this installation. The prepare that was refused can then be retried.
   */
  acceptAgpl: () =>
    api.post<{ status: "accepted"; marker_path: string }>("/api/ml/accept-agpl"),

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
