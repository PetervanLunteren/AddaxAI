/**
 * Models API client.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";
import type { ModelInfo, ModelStatusResponse, TaxonomyResponse } from "./types";

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
   * Check if model weights and environment are ready
   */
  getModelStatus: (modelId: string) =>
    api.get<ModelStatusResponse>(`/api/ml/models/${modelId}/status`),

  /**
   * Download model weights
   */
  prepareWeights: (modelId: string) =>
    api.post(`/api/ml/models/${modelId}/prepare-weights`),

  /**
   * Build model environment
   */
  prepareEnvironment: (modelId: string) =>
    api.post(`/api/ml/models/${modelId}/prepare-env`),

  /**
   * Get taxonomy tree for a classification model
   */
  getTaxonomy: (modelId: string) =>
    api.get<TaxonomyResponse>(`/api/ml/models/${modelId}/taxonomy`),
};
