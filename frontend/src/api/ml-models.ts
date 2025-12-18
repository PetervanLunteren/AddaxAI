/**
 * ML Models API client
 */

import type { ModelStatusResponse, ModelPrepareResponse } from "./types";

const API_BASE = "http://localhost:8000";

export const mlModelsApi = {
  /**
   * Check model status (weights + environment ready)
   */
  getStatus: async (modelId: string): Promise<ModelStatusResponse> => {
    const response = await fetch(`${API_BASE}/api/ml/models/${modelId}/status`);
    if (!response.ok) {
      throw new Error(`Failed to get model status: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Prepare model (download weights + build environment)
   */
  prepare: async (modelId: string): Promise<ModelPrepareResponse> => {
    const response = await fetch(`${API_BASE}/api/ml/models/${modelId}/prepare`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to prepare model: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Download model weights only
   */
  prepareWeights: async (modelId: string): Promise<ModelPrepareResponse> => {
    const response = await fetch(`${API_BASE}/api/ml/models/${modelId}/prepare-weights`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to download weights: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Build model environment only
   */
  prepareEnv: async (modelId: string): Promise<ModelPrepareResponse> => {
    const response = await fetch(`${API_BASE}/api/ml/models/${modelId}/prepare-env`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to build environment: ${response.statusText}`);
    }
    return response.json();
  },
};
