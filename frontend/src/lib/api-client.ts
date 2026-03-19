/**
 * API client for AddaxAI backend.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit error handling
 * - No silent failures
 */

import { logger } from "./logger";

export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

/**
 * Base fetch wrapper with error handling.
 *
 * Crashes (throws) on network errors or non-2xx responses.
 * This is intentional - we want to surface errors immediately.
 */
async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const method = options?.method || "GET";

  logger.info(`API ${method} ${endpoint}`);

  try {
    // Skip Content-Type for FormData (browser sets multipart boundary)
    const isFormData = options?.body instanceof FormData;
    const headers = isFormData
      ? { ...options?.headers }
      : { "Content-Type": "application/json", ...options?.headers };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    // Handle non-2xx responses
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const detail = errorData.detail;
      const errorMsg =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((e: { msg?: string }) => e.msg || JSON.stringify(e)).join("; ")
            : `HTTP ${response.status}: ${response.statusText}`;

      logger.error(`API ${method} ${endpoint} failed: ${errorMsg}`, {
        status: response.status,
        endpoint,
      });

      throw new Error(errorMsg);
    }

    // Handle 204 No Content
    if (response.status === 204) {
      logger.info(`API ${method} ${endpoint} → 204 No Content`);
      return undefined as T;
    }

    logger.info(`API ${method} ${endpoint} → ${response.status} OK`);
    return await response.json();
  } catch (error) {
    // Abort errors are expected (tab switches, React Strict Mode) — re-throw silently
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }

    // Re-throw with more context
    if (error instanceof Error) {
      // Don't log again if we already logged above
      if (!error.message.includes("HTTP")) {
        logger.error(`API ${method} ${endpoint} error: ${error.message}`, {
          endpoint,
          error: error.message,
        });
      }
      throw new Error(`API request failed: ${error.message}`);
    }
    logger.error(`API ${method} ${endpoint} unknown error`, { endpoint });
    throw error;
  }
}

export const api = {
  /**
   * GET request
   */
  get: <T>(endpoint: string, options?: { signal?: AbortSignal }): Promise<T> => {
    return apiFetch<T>(endpoint, { method: "GET", signal: options?.signal });
  },

  /**
   * POST request
   */
  post: <T>(endpoint: string, data?: unknown, options?: { signal?: AbortSignal }): Promise<T> => {
    return apiFetch<T>(endpoint, {
      method: "POST",
      body: data ? JSON.stringify(data) : undefined,
      signal: options?.signal,
    });
  },

  /**
   * PATCH request
   */
  patch: <T>(endpoint: string, data: unknown): Promise<T> => {
    return apiFetch<T>(endpoint, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  },

  /**
   * DELETE request
   */
  delete: <T>(endpoint: string): Promise<T> => {
    return apiFetch<T>(endpoint, { method: "DELETE" });
  },

  /**
   * Upload a file via multipart form data
   */
  upload: <T>(endpoint: string, file: File, fieldName = "file"): Promise<T> => {
    const formData = new FormData();
    formData.append(fieldName, file);
    return apiFetch<T>(endpoint, { method: "POST", body: formData });
  },
};
