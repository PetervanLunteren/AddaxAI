/**
 * Statistics API endpoints.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit operations
 */

import { api } from "../lib/api-client";

// --- Response types (matching backend schemas) ---

export interface DashboardOverview {
  total_files: number;
  total_detections: number;
  total_events: number;
  total_deployments: number;
  total_sites: number;
  first_file_date: string | null;
  last_file_date: string | null;
}

export interface SpeciesCount {
  species: string;
  count: number;
}

export interface HourlyCount {
  hour: number;
  count: number;
}

export interface ActivityPatternResponse {
  hours: HourlyCount[];
  total_detections: number;
}

export interface DetectionTrendPoint {
  date: string;
  count: number;
}

export interface DetectionCategories {
  animal_count: number;
  person_count: number;
  vehicle_count: number;
  empty_count: number;
}

export interface VerificationProgress {
  total_files: number;
  verified_files: number;
}

// --- Shared helpers ---

/**
 * Build query string with project_id and optional filter params.
 * Only includes params that are actually provided.
 */
function buildParams(
  projectId: string,
  options?: { species?: string; siteIds?: string; dateFrom?: string; dateTo?: string }
): string {
  const params = new URLSearchParams();
  params.set("project_id", projectId);

  if (options?.species) params.set("species", options.species);
  if (options?.siteIds) params.set("site_ids", options.siteIds);
  if (options?.dateFrom) params.set("date_from", options.dateFrom);
  if (options?.dateTo) params.set("date_to", options.dateTo);

  return params.toString();
}

// --- API client ---

export const statisticsApi = {
  /**
   * Dashboard overview counts (files, detections, events, etc.)
   */
  getOverview: (projectId: string, siteIds?: string, dateFrom?: string, dateTo?: string) => {
    const query = buildParams(projectId, { siteIds, dateFrom, dateTo });
    return api.get<DashboardOverview>(`/api/statistics/overview?${query}`);
  },

  /**
   * Species distribution (species name + detection count)
   */
  getSpeciesDistribution: (projectId: string, siteIds?: string, dateFrom?: string, dateTo?: string) => {
    const query = buildParams(projectId, { siteIds, dateFrom, dateTo });
    return api.get<SpeciesCount[]>(`/api/statistics/species?${query}`);
  },

  /**
   * Hourly activity pattern, optionally filtered by species
   */
  getActivityPattern: (
    projectId: string,
    params?: { species?: string; siteIds?: string; dateFrom?: string; dateTo?: string }
  ) => {
    const query = buildParams(projectId, params);
    return api.get<ActivityPatternResponse>(`/api/statistics/activity-pattern?${query}`);
  },

  /**
   * Daily detection trend over time, optionally filtered by species
   */
  getDetectionTrend: (
    projectId: string,
    params?: { species?: string; siteIds?: string; dateFrom?: string; dateTo?: string }
  ) => {
    const query = buildParams(projectId, params);
    return api.get<DetectionTrendPoint[]>(`/api/statistics/detection-trend?${query}`);
  },

  /**
   * Detection category breakdown (animal, person, vehicle, empty)
   */
  getDetectionCategories: (projectId: string, siteIds?: string, dateFrom?: string, dateTo?: string) => {
    const query = buildParams(projectId, { siteIds, dateFrom, dateTo });
    return api.get<DetectionCategories>(`/api/statistics/categories?${query}`);
  },

  /**
   * Verification progress (total vs verified file counts)
   */
  getVerificationProgress: (projectId: string, siteIds?: string, dateFrom?: string, dateTo?: string) => {
    const query = buildParams(projectId, { siteIds, dateFrom, dateTo });
    return api.get<VerificationProgress>(`/api/statistics/verification-progress?${query}`);
  },
};
