/**
 * Deployment timeline endpoint (Insights → Deployment timeline page).
 *
 * Types mirror backend/app/api/schemas/timeline.py.
 */

import { api } from "../lib/api-client";

export interface TrapNightInterval {
  start: string; // YYYY-MM-DD
  end: string;
  trap_nights: number;
}

export interface TimelineDeployment {
  deployment_id: string;
  deployment_label: string;
  camera_model: string | null;
  configured_start: string;
  configured_end: string | null;
  intervals: TrapNightInterval[];
  file_count: number;
}

export interface TimelineSite {
  site_id: string | null;
  site_name: string;
  deployments: TimelineDeployment[];
}

export interface ConcurrentPoint {
  date: string;
  count: number;
}

export interface TimelineMetrics {
  site_count: number;
  deployment_count: number;
  total_trap_nights: number;
  median_deployment_length_days: number | null;
  max_concurrent_cameras: number;
}

export interface TimelineResponse {
  sites: TimelineSite[];
  concurrent_cameras: ConcurrentPoint[];
  metrics: TimelineMetrics;
  date_range_from: string | null;
  date_range_to: string | null;
}

export interface TimelineFilters {
  siteIds?: string[];
  dateFrom?: string;
  dateTo?: string;
}

export const timelineApi = {
  get: (projectId: string, filters: TimelineFilters = {}) => {
    const params = new URLSearchParams();
    params.set("project_id", projectId);
    if (filters.siteIds?.length) params.set("site_ids", filters.siteIds.join(","));
    if (filters.dateFrom) params.set("date_from", filters.dateFrom);
    if (filters.dateTo) params.set("date_to", filters.dateTo);
    return api.get<TimelineResponse>(
      `/api/statistics/timeline?${params.toString()}`
    );
  },
};
