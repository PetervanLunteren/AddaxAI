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
  total_observations: number;
  total_events: number;
  total_deployments: number;
  total_sites: number;
  trap_nights: number;
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

export interface SunBands {
  /** Fractional hour (0-24) when civil twilight starts. */
  dawn: number;
  /** Fractional hour when the sun clears the horizon. */
  sunrise: number;
  /** Fractional hour when the sun drops below the horizon. */
  sunset: number;
  /** Fractional hour when civil twilight ends. */
  dusk: number;
}

export interface ActivityPatternResponse {
  hours: HourlyCount[];
  total_observations: number;
  /** Null when the project has no sites or astral can't compute. */
  sun_bands: SunBands | null;
  /** Deployments in the filtered set that have no camera site assigned.
   * They are silently excluded from sun-band averaging; the UI
   * surfaces a banner when this is non-zero. */
  deployments_without_site: number;
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

export interface SpeciesObservationCount {
  label: string;
  label_taxonomy_id: string | null;
  count: number;
}

export interface ObservationRateMapFeature {
  deployment_id: string;
  site_id: string;
  site_name: string;
  latitude: number;
  longitude: number;
  start_date_local: string;
  end_date_local: string | null;
  trap_nights: number;
  observation_count: number;
  rate_per_100: number;
  species_breakdown: SpeciesObservationCount[];
}

export interface ObservationRateMapResponse {
  features: ObservationRateMapFeature[];
  /** Deployments that passed the filter but have no site (and thus no
   * lat/lon). Surfaces in the map page as a banner. */
  deployments_without_site: number;
}

export interface ObservationRateMapFilters {
  siteIds?: string[];
  dateFrom?: string;
  dateTo?: string;
  labelTaxonomyIds?: string[];
}

// --- Activity overlap (Plots → Activity overlap page) ---

export type DielClass =
  | "diurnal"
  | "nocturnal"
  | "crepuscular"
  | "cathemeral";

export type DeltaEstimator = "delta1" | "delta4";

export type SampleSizeWarning = "low_n_30" | "low_n_50" | "low_n_75";

export type TimeAxis = "clock" | "sun";

export interface SpeciesActivity {
  label: string;
  n: number;
  /** Decimal hours [0..24) for the rug ticks under the curve. Capped at 5000. */
  raw_detection_times: number[];
  /** 240-point von Mises KDE over [0..24), normalized to integrate to 1. */
  kde_density: number[];
  diel_class: DielClass;
  /** {"day": number, "night": number, "twilight": number} summing to ~1. */
  diel_density_by_phase: Record<string, number>;
  sample_size_warning: SampleSizeWarning | null;
  /** Observations skipped because their date had no defined sunrise (polar). */
  dropped_polar: number;
}

export interface OverlapStat {
  delta_estimator: DeltaEstimator;
  delta: number;
  ci_low: number;
  ci_high: number;
  bootstrap_reps: number;
  min_n: number;
}

export interface ActivityOverlapResponse {
  species_a: SpeciesActivity;
  species_b: SpeciesActivity | null;
  overlap: OverlapStat | null;
  /** Single-reference clock-time bands (midpoint date, project avg lat/lon). */
  sun_bands: SunBands | null;
  /** ISO date (YYYY-MM-DD) the clock-mode bands were computed for. Null when sun_bands is null. */
  sun_bands_reference_date: string | null;
  /** Mean-anchored dawn/sunrise/sunset/dusk across observations. Non-null only in sun mode. */
  anchor_sun_bands: SunBands | null;
  /** The axis the returned KDE is in. May differ from the requested axis when
   * sun mode was not possible (no site coordinates, all-polar dataset) and
   * the backend silently downgraded to clock. */
  time_axis: TimeAxis;
  /** IANA timezone the project's camera clocks are set to. */
  project_timezone: string;
  independence_interval_seconds: number;
  /** Deployments in the filtered set that have no camera site assigned.
   * They are silently excluded from sun-band averaging; the UI
   * surfaces a banner in sun mode when this is non-zero. */
  deployments_without_site: number;
}

export interface ActivityOverlapFilters {
  speciesA: string;
  speciesB?: string;
  siteIds?: string[];
  dateFrom?: string;
  dateTo?: string;
  taxonomicRank?: string;
  timeAxis?: TimeAxis;
}

// --- Shared helpers ---

/**
 * Build query string with project_id and optional filter params.
 * Only includes params that are actually provided.
 */
function buildParams(
  projectId: string,
  options?: { species?: string; siteIds?: string; dateFrom?: string; dateTo?: string; taxonomicRank?: string }
): string {
  const params = new URLSearchParams();
  params.set("project_id", projectId);

  if (options?.species) params.set("species", options.species);
  if (options?.siteIds) params.set("site_ids", options.siteIds);
  if (options?.dateFrom) params.set("date_from", options.dateFrom);
  if (options?.dateTo) params.set("date_to", options.dateTo);
  if (options?.taxonomicRank) params.set("taxonomic_rank", options.taxonomicRank);

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
  getSpeciesDistribution: (projectId: string, siteIds?: string, dateFrom?: string, dateTo?: string, taxonomicRank?: string, countMode?: string) => {
    const query = buildParams(projectId, { siteIds, dateFrom, dateTo, taxonomicRank });
    const modeParam = countMode ? `&count_mode=${countMode}` : "";
    return api.get<SpeciesCount[]>(`/api/statistics/species?${query}${modeParam}`);
  },

  /**
   * Hourly activity pattern, optionally filtered by species
   */
  getActivityPattern: (
    projectId: string,
    params?: { species?: string; siteIds?: string; dateFrom?: string; dateTo?: string; taxonomicRank?: string }
  ) => {
    const query = buildParams(projectId, params);
    return api.get<ActivityPatternResponse>(`/api/statistics/activity-pattern?${query}`);
  },

  /**
   * Daily detection trend over time, optionally filtered by species
   */
  getDetectionTrend: (
    projectId: string,
    params?: { species?: string; siteIds?: string; dateFrom?: string; dateTo?: string; taxonomicRank?: string }
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

  /**
   * Per-deployment observation rate features for the Map page.
   * Rate = observations / trap nights * 100, where observations is
   * sum(EventObservation.max_n) per event.
   */
  getObservationRateMap: (
    projectId: string,
    filters?: ObservationRateMapFilters
  ) => {
    const params = new URLSearchParams();
    params.set("project_id", projectId);
    if (filters?.siteIds?.length) {
      params.set("site_ids", filters.siteIds.join(","));
    }
    if (filters?.dateFrom) params.set("date_from", filters.dateFrom);
    if (filters?.dateTo) params.set("date_to", filters.dateTo);
    if (filters?.labelTaxonomyIds?.length) {
      params.set("label_taxonomy_ids", filters.labelTaxonomyIds.join(","));
    }
    return api.get<ObservationRateMapResponse>(
      `/api/statistics/observation-rate-map?${params.toString()}`
    );
  },

  /**
   * Activity overlap payload for the Plots → Activity overlap page.
   * Returns 1- or 2-species KDE curves, sun bands, diel classification,
   * and the Ridout & Linkie overlap coefficient Δ with bootstrap CI
   * (when both species have data).
   */
  getActivityOverlap: (
    projectId: string,
    filters: ActivityOverlapFilters
  ) => {
    const params = new URLSearchParams();
    params.set("project_id", projectId);
    params.set("species_a", filters.speciesA);
    if (filters.speciesB) params.set("species_b", filters.speciesB);
    if (filters.siteIds?.length) {
      params.set("site_ids", filters.siteIds.join(","));
    }
    if (filters.dateFrom) params.set("date_from", filters.dateFrom);
    if (filters.dateTo) params.set("date_to", filters.dateTo);
    if (filters.taxonomicRank) params.set("taxonomic_rank", filters.taxonomicRank);
    if (filters.timeAxis) params.set("time_axis", filters.timeAxis);
    return api.get<ActivityOverlapResponse>(
      `/api/statistics/activity-overlap?${params.toString()}`
    );
  },
};
