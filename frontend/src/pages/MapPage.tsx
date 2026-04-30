/**
 * Map page — spatial view of deployments colored by observation rate.
 *
 * Page owns:
 *   - Filter state (sites, dates, labels) — persisted to the URL
 *   - View mode (hexbins / points / clusters) — persisted to localStorage
 *   - Base layer (positron / satellite / osm) — persisted to localStorage
 *
 * Passes all of it down to MapFilterBar (the controls) and
 * ObservationRateMap (the renderer) so the two stay in sync.
 */

import { useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { eventsApi } from "../api/events";
import { sitesApi } from "../api/sites";
import { BugReportButton } from "../components/diagnostics/BugReportButton";
import type { ObservationRateMapFilters } from "../api/statistics";
import {
  MapFilterBar,
  type BaseLayer,
  type MapFilters,
  type ViewMode,
} from "../components/map/MapFilterBar";
import { ObservationRateMap } from "../components/map/ObservationRateMap";
import { NoSiteBanner } from "../components/deployments/NoSiteBanner";
import { PlotExplainer } from "../components/plots/PlotExplainer";
import {
  InsightsFilterChips,
  buildSiteNameMap,
  dateChips,
  labelChips,
  siteChips,
} from "../components/plots/InsightsFilterChips";
import { useNoSiteDeployments } from "../hooks/useNoSiteDeployments";
import {
  filtersFromSearchParams,
  filtersToSearchParams,
  type FilterSchema,
} from "../lib/filter-url";

const FILTER_SCHEMA: FilterSchema = {
  site_ids: "string[]",
  date_from: "date",
  date_to: "date",
  labels: "string[]",
};

const VIEW_MODE_KEY = "addaxai:map-view-mode";
const BASE_LAYER_KEY = "addaxai:map-base-layer";

function readStoredViewMode(): ViewMode {
  const saved = localStorage.getItem(VIEW_MODE_KEY);
  if (saved === "hexbins" || saved === "points" || saved === "clusters") {
    return saved;
  }
  return "hexbins";
}

function readStoredBaseLayer(): BaseLayer {
  const saved = localStorage.getItem(BASE_LAYER_KEY);
  if (saved === "positron" || saved === "satellite" || saved === "osm") {
    return saved;
  }
  return "positron";
}

export function MapPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const [viewMode, setViewMode] = useState<ViewMode>(readStoredViewMode);
  const [baseLayer, setBaseLayer] = useState<BaseLayer>(readStoredBaseLayer);

  useEffect(() => {
    localStorage.setItem(VIEW_MODE_KEY, viewMode);
  }, [viewMode]);

  useEffect(() => {
    localStorage.setItem(BASE_LAYER_KEY, baseLayer);
  }, [baseLayer]);

  const filters = useMemo<MapFilters>(() => {
    const parsed = filtersFromSearchParams(searchParams, FILTER_SCHEMA);
    return {
      site_ids: parsed.site_ids as string[] | undefined,
      date_from: parsed.date_from as string | undefined,
      date_to: parsed.date_to as string | undefined,
      labels: parsed.labels as string[] | undefined,
    };
  }, [searchParams]);

  const handleFiltersChange = (next: MapFilters) => {
    setSearchParams(
      filtersToSearchParams(
        {
          site_ids: next.site_ids,
          date_from: next.date_from,
          date_to: next.date_to,
          labels: next.labels,
        },
        FILTER_SCHEMA
      )
    );
  };

  const apiFilters: ObservationRateMapFilters = useMemo(
    () => ({
      siteIds: filters.site_ids,
      dateFrom: filters.date_from,
      dateTo: filters.date_to,
      labelTaxonomyIds: filters.labels,
    }),
    [filters]
  );

  const { data: noSite } = useNoSiteDeployments(projectId);

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId!),
    enabled: !!projectId,
  });
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId!),
    enabled: !!projectId,
  });

  const chips = useMemo(() => {
    const siteNames = buildSiteNameMap(sites);
    return [
      ...siteChips(filters.site_ids, siteNames, (next) =>
        handleFiltersChange({
          ...filters,
          site_ids: next.length ? next : undefined,
        }),
      ),
      ...dateChips(
        filters.date_from,
        filters.date_to,
        () => handleFiltersChange({ ...filters, date_from: undefined }),
        () => handleFiltersChange({ ...filters, date_to: undefined }),
      ),
      ...labelChips(filters.labels, filterOptions?.display_labels, (next) =>
        handleFiltersChange({
          ...filters,
          labels: next.length ? next : undefined,
        }),
      ),
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, sites, filterOptions]);

  const clearAllDataFilters = () => {
    handleFiltersChange({
      site_ids: undefined,
      date_from: undefined,
      date_to: undefined,
      labels: undefined,
    });
  };

  if (!projectId) {
    return <div>Project ID missing</div>;
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Map</h1>
              <p className="text-sm text-muted-foreground">
                Observation rate per 100 trap nights
              </p>
            </div>
            <BugReportButton />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <NoSiteBanner
          projectId={projectId}
          count={noSite?.count ?? 0}
          reason="They are not shown on the map."
        />
        <MapFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          baseLayer={baseLayer}
          onBaseLayerChange={setBaseLayer}
        />
        <InsightsFilterChips chips={chips} onClearAll={clearAllDataFilters} />
        <ObservationRateMap
          projectId={projectId}
          filters={apiFilters}
          viewMode={viewMode}
          baseLayer={baseLayer}
        />

        <PlotExplainer
          plotKey="map"
          what={
            <p>
              A marker per deployment, coloured by its observation rate
              per 100 trap nights. Three layer modes: hexbins aggregate
              nearby deployments onto a hex grid, points show each
              deployment individually, and clusters group nearby points
              into a single circle with the count inside. The labels
              filter restricts the observation count to the selected
              taxa.
            </p>
          }
          how={
            <p>
              rate = observations / trap_nights × 100, where
              observations is the sum of MaxN across all events passing
              the active filters, and trap_nights is the deployment's
              active days. Events respect the project's detection
              threshold with the verified override applied, so verified
              detections count even when they fall below threshold.
              Hexbin colour scaling is per-render, so a hex's shade
              reflects its rank within the current view rather than an
              absolute comparison across projects.
            </p>
          }
        />
      </main>
    </div>
  );
}

export default MapPage;
