/**
 * Map page — spatial view of sites colored by observation rate.
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

import { sequencesApi } from "../api/sequences";
import { sitesApi } from "../api/sites";
import { speciesLabelMap } from "../lib/species-name-mode";
import {
  statisticsApi,
  type ObservationRateMapFeature,
  type ObservationRateMapFilters,
} from "../api/statistics";
import {
  MapFilterBar,
  type MapFilters,
  type ViewMode,
} from "../components/map/MapFilterBar";
import { ObservationRateMap } from "../components/map/ObservationRateMap";
import { NoSiteBanner } from "../components/deployments/NoSiteBanner";
import { MissingDatesBanner } from "../components/dashboard/MissingDatesWarning";
import { PlotExplainer } from "../components/plots/PlotExplainer";
import {
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
import { PageHeader } from "../components/layout/PageHeader";
import { DATA_CSV, DownloadMenu } from "../components/plots/DownloadMenu";
import { isRateRamp, type RateRamp } from "../lib/heat-color-scale";
import { mapSitesCsv, rateCsvScope, saveCsv } from "../lib/insights-csv";

const FILTER_SCHEMA: FilterSchema = {
  site_ids: "string[]",
  date_from: "date",
  date_to: "date",
  labels: "string[]",
};

const VIEW_MODE_KEY = "addaxai:map-view-mode";
const RAMP_KEY = "addaxai:map-color-ramp";

function readStoredViewMode(): ViewMode {
  const saved = localStorage.getItem(VIEW_MODE_KEY);
  if (saved === "hexbins" || saved === "points" || saved === "clusters") {
    return saved;
  }
  return "hexbins";
}

function readStoredRamp(): RateRamp {
  const saved = localStorage.getItem(RAMP_KEY);
  return isRateRamp(saved) ? saved : "teal";
}

export function MapPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const [viewMode, setViewMode] = useState<ViewMode>(readStoredViewMode);
  const [ramp, setRamp] = useState<RateRamp>(readStoredRamp);

  useEffect(() => {
    localStorage.setItem(VIEW_MODE_KEY, viewMode);
  }, [viewMode]);

  useEffect(() => {
    localStorage.setItem(RAMP_KEY, ramp);
  }, [ramp]);

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

  const { data, isLoading, error } = useQuery({
    queryKey: ["observation-rate-map", projectId, apiFilters],
    queryFn: () => statisticsApi.getObservationRateMap(projectId!, apiFilters),
    enabled: !!projectId,
  });

  const { data: noSite } = useNoSiteDeployments(projectId);

  // Every site in the filters, not only those in view.
  const saveSitesCsv = (features: ObservationRateMapFeature[]) => {
    const scope = rateCsvScope({
      dateFrom: filters.date_from,
      dateTo: filters.date_to,
      siteIds: filters.site_ids,
      siteNames: buildSiteNameMap(sites),
      labelIds: filters.labels,
      labelNames: filterOptions ? speciesLabelMap(filterOptions) : {},
    });
    saveCsv("map-sites", mapSitesCsv(features, scope), {
      from: scope.dateFrom,
      to: scope.dateTo,
    });
  };

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId!),
    enabled: !!projectId,
  });
  const { data: filterOptions } = useQuery({
    queryKey: ["sequence-filter-options", projectId],
    queryFn: () => sequencesApi.getFilterOptions(projectId!),
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
      ...labelChips(filters.labels, filterOptions ? speciesLabelMap(filterOptions) : undefined, (next) =>
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
      <PageHeader
        title="Map"
        subtitle="Compare observation rates across your sites"
        actions={
          <DownloadMenu
            items={
              data?.features.length
                ? [{ label: DATA_CSV, onSelect: () => saveSitesCsv(data.features) }]
                : []
            }
          />
        }
      />

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <NoSiteBanner
          projectId={projectId}
          count={noSite?.count ?? 0}
          reason="They are not shown on the map."
        />
        <MissingDatesBanner projectId={projectId} />
        <MapFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          ramp={ramp}
          onRampChange={setRamp}
          chips={chips}
          onClearAll={clearAllDataFilters}
        />
        <ObservationRateMap
          data={data}
          isLoading={isLoading}
          error={error}
          viewMode={viewMode}
          ramp={ramp}
        />

        <PlotExplainer
          plotKey="map"
          what={
            <>
              <p>
                A marker per site, coloured by its observation rate.
                A site can have multiple deployments
                over time; each marker aggregates across them. Three
                layer modes: hexbins aggregate nearby sites onto a hex
                grid, points show each site individually, and clusters
                group nearby points into a single circle with the count
                inside. The labels filter restricts the observation
                count to the selected taxa.
              </p>
              <p>
                An observation counts individuals: each event's confirmed
                count, or the AI's count where not yet confirmed, which is
                the most individuals visible in a single frame, so the same
                animals aren't counted twice. A trap night is one day a
                camera was active.
              </p>
            </>
          }
          how={
            <p>
              rate = observations / trap_nights, shown in the unit
              chosen under View &gt; Rate unit, summed per site
              across its deployments and restricted to the active date
              and label filters. Sequences respect the project's detection
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
