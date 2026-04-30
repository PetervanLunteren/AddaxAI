/**
 * Insights → Deployment timeline page.
 *
 * Row-per-site Gantt with folder-aware trap-night intervals, plus a
 * concurrent-cameras area chart beneath the Gantt and a reactive
 * metrics strip above it. See `/Users/peter/.claude/plans/in-depth-plot-concurrent-waterfall.md`
 * for the design rationale.
 */

import { useMemo } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { sitesApi } from "../api/sites";
import { timelineApi, type TimelineResponse, type TimelineSite } from "../api/timeline";
import { BugReportButton } from "../components/diagnostics/BugReportButton";
import { DeploymentTimelineChart } from "../components/plots/DeploymentTimelineChart";
import {
  DeploymentTimelineFilterBar,
  type TimelineDensity,
  type TimelinePageFilters,
  type TimelineSort,
} from "../components/plots/DeploymentTimelineFilterBar";
import { DeploymentTimelineMetrics } from "../components/plots/DeploymentTimelineMetrics";
import { NoSiteBanner } from "../components/deployments/NoSiteBanner";
import {
  InsightsFilterChips,
  buildSiteNameMap,
  dateChips,
  siteChips,
} from "../components/plots/InsightsFilterChips";
import { PlotExplainer } from "../components/plots/PlotExplainer";
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
  sort: "string",
  density: "string",
};

const VALID_SORTS: TimelineSort[] = [
  "alpha",
  "chrono",
  "trap-nights",
  "deployments",
  "recent",
];

function parseSort(raw: string | undefined): TimelineSort {
  return VALID_SORTS.includes(raw as TimelineSort)
    ? (raw as TimelineSort)
    : "alpha";
}

function parseDensity(raw: string | undefined): TimelineDensity {
  return raw === "compact" ? "compact" : "normal";
}

function firstIntervalStart(site: TimelineSite): string {
  let earliest: string | null = null;
  for (const dep of site.deployments) {
    for (const iv of dep.intervals) {
      if (earliest === null || iv.start < earliest) earliest = iv.start;
    }
    if (earliest === null && dep.configured_start < (earliest ?? "9999")) {
      earliest = dep.configured_start;
    }
  }
  return earliest ?? "9999-12-31";
}

function lastIntervalEnd(site: TimelineSite): string {
  let latest: string | null = null;
  for (const dep of site.deployments) {
    for (const iv of dep.intervals) {
      if (latest === null || iv.end > latest) latest = iv.end;
    }
    if (dep.configured_end && (latest === null || dep.configured_end > latest)) {
      latest = dep.configured_end;
    }
  }
  return latest ?? "0000-01-01";
}

function totalTrapNights(site: TimelineSite): number {
  let total = 0;
  for (const dep of site.deployments) {
    for (const iv of dep.intervals) total += iv.trap_nights;
  }
  return total;
}

function sortSites(
  sites: TimelineSite[],
  sort: TimelineSort,
): TimelineSite[] {
  // Always pin (no-site) to the bottom regardless of the selection.
  const real = sites.filter((s) => s.site_id !== null);
  const noSite = sites.filter((s) => s.site_id === null);
  const sorted = [...real];
  switch (sort) {
    case "alpha":
      sorted.sort((a, b) => a.site_name.localeCompare(b.site_name));
      break;
    case "chrono":
      sorted.sort((a, b) =>
        firstIntervalStart(a).localeCompare(firstIntervalStart(b)),
      );
      break;
    case "trap-nights":
      sorted.sort((a, b) => totalTrapNights(b) - totalTrapNights(a));
      break;
    case "deployments":
      sorted.sort((a, b) => b.deployments.length - a.deployments.length);
      break;
    case "recent":
      sorted.sort((a, b) => lastIntervalEnd(b).localeCompare(lastIntervalEnd(a)));
      break;
  }
  return [...sorted, ...noSite];
}

export function DeploymentTimelinePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const filters = useMemo<TimelinePageFilters>(() => {
    const parsed = filtersFromSearchParams(searchParams, FILTER_SCHEMA);
    return {
      siteIds: (parsed.site_ids as string[] | undefined) ?? [],
      dateFrom: (parsed.date_from as string | undefined) ?? null,
      dateTo: (parsed.date_to as string | undefined) ?? null,
      sort: parseSort(parsed.sort as string | undefined),
      density: parseDensity(parsed.density as string | undefined),
    };
  }, [searchParams]);

  const handleFiltersChange = (next: TimelinePageFilters) => {
    setSearchParams(
      filtersToSearchParams(
        {
          site_ids: next.siteIds.length > 0 ? next.siteIds : undefined,
          date_from: next.dateFrom ?? undefined,
          date_to: next.dateTo ?? undefined,
          sort: next.sort === "alpha" ? undefined : next.sort,
          density: next.density === "normal" ? undefined : next.density,
        },
        FILTER_SCHEMA,
      ),
    );
  };

  const siteKey = filters.siteIds.slice().sort().join(",");
  const enabled = !!projectId;

  const { data, isLoading, isFetching } = useQuery({
    enabled,
    queryKey: [
      "timeline",
      projectId,
      siteKey,
      filters.dateFrom,
      filters.dateTo,
    ],
    queryFn: () =>
      timelineApi.get(projectId!, {
        siteIds: filters.siteIds.length > 0 ? filters.siteIds : undefined,
        dateFrom: filters.dateFrom ?? undefined,
        dateTo: filters.dateTo ?? undefined,
      }),
  });

  const sortedData = useMemo<TimelineResponse | undefined>(() => {
    if (!data) return undefined;
    return { ...data, sites: sortSites(data.sites, filters.sort) };
  }, [data, filters.sort]);

  const { data: noSite } = useNoSiteDeployments(projectId);

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId!),
    enabled: !!projectId,
  });

  const chips = useMemo(() => {
    const siteNames = buildSiteNameMap(sites);
    return [
      ...siteChips(filters.siteIds, siteNames, (next) =>
        handleFiltersChange({ ...filters, siteIds: next }),
      ),
      ...dateChips(
        filters.dateFrom,
        filters.dateTo,
        () => handleFiltersChange({ ...filters, dateFrom: null }),
        () => handleFiltersChange({ ...filters, dateTo: null }),
      ),
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, sites]);

  const clearAllDataFilters = () =>
    handleFiltersChange({ ...filters, siteIds: [], dateFrom: null, dateTo: null });

  if (!projectId) return null;

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Deployment timeline</h1>
              <p className="text-sm text-muted-foreground">
                Survey effort over time, grouped by site
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
          reason="They appear in a separate row at the bottom of the timeline."
        />
        <DeploymentTimelineFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
        />
        <InsightsFilterChips chips={chips} onClearAll={clearAllDataFilters} />

        <DeploymentTimelineMetrics
          metrics={sortedData?.metrics}
          loading={isLoading || isFetching}
        />

        <div className="rounded-lg border bg-card px-4 py-4">
          <DeploymentTimelineChart
            data={sortedData}
            loading={isLoading || isFetching}
            projectId={projectId}
            density={filters.density}
            onZoom={(from, to) =>
              handleFiltersChange({ ...filters, dateFrom: from, dateTo: to })
            }
          />
        </div>

        <PlotExplainer
          plotKey="deployment-timeline"
          what={
            <p>
              One row per site. Each teal bar is a folder-aware trap-night
              interval: the camera's first file to its last file in one
              subfolder. Whitespace between bars on the same row is time
              the site was not monitored. The area chart beneath shows how
              many cameras were active on each day across the whole
              survey. Drag horizontally across the date axis at the top to
              zoom into a specific range.
            </p>
          }
          how={
            <>
              <p>
                Each teal bar is one continuous camera session: the span
                from that camera's first capture to its last. Quiet days
                inside a session still count as active trap-nights, matching
                the standard convention used across the camera-trap
                literature. Whitespace between bars on a row means the site
                wasn't being monitored.
              </p>
              <p>
                Stacked bars on one row mean the deployment contained files
                from multiple cameras running at the same time. The
                concurrent-cameras chart underneath counts how many cameras
                were recording on each calendar day. Rows with many gaps, or
                deployments with tall stacks, usually flag uneven sampling
                effort, worth mentioning in methods or controlling for in
                downstream analyses.
              </p>
            </>
          }
        />
      </main>
    </div>
  );
}

export default DeploymentTimelinePage;
