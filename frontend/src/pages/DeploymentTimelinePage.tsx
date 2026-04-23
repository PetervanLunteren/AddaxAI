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

import { timelineApi, type TimelineResponse, type TimelineSite } from "../api/timeline";
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
  PlotExplainer,
  type PlotReference,
} from "../components/plots/PlotExplainer";
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

const EXPLAINER_REFERENCES: PlotReference[] = [
  {
    citation:
      "Bubnicki, J. W., Norton, B., Baskauf, S. J., et al. (2024). "
      + "Camtrap DP: an open standard for the FAIR exchange and reuse of "
      + "camera trap data. Remote Sensing in Ecology and Conservation.",
    url: "https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.374",
  },
  {
    citation:
      "Niedballa, J., Sollmann, R., Courtiol, A., & Wilting, A. (2016). "
      + "camtrapR: an R package for efficient camera trap data management. "
      + "Methods in Ecology and Evolution, 7(12), 1457–1462.",
    url: "https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12600",
  },
  {
    citation:
      "Meek, P. D., Ballard, G., Claridge, A., et al. (2014). "
      + "Recommended guiding principles for reporting on camera trapping "
      + "research. Biodiversity and Conservation, 23(9), 2321–2343.",
  },
  {
    citation:
      "Rovero, F., & Zimmermann, F. (2016). Camera Trapping for Wildlife "
      + "Research. Pelagic Publishing.",
  },
  {
    citation:
      "Burton, A. C., Neilson, E., Moreira, D., et al. (2015). Wildlife "
      + "camera trapping: a review and recommendations for linking surveys "
      + "to ecological processes. Journal of Applied Ecology, 52(3), 675–685.",
    url: "https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/1365-2664.12432",
  },
];

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

  if (!projectId) return null;

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold tracking-tight">Deployment timeline</h1>
          <p className="text-sm text-muted-foreground">
            Survey effort over time, grouped by site, with concurrent-cameras
            coverage.
          </p>
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
              survey.
            </p>
          }
          how={
            <>
              <p>
                Trap-night intervals are computed the same way the Dashboard
                counts trap-nights, because both read the same primitive:
                files are bucketed by their parent subfolder, each subfolder
                becomes an inclusive [first-file, last-file] interval, and
                every interval is plotted as its own bar. The deployment's
                total is the sum of each subfolder's inclusive day count,
                minus 1 for every pair of subfolders that share an exact
                boundary day (the Reconyx / Bushnell rollover case where
                `100MEDIA` ends on the same day `101MEDIA` starts). The
                concurrent-cameras chart is a sweep line over every
                interval's endpoints, so parallel subfolders raise the
                concurrent count. Dates are rendered in the project's
                camera-local timezone; there is no timezone conversion.
              </p>
              <p>
                Each teal bar is a continuous capture interval: a
                subfolder's first-file to last-file span, with adjacent
                rollover subfolders (where one's end date equals the
                next's start date) merged into one bar. Quiet days inside
                an interval are still counted as trap-nights. Gaps only
                appear when no subfolder covers that window. This matches
                the trap-night convention used in the wider camera-trap
                literature (Rovero &amp; Zimmermann 2016, Meek et al.
                2014).
              </p>
            </>
          }
          references={EXPLAINER_REFERENCES}
        />
      </main>
    </div>
  );
}

export default DeploymentTimelinePage;
