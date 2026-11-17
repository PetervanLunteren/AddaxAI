/**
 * Trend page: the observation rate over time, with its uncertainty.
 *
 * The site is the sampling unit. Each site gets its own rate, the chart
 * draws the mean of those rates per week or month with a 95% confidence
 * band, and the table below gives the same per species for the whole
 * date range (backend: `crud/rate_trend.py`).
 *
 * Filters are the Explore tab's (labels, sites, site tags, dates), with
 * the same hook, bar and URL parameters, so a link from Explore lands on
 * the same selection. The period size is a control on the chart card: it
 * changes how the chart is drawn, not which data is in scope.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { eventsApi } from "../api/events";
import { statisticsApi } from "../api/statistics";
import { ExploreFilterBar } from "../components/dashboard/ExploreFilterBar";
import { MissingDatesBanner } from "../components/dashboard/MissingDatesWarning";
import { NoSiteBanner } from "../components/deployments/NoSiteBanner";
import { buildSiteNameMap } from "../components/plots/InsightsFilterChips";
import { PlotExplainer } from "../components/plots/PlotExplainer";
import { RateTrendCard, type TrendPeriodSize } from "../components/plots/RateTrendCard";
import { SpeciesRateTable } from "../components/plots/SpeciesRateTable";
import { Callout } from "../components/ui/callout";
import { useNoSiteDeployments } from "../hooks/useNoSiteDeployments";
import { downloadBlob } from "../lib/download";
import { speciesLabelMap } from "../lib/species-name-mode";
import { speciesRatesCsv, trendPeriodsCsv, type TrendCsvScope } from "../lib/trend-csv";
import { useExploreFilters } from "./dashboard/useExploreFilters";

/** Saves a download named after what it holds and its dates, so a
 *  "before" and an "after" file can be told apart. */
function saveCsv(name: string, fileScope: TrendCsvScope, csv: string) {
  const { dateFrom, dateTo } = fileScope;
  const dates = dateFrom && dateTo ? `-${dateFrom}-to-${dateTo}` : "";
  downloadBlob(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
    `addaxai-${name}${dates}.csv`,
  );
}

/** Names for a download's filter columns, or "all" without a filter. */
function namesOrAll(ids: string[] | undefined, names: Record<string, string>): string {
  return ids?.length ? ids.map((id) => names[id] ?? id).join("; ") : "all";
}

export function TrendPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const id = projectId as string;
  const explore = useExploreFilters(id);
  const { scope, ready } = explore;
  const showCards = ready && !explore.noSiteMatch;

  // Until the user picks one, the backend chooses week or month from
  // the span of the data.
  const [picked, setPicked] = useState<TrendPeriodSize>();

  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "rate-trend", id, scope, picked],
    queryFn: () => statisticsApi.getRateTrend(id, scope, picked),
    enabled: showCards,
  });
  const periodSize: TrendPeriodSize =
    picked ?? (data?.granularity === "month" ? "month" : "week");

  const { data: noSite } = useNoSiteDeployments(id);
  // Same query as the filter bar's, so this is a cache hit.
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", id],
    queryFn: () => eventsApi.getFilterOptions(id),
  });

  const csvScope = (): TrendCsvScope => ({
    // The filter when one is set, else the first and last day with effort.
    dateFrom: scope.dateFrom ?? data?.first_date ?? "",
    dateTo: scope.dateTo ?? data?.last_date ?? "",
    sites: namesOrAll(scope.siteIds, buildSiteNameMap(explore.sites)),
    labels: namesOrAll(
      scope.labelTaxonomyIds,
      filterOptions ? speciesLabelMap(filterOptions) : {},
    ),
  });

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-card/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Trend</h1>
              <p className="text-sm text-muted-foreground">
                Observation rate over time, with its uncertainty
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <NoSiteBanner
          projectId={id}
          count={noSite?.count ?? 0}
          reason="They are not part of the trend."
        />
        <MissingDatesBanner projectId={id} />
        <ExploreFilterBar
          projectId={id}
          filters={explore.filters}
          scopeSiteIds={scope.siteIds}
          sites={explore.sites}
          update={explore.update}
          clearAll={explore.clearAll}
        />

        {explore.noSiteMatch ? (
          <Callout variant="info">
            No site has these tags. Pick other sites or tags to see results.
          </Callout>
        ) : !ready ? (
          <p className="text-sm text-muted-foreground">Loading...</p>
        ) : (
          <>
            <RateTrendCard
              projectId={id}
              data={data}
              isLoading={isLoading}
              periodSize={periodSize}
              onPeriodSizeChange={setPicked}
              onDownload={() => {
                const fileScope = csvScope();
                saveCsv(
                  `trend-${periodSize}`,
                  fileScope,
                  trendPeriodsCsv(data?.periods ?? [], fileScope),
                );
              }}
            />
            <SpeciesRateTable
              species={data?.species ?? []}
              isLoading={isLoading}
              onDownload={() => {
                const fileScope = csvScope();
                saveCsv(
                  "species-rates",
                  fileScope,
                  speciesRatesCsv(data?.species ?? [], fileScope),
                );
              }}
            />
          </>
        )}

        <PlotExplainer
          plotKey="trend"
          what={
            <>
              <p>
                Each site gets its own observation rate. The line is the
                average of those site rates per week or month, and the band
                shows how sure that average is: with more sites, or sites
                that agree, the band is narrower. The table gives the same
                per species for the whole date range.
              </p>
              <p>
                An observation counts individuals: each event's confirmed
                count, or the AI's count where not yet confirmed. A trap
                night is one day a camera was active. The average of the
                site rates differs from the single rate on the dashboard
                when sites ran for different lengths of time: here every
                site weighs the same.
              </p>
            </>
          }
          how={
            <>
              <p>
                Per site, rate = observations / trap_nights, shown in the
                unit chosen under View &gt; Rate unit. A site with effort
                and no observations is a rate of zero. Then mean and
                standard deviation across the n sites, and 95% confidence
                interval = mean ± t × SD / √n, with t from Student's
                distribution for n − 1 degrees of freedom. The lower bound
                stops at zero.
              </p>
              <p>
                In the chart a site counts in a week or month when it was
                active on at least 80% of its days, so a camera that
                started or stopped partway does not distort the period. A
                period without such a site is a gap, not a zero. In the
                table every site with at least one trap night in the date
                range counts. Deployments without a site are left out.
              </p>
              <p>
                With few sites the interval is wide and rough. Read the
                number of sites next to it.
              </p>
            </>
          }
        />
      </main>
    </div>
  );
}

export default TrendPage;
