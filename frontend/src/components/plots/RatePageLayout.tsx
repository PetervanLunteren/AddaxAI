/**
 * The frame the Trend and Species rates pages share: the page header
 * with its Download CSV button, the banners, the Explore tab's filter
 * bar, and "About this view" under the page's one chart or table.
 *
 * One component so the two pages cannot drift. Both take the site as the
 * sampling unit and both read the Explore tab's filters from the URL, so
 * a link from Explore lands on the same selection on either page.
 */

import { useQuery } from "@tanstack/react-query";

import { eventsApi } from "../../api/events";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { speciesLabelMap } from "../../lib/species-name-mode";
import { saveTrendCsv, trendCsvScope, type TrendCsvScope } from "../../lib/trend-csv";
import type { useExploreFilters } from "../../pages/dashboard/useExploreFilters";
import { ExploreFilterBar } from "../dashboard/ExploreFilterBar";
import { MissingDatesBanner } from "../dashboard/MissingDatesWarning";
import { NoSiteBanner } from "../deployments/NoSiteBanner";
import { Callout } from "../ui/callout";
import { DownloadCsvButton } from "./DownloadCsvButton";
import { buildSiteNameMap } from "./InsightsFilterChips";

interface RatePageLayoutProps {
  projectId: string;
  title: string;
  subtitle: string;
  explore: ReturnType<typeof useExploreFilters>;
  /** First and last day with effort in scope, from the page's response. */
  dates: { first: string | null; last: string | null } | undefined;
  /** What Download CSV saves: a file name and the rows as CSV. Left out
   *  while there is nothing to save, which disables the button. */
  download?: { name: string; csv: (scope: TrendCsvScope) => string };
  /** The page's own display control in the filter bar. */
  filterExtra?: React.ReactNode;
  /** The page's `PlotExplainer`. */
  explainer: React.ReactNode;
  /** The page's one chart or table. */
  children: React.ReactNode;
}

export function RatePageLayout({
  projectId,
  title,
  subtitle,
  explore,
  dates,
  download,
  filterExtra,
  explainer,
  children,
}: RatePageLayoutProps) {
  const { scope } = explore;
  const { data: noSite } = useNoSiteDeployments(projectId);
  // Same query as the filter bar's, so this is a cache hit.
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
  });

  const save = () => {
    if (!download) return;
    const fileScope = trendCsvScope({
      dateFrom: scope.dateFrom,
      dateTo: scope.dateTo,
      firstDate: dates?.first,
      lastDate: dates?.last,
      siteIds: scope.siteIds,
      siteNames: buildSiteNameMap(explore.sites),
      labelIds: scope.labelTaxonomyIds,
      labelNames: filterOptions ? speciesLabelMap(filterOptions) : {},
    });
    saveTrendCsv(download.name, fileScope, download.csv(fileScope));
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-card/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
              <p className="text-sm text-muted-foreground">{subtitle}</p>
            </div>
            <DownloadCsvButton onClick={save} disabled={!download} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <NoSiteBanner
          projectId={projectId}
          count={noSite?.count ?? 0}
          reason="They have no site to count them under, so they are left out here."
        />
        <MissingDatesBanner projectId={projectId} />
        <ExploreFilterBar
          projectId={projectId}
          filters={explore.filters}
          scopeSiteIds={scope.siteIds}
          sites={explore.sites}
          update={explore.update}
          clearAll={explore.clearAll}
        >
          {filterExtra}
        </ExploreFilterBar>

        {explore.noSiteMatch ? (
          <Callout variant="info">
            No site has these tags. Pick other sites or tags to see results.
          </Callout>
        ) : !explore.ready ? (
          <p className="text-sm text-muted-foreground">Loading...</p>
        ) : (
          children
        )}

        {explainer}
      </main>
    </div>
  );
}
